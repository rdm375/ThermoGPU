#include "thermogpu/thermo_batch_cuda.hpp"

#include "thermomodel/caloric.hpp"
#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thermogpu::thermo {
namespace {

namespace detail = thermomodel::detail;

void cuda_check(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        std::ostringstream message;
        message << operation << ": "
                << cudaGetErrorString(error);
        throw std::runtime_error(message.str());
    }
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count)
    {
        if (count != 0) {
            cuda_check(
                cudaMalloc(reinterpret_cast<void**>(&data_),
                           count * sizeof(T)),
                "cudaMalloc"
            );
        }
    }

    ~DeviceBuffer()
    {
        if (data_)
            cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

struct DeviceInput {
    double T;
    double p;
};

struct DeviceOutput {
    detail::PRTPResult result;
    int error;
};

enum DeviceError : int {
    Success = 0,
    InvalidInput = 1,
    OutOfRange = 2,
    NoPhysicalRoot = 3,
    MultipleRoots = 4,
    UnstableRoot = 5,
    NonphysicalProperties = 6,
    InvalidRootSelection = 7
};

__global__ void evaluate_kernel(
    detail::PRPolynomialStateParameters parameters,
    detail::PRTPRootPolicy policy,
    double Tmin,
    double Tmax,
    double pmin,
    double pmax,
    const DeviceInput* inputs,
    DeviceOutput* outputs,
    std::size_t count)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= count)
        return;

    const double T = inputs[i].T;
    const double p = inputs[i].p;

    DeviceOutput output{};
    output.error = InvalidInput;

    if (!(isfinite(T) && T > 0.0 &&
          isfinite(p) && p > 0.0)) {
        outputs[i] = output;
        return;
    }

    if (T < Tmin || T > Tmax ||
        p < pmin || p > pmax) {
        output.error = OutOfRange;
        outputs[i] = output;
        return;
    }

    output.result = detail::pr_evaluate_Tp_polynomial(
        parameters, T, p, policy
    );

    switch (output.result.status) {
        case detail::PRTPStatus::Success:
            output.error = Success;
            break;

        case detail::PRTPStatus::NoPhysicalRoot:
            output.error = NoPhysicalRoot;
            break;

        case detail::PRTPStatus::MultipleRoots:
            output.error = MultipleRoots;
            break;

        case detail::PRTPStatus::UnstableRoot:
            output.error = UnstableRoot;
            break;

        case detail::PRTPStatus::NonphysicalProperties:
            output.error = NonphysicalProperties;
            break;

        default:
            output.error = InvalidRootSelection;
            break;
    }

    outputs[i] = output;
}

detail::PRTPRootPolicy convert_policy(
    thermomodel::PRRootPolicy policy)
{
    switch (policy) {
        case thermomodel::PRRootPolicy::Vapor:
            return detail::PRTPRootPolicy::Vapor;

        case thermomodel::PRRootPolicy::Liquid:
            return detail::PRTPRootPolicy::Liquid;

        case thermomodel::PRRootPolicy::RejectMultiple:
            return detail::PRTPRootPolicy::RejectMultiple;
    }

    throw std::invalid_argument("invalid PR root policy");
}

detail::PRPolynomialStateParameters make_device_parameters(
    const CUDAPRParameters& parameters)
{
    const auto& pr = parameters.pr;

    detail::PolynomialCpKernel polynomial{};
    polynomial.scale = parameters.caloric_scale;

    for (std::size_t k = 0; k < 5; ++k)
        polynomial.a[k] = parameters.caloric_coefficients[k];

    return {
        {
            detail::make_pr_coefficients(
                pr.molar_mass,
                pr.Tc,
                pr.pc,
                pr.acentric
            ),
            pr.pref
        },
        polynomial,
        pr.Tref
    };
}

void translate_error(
    int error,
    thermomodel::ErrorCode& code,
    std::string& message)
{
    using thermomodel::ErrorCode;

    switch (error) {
        case Success:
            code = ErrorCode::None;
            message.clear();
            break;

        case InvalidInput:
            code = ErrorCode::InvalidInput;
            message = "invalid T,p";
            break;

        case OutOfRange:
            code = ErrorCode::OutOfRange;
            message = "outside PR domain";
            break;

        case NoPhysicalRoot:
            code = ErrorCode::NoConvergence;
            message = "no physical PR volume root";
            break;

        case MultipleRoots:
            code = ErrorCode::AmbiguousState;
            message =
                "multiple PR volume roots; specify root policy";
            break;

        case UnstableRoot:
            code = ErrorCode::AmbiguousState;
            message = "unstable or singular PR root";
            break;

        case NonphysicalProperties:
            code = ErrorCode::Unsupported;
            message =
                "nonphysical heat capacities or sound speed";
            break;

        default:
            code = ErrorCode::InvalidInput;
            message = "invalid PR root selection";
            break;
    }
}

BatchResult make_result(std::size_t n)
{
    BatchResult result;

    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    result.Z.assign(n, nan);
    result.rho.assign(n, nan);
    result.h.assign(n, nan);
    result.s.assign(n, nan);
    result.u.assign(n, nan);
    result.cp.assign(n, nan);
    result.cv.assign(n, nan);
    result.sound_speed.assign(n, nan);

    result.phase.assign(n, thermomodel::Phase::Unknown);
    result.error.assign(n, thermomodel::ErrorCode::None);
    result.message.resize(n);

    return result;
}

} // namespace

struct CUDAThermoWorkspace::Impl {
    explicit Impl(const CUDAPRParameters& configuration, std::size_t max_count)
        : capacity(max_count),
          device_inputs(max_count), device_outputs(max_count),
          host_inputs(max_count), host_outputs(max_count)
    {
        auto caloric = std::make_shared<thermomodel::PolynomialCp>(
            configuration.caloric_coefficients, configuration.caloric_scale);
        const thermomodel::PengRobinson validated(caloric, configuration.pr);
        pr = validated.parameters();
        policy = convert_policy(pr.root_policy);
        parameters = make_device_parameters(configuration);
        if (capacity > static_cast<std::size_t>(
                std::numeric_limits<unsigned int>::max()) * 128ULL)
            throw std::length_error("CUDA thermodynamic workspace too large");
    }

    std::size_t capacity;
    std::size_t count = 0;
    bool uploaded = false;
    bool evaluated = false;
    thermomodel::PRParameters pr{};
    detail::PRTPRootPolicy policy{};
    detail::PRPolynomialStateParameters parameters{};
    DeviceBuffer<DeviceInput> device_inputs;
    DeviceBuffer<DeviceOutput> device_outputs;
    std::vector<DeviceInput> host_inputs;
    mutable std::vector<DeviceOutput> host_outputs;
};

CUDAThermoWorkspace::CUDAThermoWorkspace(
    const CUDAPRParameters& parameters, std::size_t capacity)
    : impl_(std::make_unique<Impl>(parameters, capacity)) {}

CUDAThermoWorkspace::~CUDAThermoWorkspace() = default;
CUDAThermoWorkspace::CUDAThermoWorkspace(CUDAThermoWorkspace&&) noexcept = default;
CUDAThermoWorkspace& CUDAThermoWorkspace::operator=(CUDAThermoWorkspace&&) noexcept = default;

std::size_t CUDAThermoWorkspace::capacity() const noexcept {
    return impl_ ? impl_->capacity : 0;
}
std::size_t CUDAThermoWorkspace::size() const noexcept {
    return impl_ ? impl_->count : 0;
}

void CUDAThermoWorkspace::upload(const BatchInput& input) {
    if (!impl_) throw std::logic_error("moved-from CUDA workspace");
    if (input.temperature_K.size() != input.pressure_Pa.size())
        throw std::invalid_argument("thermodynamic batch T/p size mismatch");
    if (input.size() > impl_->capacity)
        throw std::length_error("CUDA thermodynamic workspace capacity exceeded");
    impl_->uploaded = false;
    impl_->evaluated = false;
    impl_->count = input.size();
    for (std::size_t i = 0; i < impl_->count; ++i)
        impl_->host_inputs[i] = {input.temperature_K[i], input.pressure_Pa[i]};
    if (impl_->count)
        cuda_check(cudaMemcpy(impl_->device_inputs.data(), impl_->host_inputs.data(),
                              impl_->count * sizeof(DeviceInput), cudaMemcpyHostToDevice),
                   "CUDA thermodynamic input transfer");
    impl_->uploaded = true;
}

void CUDAThermoWorkspace::evaluate() {
    if (!impl_ || !impl_->uploaded)
        throw std::logic_error("CUDA workspace requires upload before evaluate");
    impl_->evaluated = false;
    if (impl_->count) {
        constexpr unsigned int threads = 128;
        const auto blocks = static_cast<unsigned int>(
            impl_->count / threads + (impl_->count % threads != 0));
        evaluate_kernel<<<blocks, threads>>>(
            impl_->parameters, impl_->policy,
            impl_->pr.Tmin, impl_->pr.Tmax, impl_->pr.pmin, impl_->pr.pmax,
            impl_->device_inputs.data(), impl_->device_outputs.data(), impl_->count);
        cuda_check(cudaGetLastError(), "CUDA thermodynamic kernel launch");
        cuda_check(cudaDeviceSynchronize(), "CUDA thermodynamic kernel execution");
    }
    impl_->evaluated = true;
}

BatchResult CUDAThermoWorkspace::download() const {
    if (!impl_ || !impl_->evaluated)
        throw std::logic_error("CUDA workspace requires evaluate before download");
    const std::size_t n = impl_->count;
    BatchResult result = make_result(n);
    if (n)
        cuda_check(cudaMemcpy(impl_->host_outputs.data(), impl_->device_outputs.data(),
                              n * sizeof(DeviceOutput), cudaMemcpyDeviceToHost),
                   "CUDA thermodynamic output transfer");
    const auto& host_outputs = impl_->host_outputs;
    const auto& pr = impl_->pr;
    const double b = impl_->parameters.pr.coefficients.b;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& output = host_outputs[i];

        translate_error(
            output.error,
            result.error[i],
            result.message[i]
        );

        if (output.error != Success)
            continue;

        const auto& state = output.result.state;

        result.Z[i] = state.Z;
        result.rho[i] = state.rho;
        result.h[i] = state.h;
        result.s[i] = state.s;
        result.u[i] = state.u;
        result.cp[i] = state.cp;
        result.cv[i] = state.cv;
        result.sound_speed[i] = state.sound_speed;

        result.phase[i] =
            state.T >= pr.Tc
                ? thermomodel::Phase::Supercritical
                : (
                    output.result.selected_v > 3.0*b
                        ? thermomodel::Phase::Vapor
                        : thermomodel::Phase::Liquid
                  );
    }

    return result;
}

BatchResult evaluate_batch_cuda(
    const CUDAPRParameters& parameters, const BatchInput& input)
{
    if (input.temperature_K.size() != input.pressure_Pa.size())
        throw std::invalid_argument("thermodynamic batch T/p size mismatch");
    CUDAThermoWorkspace workspace(parameters, input.size());
    workspace.upload(input);
    workspace.evaluate();
    return workspace.download();
}

} // namespace thermogpu::thermo
