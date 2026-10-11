#include "thermogpu/thermo_batch_cuda.hpp"

#include "thermomodel/caloric.hpp"
#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

using DeviceInput = CUDADeviceInput;
using DeviceOutput = CUDADeviceOutput;

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

__global__ void compact_output_kernel(
    const DeviceOutput* full,
    CUDACompactOutput* compact,
    std::size_t count)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= count)
        return;

    const DeviceOutput& src = full[i];

    CUDACompactOutput dst{};
    dst.error = src.error;

    if (src.error == Success) {
        dst.Z = src.result.state.Z;
        dst.rho = src.result.state.rho;
    } else {
        dst.Z = nan("");
        dst.rho = nan("");
    }

    compact[i] = dst;
}


__global__ void evaluate_compact_kernel(
    detail::PRPolynomialStateParameters parameters,
    detail::PRTPRootPolicy policy,
    double Tmin,
    double Tmax,
    double pmin,
    double pmax,
    const DeviceInput* inputs,
    CUDACompactOutput* outputs,
    std::size_t count)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (i >= count)
        return;

    const double T = inputs[i].T;
    const double p = inputs[i].p;

    CUDACompactOutput output{};
    output.error = InvalidInput;
    output.Z = nan("");
    output.rho = nan("");

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

    const auto result = detail::pr_evaluate_Tp_polynomial(
        parameters, T, p, policy
    );

    switch (result.status) {
        case detail::PRTPStatus::Success:
            output.error = Success;
            output.Z = result.state.Z;
            output.rho = result.state.rho;
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


// Diagnostic only. Unused properties are zero-initialized,
// NOT valid thermodynamic properties.
__global__ void evaluate_padded_kernel(
    detail::PRPolynomialStateParameters parameters,
    detail::PRTPRootPolicy policy,
    double Tmin, double Tmax, double pmin, double pmax,
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

    const auto result = detail::pr_evaluate_Tp_polynomial(
        parameters, T, p, policy);

    switch (result.status) {
        case detail::PRTPStatus::Success:
            output.error = Success;
            output.result.state.Z = result.state.Z;
            output.result.state.rho = result.state.rho;
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

void prepare_result(BatchResult& result, std::size_t n)
{
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

    for (auto& message : result.message)
        message.clear();
}

BatchResult make_result(std::size_t n)
{
    BatchResult result;
    prepare_result(result, n);
    return result;
}

} // namespace

struct CUDAThermoWorkspace::Impl {
    explicit Impl(const CUDAPRParameters& configuration, std::size_t max_count)
        : capacity(max_count),
          device_inputs(max_count), device_outputs(max_count),
          host_inputs(max_count), host_outputs(max_count),
          compact_outputs(max_count), host_compact(max_count),
          fused_outputs(max_count), host_fused(max_count),
          padded_outputs(max_count), host_padded(max_count)
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
    DeviceBuffer<CUDACompactOutput> compact_outputs;
    mutable std::vector<CUDACompactOutput> host_compact;
    DeviceBuffer<CUDACompactOutput> fused_outputs;
    mutable std::vector<CUDACompactOutput> host_fused;
    DeviceBuffer<DeviceOutput> padded_outputs;
    mutable std::vector<DeviceOutput> host_padded;
};

CUDAThermoWorkspace::CUDAThermoWorkspace(
    const CUDAPRParameters& parameters, std::size_t capacity)
    : impl_(std::make_unique<Impl>(parameters, capacity)) {}

CUDAThermoWorkspace::~CUDAThermoWorkspace() = default;
CUDAThermoWorkspace::CUDAThermoWorkspace(CUDAThermoWorkspace&&) noexcept = default;
CUDAThermoWorkspace& CUDAThermoWorkspace::operator=(CUDAThermoWorkspace&&) noexcept = default;

void CUDAThermoWorkspace::evaluate_device(
    const CUDADeviceInput* inputs,
    CUDADeviceOutput* outputs,
    std::size_t count,
    cudaStream_t stream) const
{
    if (!impl_) throw std::logic_error("moved-from CUDA workspace");
    if (count > impl_->capacity)
        throw std::length_error("CUDA thermodynamic workspace capacity exceeded");
    if (count == 0) return;
    if (!inputs || !outputs)
        throw std::invalid_argument("null CUDA thermodynamic device pointer");
    constexpr unsigned int threads = 128;
    const auto blocks = static_cast<unsigned int>(
        count / threads + (count % threads != 0));
    evaluate_kernel<<<blocks, threads, 0, stream>>>(
        impl_->parameters, impl_->policy,
        impl_->pr.Tmin, impl_->pr.Tmax, impl_->pr.pmin, impl_->pr.pmax,
        inputs, outputs, count);
    cuda_check(cudaGetLastError(), "CUDA thermodynamic device kernel launch");
}

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


CUDAThermoWorkspace::CompactProfile
CUDAThermoWorkspace::profile_compact() const
{
    if (!impl_ || !impl_->evaluated)
        throw std::logic_error(
            "profile_compact requires completed evaluate"
        );

    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };

    CompactProfile result{};
    const std::size_t n = impl_->count;

    result.count = n;
    result.full_record_bytes = sizeof(DeviceOutput);
    result.compact_record_bytes = sizeof(CUDACompactOutput);

    if (!n)
        return result;

    // Transfer complete output.
    auto t0 = Clock::now();
    cuda_check(
        cudaMemcpy(
            impl_->host_outputs.data(),
            impl_->device_outputs.data(),
            n * sizeof(DeviceOutput),
            cudaMemcpyDeviceToHost),
        "compact profile full D2H"
    );
    auto t1 = Clock::now();
    result.full_d2h_ms = ms(t0, t1);

    // GPU-side compact packing, timed with CUDA events.
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(cudaEventCreate(&start), "compact event create start");

    try {
        cuda_check(cudaEventCreate(&stop),
                   "compact event create stop");

        cuda_check(cudaEventRecord(start),
                   "compact event record start");

        constexpr unsigned int threads = 128;
        const unsigned int blocks = static_cast<unsigned int>(
            n / threads + (n % threads != 0)
        );

        compact_output_kernel<<<blocks, threads>>>(
            impl_->device_outputs.data(),
            impl_->compact_outputs.data(),
            n
        );

        cuda_check(cudaGetLastError(), "compact kernel launch");
        cuda_check(cudaEventRecord(stop),
                   "compact event record stop");
        cuda_check(cudaEventSynchronize(stop),
                   "compact kernel synchronization");

        float event_ms = 0.0f;
        cuda_check(
            cudaEventElapsedTime(&event_ms, start, stop),
            "compact kernel elapsed time"
        );

        result.pack_kernel_ms = event_ms;
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    // Transfer compact output.
    auto t2 = Clock::now();
    cuda_check(
        cudaMemcpy(
            impl_->host_compact.data(),
            impl_->compact_outputs.data(),
            n * sizeof(CUDACompactOutput),
            cudaMemcpyDeviceToHost),
        "compact profile compact D2H"
    );
    auto t3 = Clock::now();
    result.compact_d2h_ms = ms(t2, t3);

    // Compare host extraction of the two layouts.
    // Both loops perform the same checksum operation.
    double full_sum = 0.0;
    auto t4 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = impl_->host_outputs[i];
        if (x.error == Success)
            full_sum += x.result.state.Z + x.result.state.rho;
        full_sum += static_cast<double>(x.error);
    }
    auto t5 = Clock::now();
    result.full_extract_ms = ms(t4, t5);

    double compact_sum = 0.0;
    auto t6 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& x = impl_->host_compact[i];
        if (x.error == Success)
            compact_sum += x.Z + x.rho;
        compact_sum += static_cast<double>(x.error);
    }
    auto t7 = Clock::now();
    result.compact_extract_ms = ms(t6, t7);

    // Exact elementwise parity, including error codes and NaNs.
    auto equal_double = [](double a, double b) {
        return a == b || (std::isnan(a) && std::isnan(b));
    };

    const double nan_value =
        std::numeric_limits<double>::quiet_NaN();

    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = impl_->host_outputs[i];
        const auto& d = impl_->host_compact[i];

        const double expected_Z =
            a.error == Success ? a.result.state.Z : nan_value;
        const double expected_rho =
            a.error == Success ? a.result.state.rho : nan_value;

        if (a.error != d.error ||
            !equal_double(expected_Z, d.Z) ||
            !equal_double(expected_rho, d.rho))
            ++result.mismatches;
    }

    // Checksums are also expected to agree because traversal
    // order and arithmetic expressions are identical.
    if (!equal_double(full_sum, compact_sum))
        ++result.mismatches;

    result.checksum = compact_sum;
    return result;
}


CUDAThermoWorkspace::FusedProfile
CUDAThermoWorkspace::profile_fused(bool fused_first) const
{
    if (!impl_ || !impl_->uploaded)
        throw std::logic_error(
            "profile_fused requires uploaded inputs"
        );

    FusedProfile profile{};
    const std::size_t n = impl_->count;
    profile.count = n;

    if (n == 0)
        return profile;

    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        n / threads + (n % threads != 0)
    );

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(cudaEventCreate(&start), "fused event start");

    try {
        cuda_check(cudaEventCreate(&stop), "fused event stop");

        auto timed = [&](auto launch) -> double {
            cuda_check(cudaEventRecord(start), "event start");
            launch();
            cuda_check(cudaGetLastError(), "diagnostic kernel launch");
            cuda_check(cudaEventRecord(stop), "event stop");
            cuda_check(cudaEventSynchronize(stop), "event synchronize");

            float elapsed = 0.0f;
            cuda_check(
                cudaEventElapsedTime(&elapsed, start, stop),
                "event elapsed"
            );
            return elapsed;
        };

        auto reference = [&] {
            profile.reference_kernel_ms = timed([&] {
                evaluate_kernel<<<blocks, threads>>>(
                    impl_->parameters, impl_->policy,
                    impl_->pr.Tmin, impl_->pr.Tmax,
                    impl_->pr.pmin, impl_->pr.pmax,
                    impl_->device_inputs.data(),
                    impl_->device_outputs.data(), n
                );
            });

            profile.packing_kernel_ms = timed([&] {
                compact_output_kernel<<<blocks, threads>>>(
                    impl_->device_outputs.data(),
                    impl_->compact_outputs.data(), n
                );
            });
        };

        auto fused = [&] {
            profile.fused_kernel_ms = timed([&] {
                evaluate_compact_kernel<<<blocks, threads>>>(
                    impl_->parameters, impl_->policy,
                    impl_->pr.Tmin, impl_->pr.Tmax,
                    impl_->pr.pmin, impl_->pr.pmax,
                    impl_->device_inputs.data(),
                    impl_->fused_outputs.data(), n
                );
            });
        };

        if (fused_first) {
            fused();
            reference();
        } else {
            reference();
            fused();
        }

        cuda_check(
            cudaMemcpy(
                impl_->host_compact.data(),
                impl_->compact_outputs.data(),
                n * sizeof(CUDACompactOutput),
                cudaMemcpyDeviceToHost
            ),
            "reference compact D2H"
        );

        cuda_check(
            cudaMemcpy(
                impl_->host_fused.data(),
                impl_->fused_outputs.data(),
                n * sizeof(CUDACompactOutput),
                cudaMemcpyDeviceToHost
            ),
            "fused compact D2H"
        );

        auto equal = [](double a, double b) {
            return a == b || (std::isnan(a) && std::isnan(b));
        };

        double checksum = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& a = impl_->host_compact[i];
            const auto& b = impl_->host_fused[i];

            if (a.error != b.error ||
                !equal(a.Z, b.Z) ||
                !equal(a.rho, b.rho))
                ++profile.mismatches;

            checksum += static_cast<double>(b.error);
            if (b.error == Success)
                checksum += b.Z + b.rho;
        }

        profile.checksum = checksum;

    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    return profile;
}


CUDAThermoWorkspace::StorageProfile
CUDAThermoWorkspace::profile_storage(bool compact_first) const
{
    if (!impl_ || !impl_->uploaded)
        throw std::logic_error(
            "profile_storage requires uploaded inputs");

    StorageProfile profile{};
    const std::size_t n = impl_->count;

    profile.count = n;
    profile.full_record_bytes = sizeof(DeviceOutput);
    profile.compact_record_bytes = sizeof(CUDACompactOutput);

    if (!n)
        return profile;

    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        n / threads + (n % threads != 0));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_check(cudaEventCreate(&start), "storage event start");

    try {
        cuda_check(cudaEventCreate(&stop), "storage event stop");

        auto timed = [&](auto launch) -> double {
            cuda_check(cudaEventRecord(start), "storage start");
            launch();
            cuda_check(cudaGetLastError(), "storage kernel launch");
            cuda_check(cudaEventRecord(stop), "storage stop");
            cuda_check(cudaEventSynchronize(stop), "storage sync");

            float elapsed = 0.0f;
            cuda_check(
                cudaEventElapsedTime(&elapsed, start, stop),
                "storage elapsed");

            return elapsed;
        };

        auto full = [&] {
            profile.full_ms = timed([&] {
                evaluate_kernel<<<blocks, threads>>>(
                    impl_->parameters, impl_->policy,
                    impl_->pr.Tmin, impl_->pr.Tmax,
                    impl_->pr.pmin, impl_->pr.pmax,
                    impl_->device_inputs.data(),
                    impl_->device_outputs.data(), n);
            });
        };

        auto padded = [&] {
            profile.padded_ms = timed([&] {
                evaluate_padded_kernel<<<blocks, threads>>>(
                    impl_->parameters, impl_->policy,
                    impl_->pr.Tmin, impl_->pr.Tmax,
                    impl_->pr.pmin, impl_->pr.pmax,
                    impl_->device_inputs.data(),
                    impl_->padded_outputs.data(), n);
            });
        };

        auto compact = [&] {
            profile.compact_ms = timed([&] {
                evaluate_compact_kernel<<<blocks, threads>>>(
                    impl_->parameters, impl_->policy,
                    impl_->pr.Tmin, impl_->pr.Tmax,
                    impl_->pr.pmin, impl_->pr.pmax,
                    impl_->device_inputs.data(),
                    impl_->fused_outputs.data(), n);
            });
        };

        if (compact_first) {
            compact();
            padded();
            full();
        } else {
            full();
            padded();
            compact();
        }

        cuda_check(
            cudaMemcpy(
                impl_->host_outputs.data(),
                impl_->device_outputs.data(),
                n * sizeof(DeviceOutput),
                cudaMemcpyDeviceToHost),
            "storage full D2H");

        cuda_check(
            cudaMemcpy(
                impl_->host_padded.data(),
                impl_->padded_outputs.data(),
                n * sizeof(DeviceOutput),
                cudaMemcpyDeviceToHost),
            "storage padded D2H");

        cuda_check(
            cudaMemcpy(
                impl_->host_fused.data(),
                impl_->fused_outputs.data(),
                n * sizeof(CUDACompactOutput),
                cudaMemcpyDeviceToHost),
            "storage compact D2H");

        auto equal = [](double a, double b) {
            return a == b ||
                   (std::isnan(a) && std::isnan(b));
        };

        const double nan =
            std::numeric_limits<double>::quiet_NaN();

        for (std::size_t i = 0; i < n; ++i) {
            const auto& a = impl_->host_outputs[i];
            const auto& b = impl_->host_padded[i];
            const auto& c = impl_->host_fused[i];

            const double expected_z =
                a.error == Success ? a.result.state.Z : nan;

            const double expected_rho =
                a.error == Success ? a.result.state.rho : nan;

            if (a.error != b.error ||
                a.error != c.error ||
                !equal(expected_z, c.Z) ||
                !equal(expected_rho, c.rho) ||
                (a.error == Success &&
                 (!equal(expected_z, b.result.state.Z) ||
                  !equal(expected_rho, b.result.state.rho))))
                ++profile.mismatches;

            profile.checksum += static_cast<double>(c.error);

            if (c.error == Success)
                profile.checksum += c.Z + c.rho;
        }

    } catch (...) {
        if (stop)
            cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    return profile;
}

void CUDAThermoWorkspace::download_into(
    BatchResult& result) const
{
    if (!impl_ || !impl_->evaluated)
        throw std::logic_error(
            "CUDA workspace requires evaluate before download"
        );

    const std::size_t n = impl_->count;

    prepare_result(result, n);

    if (n) {
        cuda_check(
            cudaMemcpy(
                impl_->host_outputs.data(),
                impl_->device_outputs.data(),
                n * sizeof(DeviceOutput),
                cudaMemcpyDeviceToHost
            ),
            "CUDA thermodynamic output transfer"
        );
    }

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
}

BatchResult CUDAThermoWorkspace::download() const
{
    BatchResult result;
    download_into(result);
    return result;
}

CUDAThermoWorkspace::Profile CUDAThermoWorkspace::profile(const BatchInput& input) {
    if (!impl_) throw std::logic_error("moved-from CUDA workspace");
    if (input.temperature_K.size() != input.pressure_Pa.size())
        throw std::invalid_argument("thermodynamic batch T/p size mismatch");
    if (input.size() > impl_->capacity)
        throw std::length_error("CUDA thermodynamic workspace capacity exceeded");
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    Profile profile{};
    impl_->uploaded = false;
    impl_->evaluated = false;
    impl_->count = input.size();
    const auto n = impl_->count;
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i)
        impl_->host_inputs[i] = {input.temperature_K[i], input.pressure_Pa[i]};
    auto t1 = Clock::now();
    profile.pack_ms = ms(t0, t1);
    if (n) cuda_check(cudaMemcpy(impl_->device_inputs.data(), impl_->host_inputs.data(),
                                 n*sizeof(DeviceInput), cudaMemcpyHostToDevice),
                      "profile input transfer");
    auto t2 = Clock::now();
    profile.h2d_ms = ms(t1, t2);
    impl_->uploaded = true;

    // Record on the default stream used by evaluate(); the kernel and
    // both events are ordered in that stream. CUDA events are created
    // outside the measured interval.
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "profile event start create");
    try {
        cuda_check(cudaEventCreate(&stop), "profile event stop create");
        if (n) {
            cuda_check(cudaEventRecord(start), "profile event start");
            constexpr unsigned int threads = 128;
            const auto blocks = static_cast<unsigned int>(n/threads + (n%threads != 0));
            evaluate_kernel<<<blocks, threads>>>(
                impl_->parameters, impl_->policy,
                impl_->pr.Tmin, impl_->pr.Tmax, impl_->pr.pmin, impl_->pr.pmax,
                impl_->device_inputs.data(), impl_->device_outputs.data(), n);
            cuda_check(cudaGetLastError(), "profile kernel launch");
            cuda_check(cudaEventRecord(stop), "profile event stop");
            cuda_check(cudaEventSynchronize(stop), "profile kernel sync");
            float elapsed = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed, start, stop), "profile kernel event time");
            profile.kernel_event_ms = elapsed;
        }
        auto t3 = Clock::now();
        profile.kernel_wall_ms = ms(t2, t3);
        impl_->evaluated = true;
        if (n) cuda_check(cudaMemcpy(impl_->host_outputs.data(), impl_->device_outputs.data(),
                                     n*sizeof(DeviceOutput), cudaMemcpyDeviceToHost),
                          "profile output transfer");
        auto t4 = Clock::now();
        profile.d2h_ms = ms(t3, t4);
        // download() would copy device results a second time. Instead,
        // convert the already transferred output directly.
        // G4.5c.2c.5a: conversion substage profiling.
        // Keep the full BatchResult semantics unchanged.
        auto init_begin = Clock::now();
        profile.result = make_result(n);
        auto init_end = Clock::now();
        profile.result_init_ms = ms(init_begin, init_end);

        auto error_begin = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& output = impl_->host_outputs[i];
            translate_error(
                output.error,
                profile.result.error[i],
                profile.result.message[i]
            );
        }
        auto error_end = Clock::now();
        profile.error_translation_ms = ms(error_begin, error_end);

        const auto& pr = impl_->pr;
        const double b = impl_->parameters.pr.coefficients.b;

        auto numeric_begin = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& output = impl_->host_outputs[i];

            if (output.error != Success)
                continue;

            const auto& state = output.result.state;

            profile.result.Z[i] = state.Z;
            profile.result.rho[i] = state.rho;
            profile.result.h[i] = state.h;
            profile.result.s[i] = state.s;
            profile.result.u[i] = state.u;
            profile.result.cp[i] = state.cp;
            profile.result.cv[i] = state.cv;
            profile.result.sound_speed[i] = state.sound_speed;

            profile.result.phase[i] =
                state.T >= pr.Tc
                    ? thermomodel::Phase::Supercritical
                    : (
                        output.result.selected_v > 3.0*b
                            ? thermomodel::Phase::Vapor
                            : thermomodel::Phase::Liquid
                      );
        }
        auto numeric_end = Clock::now();
        profile.numeric_phase_ms = ms(numeric_begin, numeric_end);

        auto t5 = Clock::now();
        profile.conversion_ms = ms(t4, t5);
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }
    if (stop) cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return profile;
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
