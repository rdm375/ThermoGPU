#include "thermogpu/thermo_batch_cuda.hpp"

#include "thermomodel/caloric.hpp"
#include "thermomodel/peng_robinson.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using thermogpu::thermo::BatchInput;
using thermogpu::thermo::BatchResult;
using thermogpu::thermo::CUDAPRParameters;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double scaled_error(double a, double b)
{
    return std::fabs(a-b) /
        std::max({1.0, std::fabs(a), std::fabs(b)});
}

struct Statistics {
    std::array<double, 8> maximum_error{};
    std::size_t comparisons = 0;
    std::size_t violations = 0;
    std::size_t status_mismatches = 0;
    std::size_t phase_mismatches = 0;
    std::size_t message_mismatches = 0;
};

void compare(
    const BatchResult& cpu,
    const BatchResult& gpu,
    Statistics& statistics,
    double tolerance)
{
    require(cpu.size() == gpu.size(),
            "CPU/CUDA batch size mismatch");

    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (cpu.error[i] != gpu.error[i]) {
            ++statistics.status_mismatches;
            continue;
        }

        if (cpu.message[i] != gpu.message[i])
            ++statistics.message_mismatches;

        if (cpu.phase[i] != gpu.phase[i])
            ++statistics.phase_mismatches;

        if (!cpu.success(i))
            continue;

        const double reference[8] = {
            cpu.Z[i], cpu.rho[i],
            cpu.h[i], cpu.s[i], cpu.u[i],
            cpu.cp[i], cpu.cv[i],
            cpu.sound_speed[i]
        };

        const double actual[8] = {
            gpu.Z[i], gpu.rho[i],
            gpu.h[i], gpu.s[i], gpu.u[i],
            gpu.cp[i], gpu.cv[i],
            gpu.sound_speed[i]
        };

        for (std::size_t k = 0; k < 8; ++k) {
            ++statistics.comparisons;

            const double error =
                scaled_error(reference[k], actual[k]);

            if (std::isfinite(error))
                statistics.maximum_error[k] =
                    std::max(
                        statistics.maximum_error[k],
                        error
                    );

            if (!(error <= tolerance))
                ++statistics.violations;
        }
    }
}

void run_case(
    const CUDAPRParameters& parameters,
    const BatchInput& input,
    bool constant_cp_reference,
    Statistics& statistics)
{
    std::shared_ptr<const thermomodel::CaloricModel> caloric;

    if (constant_cp_reference) {
        caloric =
            std::make_shared<thermomodel::ConstantCp>(
                parameters.caloric_coefficients[0]
            );
    } else {
        caloric =
            std::make_shared<thermomodel::PolynomialCp>(
                parameters.caloric_coefficients,
                parameters.caloric_scale
            );
    }

    const thermomodel::PengRobinson model(
        caloric, parameters.pr
    );

    const auto cpu =
        thermogpu::thermo::evaluate_batch_scalar(
            model, input
        );

    const auto gpu =
        thermogpu::thermo::evaluate_batch_cuda(
            parameters, input
        );

    compare(cpu, gpu, statistics, 1e-12);
}

BatchInput make_input()
{
    BatchInput input;

    constexpr int nt = 81;
    constexpr int np = 81;

    input.temperature_K.reserve(nt*np + 4);
    input.pressure_Pa.reserve(nt*np + 4);

    for (int i = 0; i < nt; ++i) {
        const double T =
            100.0 + (1200.0-100.0)*i/(nt-1);

        for (int j = 0; j < np; ++j) {
            const double fraction =
                static_cast<double>(j)/(np-1);

            const double p =
                100.0*std::pow(1.0e6, fraction);

            input.temperature_K.push_back(T);
            input.pressure_Pa.push_back(p);
        }
    }

    input.temperature_K.push_back(-1.0);
    input.pressure_Pa.push_back(1.0e6);

    input.temperature_K.push_back(300.0);
    input.pressure_Pa.push_back(-1.0);

    input.temperature_K.push_back(50.0);
    input.pressure_Pa.push_back(1.0e6);

    input.temperature_K.push_back(
        std::numeric_limits<double>::quiet_NaN()
    );
    input.pressure_Pa.push_back(1.0e6);

    return input;
}

void print_statistics(const Statistics& statistics)
{
    constexpr const char* names[8] = {
        "Z", "rho", "h", "s",
        "u", "cp", "cv", "sound_speed"
    };

    std::cout << std::setprecision(17);

    std::cout << "CUDA thermodynamic comparisons: "
              << statistics.comparisons << '\n';

    for (std::size_t k = 0; k < 8; ++k) {
        std::cout << "maximum scaled " << names[k]
                  << " error: "
                  << statistics.maximum_error[k] << '\n';
    }

    std::cout << "status mismatches: "
              << statistics.status_mismatches << '\n';

    std::cout << "message mismatches: "
              << statistics.message_mismatches << '\n';

    std::cout << "phase mismatches: "
              << statistics.phase_mismatches << '\n';

    std::cout << "tolerance violations: "
              << statistics.violations << '\n';
}

} // namespace


// G4.6c.1: validate reusable caller-owned BatchResult storage.
void test_download_into_reuse(const CUDAPRParameters& parameters)
{
    using thermogpu::thermo::CUDAThermoWorkspace;

    constexpr std::size_t capacity = 8;
    CUDAThermoWorkspace workspace(parameters, capacity);
    BatchResult reused;

    auto check_numeric = [](double a, double b) {
        return (std::isnan(a) && std::isnan(b)) || a == b;
    };

    auto check_result = [&](const BatchResult& expected,
                            const BatchResult& actual) {
        require(expected.size() == actual.size(),
                "reuse result size mismatch");

        const std::size_t n = expected.size();

        require(actual.rho.size() == n &&
                actual.h.size() == n &&
                actual.s.size() == n &&
                actual.u.size() == n &&
                actual.cp.size() == n &&
                actual.cv.size() == n &&
                actual.sound_speed.size() == n &&
                actual.phase.size() == n &&
                actual.error.size() == n &&
                actual.message.size() == n,
                "reuse result vector size mismatch");

        for (std::size_t i = 0; i < n; ++i) {
            require(expected.error[i] == actual.error[i],
                    "reuse error mismatch");
            require(expected.phase[i] == actual.phase[i],
                    "reuse phase mismatch");
            require(expected.message[i] == actual.message[i],
                    "reuse message mismatch");

            const double reference[8] = {
                expected.Z[i], expected.rho[i],
                expected.h[i], expected.s[i],
                expected.u[i], expected.cp[i],
                expected.cv[i], expected.sound_speed[i]
            };

            const double values[8] = {
                actual.Z[i], actual.rho[i],
                actual.h[i], actual.s[i],
                actual.u[i], actual.cp[i],
                actual.cv[i], actual.sound_speed[i]
            };

            for (std::size_t j = 0; j < 8; ++j)
                require(check_numeric(reference[j], values[j]),
                        "reuse numeric property mismatch");
        }
    };

    auto make_case = [](std::initializer_list<std::pair<double, double>> states) {
        BatchInput input;
        for (const auto& state : states) {
            input.temperature_K.push_back(state.first);
            input.pressure_Pa.push_back(state.second);
        }
        return input;
    };

    const auto valid = make_case({
        {300.0, 5.0e6},
        {320.0, 8.0e6},
        {350.0, 3.0e6},
        {400.0, 1.0e6}
    });

    const auto invalid = make_case({
        {-1.0, 5.0e6},
        {300.0, -1.0},
        {std::numeric_limits<double>::quiet_NaN(), 1.0e6},
        {50.0, 1.0e6}
    });

    const auto mixed = make_case({
        {300.0, 5.0e6},
        {-1.0, 5.0e6},
        {320.0, 8.0e6},
        {300.0, -1.0},
        {350.0, 3.0e6},
        {50.0, 1.0e6},
        {400.0, 1.0e6},
        {std::numeric_limits<double>::quiet_NaN(), 1.0e6}
    });

    auto run = [&](const BatchInput& input) {
        workspace.upload(input);
        workspace.evaluate();

        const BatchResult fresh = workspace.download();
        workspace.download_into(reused);

        check_result(fresh, reused);

        // A second download into the same storage must remain identical.
        workspace.download_into(reused);
        check_result(fresh, reused);

        return fresh;
    };

    const auto first = run(valid);
    require(first.success(0), "expected valid initial state");

    const auto second = run(invalid);
    require(!second.success(0), "expected invalid replacement state");

    for (std::size_t i = 0; i < second.size(); ++i) {
        require(!second.success(i), "invalid case unexpectedly succeeded");

        require(std::isnan(reused.Z[i]) &&
                std::isnan(reused.rho[i]) &&
                std::isnan(reused.h[i]) &&
                std::isnan(reused.s[i]) &&
                std::isnan(reused.u[i]) &&
                std::isnan(reused.cp[i]) &&
                std::isnan(reused.cv[i]) &&
                std::isnan(reused.sound_speed[i]),
                "stale numeric value after failure");

        require(reused.phase[i] == thermomodel::Phase::Unknown,
                "stale phase after failure");
    }

    const auto third = run(mixed);
    require(third.size() == capacity, "mixed batch size mismatch");
    require(third.success(0), "valid state did not recover");

    const auto fourth = run(valid);
    require(fourth.size() == valid.size(),
            "reused output did not shrink");
    require(fourth.success(0), "valid state failed after reuse");

    for (const auto& message : reused.message)
        require(message.empty(), "stale diagnostic message after recovery");

    std::cout << "CUDA BatchResult reuse parity PASS\n";
}

// Validate caller-owned device buffers and a non-default CUDA stream.
void test_device_resident(const thermogpu::thermo::CUDAPRParameters& parameters)
{
    using thermogpu::thermo::CUDADeviceInput;
    using thermogpu::thermo::CUDADeviceOutput;
    using thermogpu::thermo::CUDAThermoWorkspace;
    constexpr std::size_t n = 4;
    const std::array<CUDADeviceInput, n> inputs{{
        {300.0, 5.0e6}, {320.0, 8.0e6},
        {-1.0, 5.0e6}, {300.0, 2.0e8}
    }};
    CUDADeviceInput* d_inputs = nullptr;
    CUDADeviceOutput* d_outputs = nullptr;
    cudaStream_t stream = nullptr;
    auto check = [](cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    };
    try {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMalloc(reinterpret_cast<void**>(&d_inputs), n * sizeof(*d_inputs)));
        check(cudaMalloc(reinterpret_cast<void**>(&d_outputs), n * sizeof(*d_outputs)));
        check(cudaMemcpyAsync(d_inputs, inputs.data(), n * sizeof(*d_inputs),
                              cudaMemcpyHostToDevice, stream));
        CUDAThermoWorkspace workspace(parameters, n);
        workspace.evaluate_device(d_inputs, d_outputs, n, stream);
        std::array<CUDADeviceOutput, n> outputs{};
        check(cudaMemcpyAsync(outputs.data(), d_outputs, n * sizeof(*d_outputs),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));

        BatchInput reference_input;
        for (const auto& input : inputs) {
            reference_input.temperature_K.push_back(input.T);
            reference_input.pressure_Pa.push_back(input.p);
        }
        const auto reference = thermogpu::thermo::evaluate_batch_cuda(
            parameters, reference_input);
        for (std::size_t i = 0; i < n; ++i) {
            require((outputs[i].error == 0) == reference.success(i),
                    "device-resident error mismatch");
            if (!reference.success(i)) continue;
            const auto& state = outputs[i].result.state;
            const double expected[] = {
                reference.Z[i], reference.rho[i], reference.h[i], reference.s[i],
                reference.u[i], reference.cp[i], reference.cv[i],
                reference.sound_speed[i]
            };
            const double actual[] = {
                state.Z, state.rho, state.h, state.s, state.u,
                state.cp, state.cv, state.sound_speed
            };
            for (std::size_t j = 0; j < 8; ++j)
                require(scaled_error(expected[j], actual[j]) <= 1e-12,
                        "device-resident property mismatch");
        }
        bool rejected = false;
        try { workspace.evaluate_device(d_inputs, d_outputs, n + 1, stream); }
        catch (const std::length_error&) { rejected = true; }
        require(rejected, "device-resident capacity overflow accepted");
        workspace.evaluate_device(nullptr, nullptr, 0, stream);
        rejected = false;
        try { workspace.evaluate_device(nullptr, d_outputs, 1, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "device-resident null input accepted");
    } catch (...) {
        if (stream) cudaStreamSynchronize(stream);
        if (d_outputs) cudaFree(d_outputs);
        if (d_inputs) cudaFree(d_inputs);
        if (stream) cudaStreamDestroy(stream);
        throw;
    }
    if (d_outputs) check(cudaFree(d_outputs));
    if (d_inputs) check(cudaFree(d_inputs));
    if (stream) check(cudaStreamDestroy(stream));
}

int main()
{
    try {
        CUDAPRParameters parameters;

        const BatchInput input = make_input();

        Statistics statistics;

        const thermomodel::PRRootPolicy policies[] = {
            thermomodel::PRRootPolicy::Vapor,
            thermomodel::PRRootPolicy::Liquid,
            thermomodel::PRRootPolicy::RejectMultiple
        };

        for (const auto policy : policies) {
            parameters.pr.root_policy = policy;

            run_case(
                parameters, input, true, statistics
            );
        }

        // Nonconstant polynomial caloric model.
        parameters.caloric_coefficients = {
            2200.0, 100.0, 20.0, 0.0, 0.0
        };

        for (const auto policy : policies) {
            parameters.pr.root_policy = policy;

            run_case(
                parameters, input, false, statistics
            );
        }

        // Empty batch.
        const BatchInput empty{};

        const auto empty_result =
            thermogpu::thermo::evaluate_batch_cuda(
                parameters, empty
            );

        require(empty_result.size() == 0,
                "empty CUDA batch failed");

        // Mismatched input dimensions.
        BatchInput invalid;
        invalid.temperature_K = {300.0};
        invalid.pressure_Pa = {};

        bool rejected = false;

        try {
            (void)thermogpu::thermo::evaluate_batch_cuda(
                parameters, invalid
            );
        } catch (const std::invalid_argument&) {
            rejected = true;
        }

        require(rejected,
                "mismatched CUDA batch was accepted");

        // G4.6c.1: reusable output parity and failure transitions.
        test_download_into_reuse(parameters);

        // Persistent allocation: repeated uploads of different sizes.
        thermogpu::thermo::CUDAThermoWorkspace workspace(parameters, input.size());
        require(workspace.capacity() == input.size(), "workspace capacity mismatch");
        bool premature_download = false;
        try { (void)workspace.download(); }
        catch (const std::logic_error&) { premature_download = true; }
        require(premature_download, "download before evaluate accepted");

        for (std::size_t count : {std::size_t{0}, std::size_t{1},
                                  std::size_t{97}, input.size(), std::size_t{3}}) {
            BatchInput subset;
            subset.temperature_K.assign(input.temperature_K.begin(),
                                        input.temperature_K.begin() + count);
            subset.pressure_Pa.assign(input.pressure_Pa.begin(),
                                      input.pressure_Pa.begin() + count);
            workspace.upload(subset);
            require(workspace.size() == count, "workspace size mismatch");
            workspace.evaluate();
            auto repeated = workspace.download();
            auto reference = thermogpu::thermo::evaluate_batch_cuda(parameters, subset);
            compare(reference, repeated, statistics, 1e-12);
            // Multiple downloads must remain valid.
            auto again = workspace.download();
            compare(reference, again, statistics, 1e-12);
        }

        bool rejected_capacity = false;
        BatchInput too_large;
        too_large.temperature_K.assign(input.size() + 1, 300.0);
        too_large.pressure_Pa.assign(input.size() + 1, 1.0e6);
        try { workspace.upload(too_large); }
        catch (const std::length_error&) { rejected_capacity = true; }
        require(rejected_capacity, "workspace capacity overflow accepted");

        test_device_resident(parameters);

        print_statistics(statistics);

        require(statistics.status_mismatches == 0,
                "CUDA status parity failure");

        require(statistics.message_mismatches == 0,
                "CUDA message parity failure");

        require(statistics.phase_mismatches == 0,
                "CUDA phase parity failure");

        require(statistics.violations == 0,
                "CUDA property parity failure");

        std::cout
            << "CUDA thermodynamic batch parity PASS\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
