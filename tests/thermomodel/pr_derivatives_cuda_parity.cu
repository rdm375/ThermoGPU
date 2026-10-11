#include "thermomodel/detail/pr_derivatives_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace thermomodel::detail;

namespace {

constexpr int nt = 401;
constexpr int nv = 301;
constexpr int count = nt * nv;

struct Input {
    double T;
    double v;
};

__global__ void evaluate_kernel(
    PRCoefficientParameters coefficients,
    const Input* inputs,
    PRPressureDerivatives* outputs,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        outputs[i] = pr_pressure_derivatives(
            coefficients,
            inputs[i].T,
            inputs[i].v
        );
    }
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const auto coefficients = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    const double Tmin = 100.0;
    const double Tmax = 1000.0;

    const double b = coefficients.b;
    const double vmin = b * (1.0 + 1e-8);
    const double vmax = 1000.0 * b;

    std::vector<Input> inputs(count);
    std::vector<PRPressureDerivatives> gpu(count);

    for (int i = 0; i < nt; ++i) {
        const double T =
            Tmin + (Tmax-Tmin)*i/(nt-1);

        for (int j = 0; j < nv; ++j) {
            const double fraction =
                static_cast<double>(j)/(nv-1);

            const double v =
                vmin * std::pow(vmax/vmin, fraction);

            inputs[i*nv+j] = {T, v};
        }
    }

    Input* device_inputs = nullptr;
    PRPressureDerivatives* device_outputs = nullptr;

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_inputs),
            sizeof(Input)*count
        ),
        "cudaMalloc(inputs)"
    );

    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(&device_outputs),
            sizeof(PRPressureDerivatives)*count
        ),
        "cudaMalloc(outputs)"
    );

    cuda_check(
        cudaMemcpy(
            device_inputs,
            inputs.data(),
            sizeof(Input)*count,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy(inputs)"
    );

    constexpr int threads = 256;
    const int blocks = (count + threads - 1)/threads;

    evaluate_kernel<<<blocks, threads>>>(
        coefficients,
        device_inputs,
        device_outputs,
        count
    );

    cuda_check(
        cudaGetLastError(),
        "evaluate_kernel launch"
    );

    cuda_check(
        cudaDeviceSynchronize(),
        "evaluate_kernel execution"
    );

    cuda_check(
        cudaMemcpy(
            gpu.data(),
            device_outputs,
            sizeof(PRPressureDerivatives)*count,
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(outputs)"
    );

    cuda_check(cudaFree(device_inputs), "cudaFree(inputs)");
    cuda_check(cudaFree(device_outputs), "cudaFree(outputs)");

    std::size_t comparisons = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;

    double max_absolute = 0.0;
    double max_scaled = 0.0;

    int worst_case = -1;
    int worst_quantity = -1;

    // Provisional diagnostic threshold, not a production tolerance.
    constexpr double tolerance = 1e-12;

    for (int i = 0; i < count; ++i) {
        const auto cpu = pr_pressure_derivatives(
            coefficients,
            inputs[i].T,
            inputs[i].v
        );

        const double reference[3] = {
            cpu.p, cpu.pv, cpu.pt
        };

        const double actual[3] = {
            gpu[i].p, gpu[i].pv, gpu[i].pt
        };

        for (int k = 0; k < 3; ++k) {
            ++comparisons;

            if (!std::isfinite(reference[k]) ||
                !std::isfinite(actual[k])) {
                ++nonfinite;
                continue;
            }

            const double absolute =
                std::abs(actual[k]-reference[k]);

            const double scale =
                std::max(1.0, std::abs(reference[k]));

            const double scaled = absolute/scale;

            max_absolute =
                std::max(max_absolute, absolute);

            if (scaled > max_scaled) {
                max_scaled = scaled;
                worst_case = i;
                worst_quantity = k;
            }

            if (scaled > tolerance)
                ++mismatches;
        }
    }

    const bool passed =
        mismatches == 0 && nonfinite == 0;

    std::cout << std::setprecision(17)
              << "PR derivative CUDA parity: "
              << (passed ? "PASS" : "FAIL") << '\n'
              << "Cases: " << count << '\n'
              << "Comparisons: " << comparisons << '\n'
              << "Value mismatches: " << mismatches << '\n'
              << "Nonfinite comparisons: " << nonfinite << '\n'
              << "Maximum absolute error: "
              << max_absolute << '\n'
              << "Maximum scaled error: "
              << max_scaled << '\n'
              << "Worst case: " << worst_case << '\n'
              << "Worst quantity (0=p, 1=pv, 2=pt): "
              << worst_quantity << '\n';

    if (worst_case >= 0) {
        std::cout << "Worst T: "
                  << inputs[worst_case].T << '\n'
                  << "Worst v: "
                  << inputs[worst_case].v << '\n';
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
