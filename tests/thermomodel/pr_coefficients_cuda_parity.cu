#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using thermomodel::detail::PRCoefficientParameters;
using thermomodel::detail::PRAttraction;
using thermomodel::detail::make_pr_coefficients;
using thermomodel::detail::pr_attraction;

__global__ void evaluate_pr_attraction(
    PRCoefficientParameters q,
    const double* temperatures,
    PRAttraction* results,
    int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count)
        results[i] = pr_attraction(q, temperatures[i]);
}

int main()
{
    constexpr int count = 801;

    const auto q = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    std::vector<double> temperatures(count);
    std::vector<PRAttraction> results(count);

    for (int i = 0; i < count; ++i)
        temperatures[i] = 200.0 + i;

    double* d_temperatures = nullptr;
    PRAttraction* d_results = nullptr;

    auto check = [](cudaError_t err) {
        if (err != cudaSuccess) {
            std::fprintf(
                stderr,
                "CUDA error: %s\n",
                cudaGetErrorString(err)
            );
            return false;
        }
        return true;
    };

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_temperatures),
            count * sizeof(double))))
        return 1;

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_results),
            count * sizeof(PRAttraction)))) {
        cudaFree(d_temperatures);
        return 1;
    }

    if (!check(cudaMemcpy(
            d_temperatures,
            temperatures.data(),
            count * sizeof(double),
            cudaMemcpyHostToDevice))) {
        cudaFree(d_temperatures);
        cudaFree(d_results);
        return 1;
    }

    evaluate_pr_attraction<<<4, 256>>>(
        q, d_temperatures, d_results, count);

    if (!check(cudaGetLastError()) ||
        !check(cudaDeviceSynchronize())) {
        cudaFree(d_temperatures);
        cudaFree(d_results);
        return 1;
    }

    if (!check(cudaMemcpy(
            results.data(),
            d_results,
            count * sizeof(PRAttraction),
            cudaMemcpyDeviceToHost))) {
        cudaFree(d_temperatures);
        cudaFree(d_results);
        return 1;
    }

    cudaFree(d_temperatures);
    cudaFree(d_results);

    double max_scaled_error = 0.0;
    double max_absolute_error = 0.0;
    double worst_temperature = 0.0;
    int worst_property = -1;
    int failures = 0;

    constexpr const char* names[] = {
        "a", "ap", "app"
    };

    for (int i = 0; i < count; ++i) {
        const double T = temperatures[i];

        const auto expected = pr_attraction(q, T);
        const auto actual = results[i];

        const double reference_values[] = {
            expected.a, expected.ap, expected.app
        };

        const double gpu_values[] = {
            actual.a, actual.ap, actual.app
        };

        for (int j = 0; j < 3; ++j) {
            const double absolute_error =
                std::abs(gpu_values[j] - reference_values[j]);

            const double scale =
                std::max(1.0, std::abs(reference_values[j]));

            const double scaled_error = absolute_error / scale;

            if (scaled_error > max_scaled_error) {
                max_scaled_error = scaled_error;
                worst_temperature = T;
                worst_property = j;
            }

            max_absolute_error =
                std::max(max_absolute_error, absolute_error);

            if (!std::isfinite(gpu_values[j]) ||
                scaled_error > 1e-12) {
                ++failures;

                if (failures <= 10) {
                    std::fprintf(
                        stderr,
                        "T=%.17g %s CPU=%.17g GPU=%.17g\n",
                        T,
                        names[j],
                        reference_values[j],
                        gpu_values[j]
                    );
                }
            }
        }
    }

    std::printf(
        "PR coefficient CUDA parity: %s\n"
        "States: %d\n"
        "Comparisons: %d\n"
        "Maximum absolute error: %.17g\n"
        "Maximum scaled error: %.17g\n"
        "Worst scaled error: T=%.17g property=%s\n",
        failures ? "FAIL" : "PASS",
        count,
        count * 3,
        max_absolute_error,
        max_scaled_error,
        worst_temperature,
        worst_property >= 0 ? names[worst_property] : "none"
    );

    return failures ? 1 : 0;
}
