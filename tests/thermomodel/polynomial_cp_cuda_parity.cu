#include "thermomodel/caloric.hpp"
#include "thermomodel/detail/polynomial_cp_kernel.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>

using thermomodel::detail::PolynomialCpKernel;

struct CaloricResult {
    double cp;
    double h;
    double s;
};

__global__ void evaluate_caloric(
    PolynomialCpKernel q,
    const double* temperatures,
    CaloricResult* results,
    int count,
    double Tref)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
        return;

    const double T = temperatures[i];

    results[i].cp =
        thermomodel::detail::polynomial_cp(q, T);

    results[i].h =
        thermomodel::detail::polynomial_enthalpy_delta(
            q, T, Tref);

    results[i].s =
        thermomodel::detail::polynomial_entropy_delta(
            q, T, Tref);
}

int main()
{
    constexpr int count = 801;
    constexpr double Tref = 298.15;

    constexpr std::array<double, 5> nasa = {
         5.14987613,
        -1.36709788e-2,
         4.91800599e-5,
        -4.84743026e-8,
         1.66693956e-11
    };

    constexpr double Ru = 8.31446261815324;
    constexpr double MW = 0.016043;

    PolynomialCpKernel q{};
    q.scale = 1000.0;

    double power = 1.0;

    std::array<double, 5> coefficients{};

    for (int k = 0; k < 5; ++k) {
        coefficients[k] =
            nasa[k] * (Ru / MW) * power;

        q.a[k] = coefficients[k];

        power *= q.scale;
    }

    thermomodel::PolynomialCp reference(
        coefficients, q.scale);

    double temperatures[count];
    CaloricResult results[count];

    for (int i = 0; i < count; ++i)
        temperatures[i] = 200.0 + i;

    double* d_temperatures = nullptr;
    CaloricResult* d_results = nullptr;

    auto check = [](cudaError_t err) {
        if (err != cudaSuccess) {
            std::fprintf(stderr, "CUDA: %s\n",
                         cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_temperatures),
            sizeof(temperatures))))
        return 1;

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_results),
            sizeof(results))))
        return 1;

    if (!check(cudaMemcpy(
            d_temperatures,
            temperatures,
            sizeof(temperatures),
            cudaMemcpyHostToDevice)))
        return 1;

    evaluate_caloric<<<4, 256>>>(
        q, d_temperatures, d_results, count, Tref);

    if (!check(cudaGetLastError()))
        return 1;

    if (!check(cudaDeviceSynchronize()))
        return 1;

    if (!check(cudaMemcpy(
            results,
            d_results,
            sizeof(results),
            cudaMemcpyDeviceToHost)))
        return 1;

    cudaFree(d_temperatures);
    cudaFree(d_results);

    double max_relative_error = 0.0;
    int failures = 0;

    for (int i = 0; i < count; ++i) {
        const double T = temperatures[i];

        const double expected[3] = {
            reference.cp(T),
            reference.enthalpy_delta(T, Tref),
            reference.entropy_thermal_delta(T, Tref)
        };

        const double actual[3] = {
            results[i].cp,
            results[i].h,
            results[i].s
        };

        // G2e.2: inspect CPU/GPU caloric rounding at failing states.
        if (T == 390.0 || T == 914.0) {
            std::printf(
                "\nG2e.2 caloric diagnostic T=%.17g\n"
                "cp CPU=%.17g GPU=%.17g delta=%.17g\n"
                "h  CPU=%.17g GPU=%.17g delta=%.17g\n"
                "s  CPU=%.17g GPU=%.17g delta=%.17g\n",
                T,
                expected[0], actual[0],
                actual[0] - expected[0],
                expected[1], actual[1],
                actual[1] - expected[1],
                expected[2], actual[2],
                actual[2] - expected[2]
            );
        }


        for (int j = 0; j < 3; ++j) {
            const double error =
                std::abs(actual[j] - expected[j]);

            const double scale =
                std::max(1.0, std::abs(expected[j]));

            const double relative_error = error / scale;

            max_relative_error =
                std::max(max_relative_error, relative_error);

            if (!std::isfinite(actual[j]) ||
                relative_error > 1e-12) {
                ++failures;

                if (failures <= 10) {
                    std::fprintf(
                        stderr,
                        "T=%.17g property=%d "
                        "expected=%.17g actual=%.17g\n",
                        T, j, expected[j], actual[j]);
                }
            }
        }
    }

    std::printf(
        "GPU caloric parity: %s\n"
        "States: %d\n"
        "Comparisons: %d\n"
        "Maximum scaled error: %.17g\n",
        failures ? "FAIL" : "PASS",
        count,
        count * 3,
        max_relative_error
    );

    return failures ? 1 : 0;
}
