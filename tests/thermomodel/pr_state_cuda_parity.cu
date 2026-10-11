#include "thermomodel/detail/pr_state_kernel.hpp"
#include "thermomodel/detail/pr_polynomial_state_kernel.hpp"
#include "thermomodel/detail/polynomial_cp_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace thermomodel::detail;

struct Input {
    double T;
    double v;
    double p;
    PRIdealCaloricState caloric;
};

__global__ void evaluate_kernel(
    PRStateKernelParameters parameters,
    const Input* inputs,
    PRStateKernelResult* outputs,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    const auto& input = inputs[i];

    outputs[i] = pr_evaluate_Tvp(
        parameters,
        input.caloric,
        input.T,
        input.v,
        input.p
    );
}

__global__ void evaluate_composed_kernel(
    PRPolynomialStateParameters parameters,
    const Input* inputs,
    PRStateKernelResult* outputs,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    const auto& input = inputs[i];

    outputs[i] = pr_evaluate_Tvp_polynomial(
        parameters,
        input.T,
        input.v,
        input.p
    );
}

static bool check(cudaError_t error, const char* operation)
{
    if (error == cudaSuccess)
        return true;

    std::cerr << operation << ": "
              << cudaGetErrorString(error) << '\n';

    return false;
}

int main()
{
    const auto coefficients = make_pr_coefficients(
        0.016043, 190.56, 4599200.0, 0.01142
    );

    constexpr double Tref = 298.15;
    constexpr double pref = 101325.0;

    constexpr double nasa[5] = {
         5.14987613,
        -1.36709788e-2,
         4.91800599e-5,
        -4.84743026e-8,
         1.66693956e-11
    };

    PolynomialCpKernel polynomial{};
    polynomial.scale = 1000.0;

    double power = 1.0;

    for (int k = 0; k < 5; ++k) {
        polynomial.a[k] =
            nasa[k] * coefficients.R * power;

        power *= polynomial.scale;
    }

    const PRStateKernelParameters parameters{
        coefficients, pref
    };

    const PRPolynomialStateParameters composed_parameters{
        parameters, polynomial, Tref
    };

    constexpr int nt = 401;
    constexpr int nv = 301;
    constexpr int n = nt * nv;

    std::vector<Input> inputs(n);
    std::vector<PRStateKernelResult> cpu(n);
    std::vector<PRStateKernelResult> gpu(n);
    std::vector<PRStateKernelResult> gpu_composed(n);

    const double b = coefficients.b;
    const double vmin = b * (1.0 + 1e-8);
    const double vmax = 1000.0 * b;

    for (int i = 0; i < nt; ++i) {
        const double T =
            200.0 + (1000.0 - 200.0) * i / (nt - 1);

        const PRIdealCaloricState caloric{
            polynomial_cp(polynomial, T),
            polynomial_enthalpy_delta(
                polynomial, T, Tref
            ),
            polynomial_entropy_delta(
                polynomial, T, Tref
            )
        };

        for (int j = 0; j < nv; ++j) {
            const int k = i * nv + j;

            const double fraction =
                static_cast<double>(j) / (nv - 1);

            const double v =
                vmin * std::pow(vmax / vmin, fraction);

            const double p =
                pr_pressure_derivatives(
                    coefficients, T, v
                ).p;

            inputs[k] = {T, v, p, caloric};

            cpu[k] = pr_evaluate_Tvp(
                parameters, caloric, T, v, p
            );
        }
    }

    Input* d_inputs = nullptr;
    PRStateKernelResult* d_outputs = nullptr;
    PRStateKernelResult* d_composed = nullptr;

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_inputs),
            n * sizeof(Input)),
            "cudaMalloc inputs"))
        return 1;

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_outputs),
            n * sizeof(PRStateKernelResult)),
            "cudaMalloc outputs"))
        return 1;

    if (!check(cudaMemcpy(
            d_inputs, inputs.data(),
            n * sizeof(Input),
            cudaMemcpyHostToDevice),
            "cudaMemcpy inputs"))
        return 1;

    if (!check(cudaMalloc(
            reinterpret_cast<void**>(&d_composed),
            n * sizeof(PRStateKernelResult)),
            "cudaMalloc composed outputs"))
        return 1;

    constexpr int threads = 128;
    const int blocks = (n + threads - 1) / threads;

    evaluate_kernel<<<blocks, threads>>>(
        parameters, d_inputs, d_outputs, n
    );

    if (!check(cudaGetLastError(), "kernel launch"))
        return 1;

    if (!check(cudaDeviceSynchronize(), "kernel execution"))
        return 1;

    evaluate_composed_kernel<<<blocks, threads>>>(
        composed_parameters,d_inputs,d_composed,n
    );

    if (!check(cudaGetLastError(), "composed kernel launch"))
        return 1;

    if (!check(cudaDeviceSynchronize(), "composed kernel execution"))
        return 1;

    if (!check(cudaMemcpy(
            gpu.data(), d_outputs,
            n * sizeof(PRStateKernelResult),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy outputs"))
        return 1;

    if (!check(cudaMemcpy(
            gpu_composed.data(), d_composed,
            n * sizeof(PRStateKernelResult),
            cudaMemcpyDeviceToHost),
            "cudaMemcpy composed outputs"))
        return 1;

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_composed);

    std::size_t comparisons = 0;
    std::size_t violations = 0;
    std::size_t nonfinite = 0;
    std::size_t status_mismatches = 0;

    std::size_t composed_comparisons = 0;
    std::size_t composed_violations = 0;
    std::size_t composed_nonfinite = 0;
    std::size_t composed_status_mismatches = 0;
    double composed_max_scaled = 0.0;
    double composed_max_absolute = 0.0;
    std::size_t composed_quantity_violations[16]{};
    double composed_quantity_max_scaled[16]{};

    double max_absolute = 0.0;
    double max_scaled = 0.0;

    int worst_case = -1;
    int worst_quantity = -1;

    constexpr const char* names[] = {
        "T", "p", "rho", "Z",
        "u", "h", "s",
        "cv", "cp", "sound_speed",
        "drho_dp_T", "drho_dT_p",
        "dh_dp_T", "dh_dT_p",
        "ds_dp_T", "ds_dT_p"
    };

    constexpr double tolerance = 1e-12;

    std::size_t quantity_violations[16]{};
    double quantity_max_absolute[16]{};
    double quantity_max_scaled[16]{};

    for (int i = 0; i < n; ++i) {
        if (cpu[i].status != gpu[i].status) {
            ++status_mismatches;
            continue;
        }

        if (cpu[i].status != PRStateStatus::Success)
            continue;

        const double reference[] = {
            cpu[i].T, cpu[i].p, cpu[i].rho, cpu[i].Z,
            cpu[i].u, cpu[i].h, cpu[i].s,
            cpu[i].cv, cpu[i].cp, cpu[i].sound_speed,
            cpu[i].drho_dp_T, cpu[i].drho_dT_p,
            cpu[i].dh_dp_T, cpu[i].dh_dT_p,
            cpu[i].ds_dp_T, cpu[i].ds_dT_p
        };

        const double actual[] = {
            gpu[i].T, gpu[i].p, gpu[i].rho, gpu[i].Z,
            gpu[i].u, gpu[i].h, gpu[i].s,
            gpu[i].cv, gpu[i].cp, gpu[i].sound_speed,
            gpu[i].drho_dp_T, gpu[i].drho_dT_p,
            gpu[i].dh_dp_T, gpu[i].dh_dT_p,
            gpu[i].ds_dp_T, gpu[i].ds_dT_p
        };

        for (int k = 0; k < 16; ++k) {
            ++comparisons;

            if (!std::isfinite(reference[k]) ||
                !std::isfinite(actual[k])) {
                ++nonfinite;
                continue;
            }

            const double absolute =
                std::abs(actual[k] - reference[k]);

            const double scaled =
                absolute / std::max(
                    1.0, std::abs(reference[k])
                );

            if (absolute > max_absolute)
                max_absolute = absolute;

            if (scaled > max_scaled) {
                max_scaled = scaled;
                worst_case = i;
                worst_quantity = k;
            }

            quantity_max_absolute[k] =
                std::max(quantity_max_absolute[k], absolute);

            quantity_max_scaled[k] =
                std::max(quantity_max_scaled[k], scaled);

            if (scaled > tolerance) {
                ++violations;
                ++quantity_violations[k];
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        if (cpu[i].status != gpu_composed[i].status) {
            ++composed_status_mismatches;
            continue;
        }

        if (cpu[i].status != PRStateStatus::Success)
            continue;

        const double reference[] = {
            cpu[i].T,cpu[i].p,cpu[i].rho,cpu[i].Z,
            cpu[i].u,cpu[i].h,cpu[i].s,
            cpu[i].cv,cpu[i].cp,cpu[i].sound_speed,
            cpu[i].drho_dp_T,cpu[i].drho_dT_p,
            cpu[i].dh_dp_T,cpu[i].dh_dT_p,
            cpu[i].ds_dp_T,cpu[i].ds_dT_p
        };

        const double actual[] = {
            gpu_composed[i].T,gpu_composed[i].p,
            gpu_composed[i].rho,gpu_composed[i].Z,
            gpu_composed[i].u,gpu_composed[i].h,
            gpu_composed[i].s,
            gpu_composed[i].cv,gpu_composed[i].cp,
            gpu_composed[i].sound_speed,
            gpu_composed[i].drho_dp_T,
            gpu_composed[i].drho_dT_p,
            gpu_composed[i].dh_dp_T,
            gpu_composed[i].dh_dT_p,
            gpu_composed[i].ds_dp_T,
            gpu_composed[i].ds_dT_p
        };

        for (int k = 0; k < 16; ++k) {
            ++composed_comparisons;

            if (!std::isfinite(reference[k]) ||
                !std::isfinite(actual[k])) {
                ++composed_nonfinite;
                continue;
            }

            const double absolute =
                std::abs(actual[k]-reference[k]);

            const double scaled =
                absolute/std::max(
                    1.0,std::abs(reference[k])
                );

            composed_max_absolute =
                std::max(composed_max_absolute,absolute);

            composed_max_scaled =
                std::max(composed_max_scaled,scaled);

            composed_quantity_max_scaled[k] =
                std::max(composed_quantity_max_scaled[k],scaled);

            if (scaled > tolerance) {
                ++composed_violations;
                ++composed_quantity_violations[k];

                if (composed_violations <= 20) {
                    std::cout
                        << std::setprecision(17)
                        << "G2e.2 violation:"
                        << " T=" << inputs[i].T
                        << " v=" << inputs[i].v
                        << " p=" << inputs[i].p
                        << " quantity=" << names[k]
                        << " CPU=" << reference[k]
                        << " GPU=" << actual[k]
                        << " absolute=" << absolute
                        << " scaled=" << scaled
                        << '\n';
                }
            }
        }
    }

    const bool passed =
        composed_violations == 0 &&
        composed_nonfinite == 0 &&
        composed_status_mismatches == 0 &&
        violations == 0 &&
        nonfinite == 0 &&
        status_mismatches == 0;

    std::cout << std::setprecision(17)
              << "PR state CUDA parity: "
              << (passed ? "PASS" : "FAIL") << '\n'
              << "Cases: " << n << '\n'
              << "Comparisons: " << comparisons << '\n'
              << "Status mismatches: "
              << status_mismatches << '\n'
              << "Tolerance violations: "
              << violations << '\n'
              << "Nonfinite: " << nonfinite << '\n'
              << "Maximum absolute error: "
              << max_absolute << '\n'
              << "Maximum scaled error: "
              << max_scaled << '\n';

    if (worst_case >= 0) {
        std::cout
            << "Worst case: T="
            << inputs[worst_case].T
            << " v=" << inputs[worst_case].v
            << " p=" << inputs[worst_case].p
            << " quantity=" << names[worst_quantity]
            << '\n';
    }

    std::cout << "\nPer-quantity diagnostics:\n";

    for (int k = 0; k < 16; ++k) {
        std::cout
            << names[k]
            << " violations=" << quantity_violations[k]
            << " max_abs=" << quantity_max_absolute[k]
            << " max_scaled=" << quantity_max_scaled[k]
            << '\n';
    }

    std::cout
        << "\nG2e.2 composed CUDA diagnostics:\n"
        << "Composition comparisons: "
        << composed_comparisons << '\n'
        << "Composition status mismatches: "
        << composed_status_mismatches << '\n'
        << "Composition tolerance violations: "
        << composed_violations << '\n'
        << "Composition nonfinite: "
        << composed_nonfinite << '\n'
        << "Composition maximum absolute error: "
        << composed_max_absolute << '\n'
        << "Composition maximum scaled error: "
        << composed_max_scaled << '\n';

    for (int k = 0; k < 16; ++k) {
        std::cout
            << names[k]
            << " violations="
            << composed_quantity_violations[k]
            << " max_scaled="
            << composed_quantity_max_scaled[k]
            << '\n';
    }

    return passed ? 0 : 1;
}
