#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace thermomodel::detail;

struct Input {
    double T;
    double p;
    PRTPRootPolicy policy;
};

__global__ void evaluate_kernel(
    PRPolynomialStateParameters parameters,
    const Input* inputs,
    PRTPResult* outputs,
    int n)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;

    if (i >= n)
        return;

    const Input input = inputs[i];

    outputs[i] = pr_evaluate_Tp_polynomial(
        parameters,
        input.T,
        input.p,
        input.policy
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

static PRPolynomialStateParameters make_parameters()
{
    const auto coefficients = make_pr_coefficients(
        0.016043, 190.56, 4599200.0, 0.01142
    );

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

    return {
        {coefficients, 101325.0},
        polynomial,
        298.15
    };
}

static double scaled_error(double a, double b)
{
    return std::fabs(a-b) /
        std::max({1.0,std::fabs(a),std::fabs(b)});
}

int main()
{
    const auto parameters = make_parameters();

    constexpr int nt = 301;
    constexpr int np = 301;
    constexpr int policies = 3;
    constexpr int n = nt*np*policies;

    std::vector<Input> inputs(n);
    std::vector<PRTPResult> cpu(n);
    std::vector<PRTPResult> gpu(n);

    const PRTPRootPolicy policy_values[] = {
        PRTPRootPolicy::Vapor,
        PRTPRootPolicy::Liquid,
        PRTPRootPolicy::RejectMultiple
    };

    for (int i = 0; i < nt; ++i) {
        const double T =
            100.0+(1200.0-100.0)*i/(nt-1);

        for (int j = 0; j < np; ++j) {
            const double fraction =
                static_cast<double>(j)/(np-1);

            const double p =
                100.0*std::pow(1.0e6,fraction);

            for (int k = 0; k < policies; ++k) {
                const int index =
                    (i*np+j)*policies+k;

                inputs[index] = {
                    T,p,policy_values[k]
                };

                cpu[index] =
                    pr_evaluate_Tp_polynomial(
                        parameters,T,p,policy_values[k]
                    );
            }
        }
    }

    Input* device_inputs = nullptr;
    PRTPResult* device_outputs = nullptr;

    if (!check(cudaMalloc(
        reinterpret_cast<void**>(&device_inputs),
        n*sizeof(Input)), "cudaMalloc inputs"))
        return 1;

    if (!check(cudaMalloc(
        reinterpret_cast<void**>(&device_outputs),
        n*sizeof(PRTPResult)), "cudaMalloc outputs"))
        return 1;

    if (!check(cudaMemcpy(
        device_inputs,inputs.data(),
        n*sizeof(Input),cudaMemcpyHostToDevice),
        "cudaMemcpy inputs"))
        return 1;

    evaluate_kernel<<<(n+255)/256,256>>>(
        parameters,device_inputs,device_outputs,n
    );

    if (!check(cudaGetLastError(),"kernel launch"))
        return 1;

    if (!check(cudaDeviceSynchronize(),"kernel execution"))
        return 1;

    if (!check(cudaMemcpy(
        gpu.data(),device_outputs,
        n*sizeof(PRTPResult),cudaMemcpyDeviceToHost),
        "cudaMemcpy outputs"))
        return 1;

    cudaFree(device_inputs);
    cudaFree(device_outputs);

    std::size_t status_mismatches = 0;
    std::size_t root_count_mismatches = 0;
    std::size_t comparisons = 0;
    std::size_t violations = 0;

    double max_scaled = 0.0;
    double max_root_Z = 0.0;
    double max_root_v = 0.0;
    double max_relative_v = 0.0;
    double max_properties = 0.0;

    std::size_t root_Z_violations = 0;
    std::size_t root_v_violations = 0;
    std::size_t property_violations = 0;

    constexpr double tolerance = 1e-12;

    for (int i = 0; i < n; ++i) {
        if (cpu[i].status != gpu[i].status) {
            ++status_mismatches;
            continue;
        }

        if (cpu[i].admissible_roots !=
            gpu[i].admissible_roots)
            ++root_count_mismatches;

        if (cpu[i].status != PRTPStatus::Success)
            continue;

        const auto& a = cpu[i].state;
        const auto& b = gpu[i].state;

        const double reference[] = {
            cpu[i].selected_Z,
            cpu[i].selected_v,
            a.T,a.p,a.rho,a.Z,
            a.u,a.h,a.s,
            a.cv,a.cp,a.sound_speed,
            a.drho_dp_T,a.drho_dT_p,
            a.dh_dp_T,a.dh_dT_p,
            a.ds_dp_T,a.ds_dT_p
        };

        const double actual[] = {
            gpu[i].selected_Z,
            gpu[i].selected_v,
            b.T,b.p,b.rho,b.Z,
            b.u,b.h,b.s,
            b.cv,b.cp,b.sound_speed,
            b.drho_dp_T,b.drho_dT_p,
            b.dh_dp_T,b.dh_dT_p,
            b.ds_dp_T,b.ds_dT_p
        };

        for (int k = 0; k < 18; ++k) {
            ++comparisons;

            const double error =
                scaled_error(reference[k],actual[k]);

            max_scaled =
                std::max(max_scaled,error);

            if (k == 0) {
                max_root_Z = std::max(max_root_Z,error);
                if (!(error <= tolerance))
                    ++root_Z_violations;
            } else if (k == 1) {
                max_root_v = std::max(max_root_v,error);

                const double relative_v =
                    std::fabs(reference[k]-actual[k]) /
                    std::max(
                        std::fabs(reference[k]),
                        std::fabs(actual[k])
                    );

                max_relative_v =
                    std::max(max_relative_v,relative_v);

                if (!(error <= tolerance))
                    ++root_v_violations;
            } else {
                max_properties =
                    std::max(max_properties,error);
                if (!(error <= tolerance))
                    ++property_violations;
            }

            if (!(error <= tolerance)) {
                ++violations;

                if (violations <= 12) {
                    std::cout << std::setprecision(17)
                              << "G2e.3 violation:"
                              << " T=" << inputs[i].T
                              << " p=" << inputs[i].p
                              << " policy="
                              << static_cast<int>(inputs[i].policy)
                              << " field=" << k
                              << " CPU=" << reference[k]
                              << " GPU=" << actual[k]
                              << " scaled=" << error
                              << '\n';
                }
            }
        }
    }

    std::cout << std::setprecision(17)
              << "G2e.3 CUDA cases: " << n << '\n'
              << "G2e.3 status mismatches: "
              << status_mismatches << '\n'
              << "G2e.3 root-count mismatches: "
              << root_count_mismatches << '\n'
              << "G2e.3 comparisons: "
              << comparisons << '\n'
              << "G2e.3 tolerance violations: "
              << violations << '\n'
              << "G2e.3 maximum scaled error: "
              << max_scaled << '\n'
              << "G2e.3 Z violations: "
              << root_Z_violations << '\n'
              << "G2e.3 v violations: "
              << root_v_violations << '\n'
              << "G2e.3 property violations: "
              << property_violations << '\n'
              << "G2e.3 maximum Z scaled error: "
              << max_root_Z << '\n'
              << "G2e.3 maximum v scaled error: "
              << max_root_v << '\n'
              << "G2e.3 maximum v relative error: "
              << max_relative_v << '\n'
              << "G2e.3 maximum property scaled error: "
              << max_properties << '\n';

    return (
        status_mismatches == 0 &&
        root_count_mismatches == 0 &&
        violations == 0
    ) ? 0 : 1;
}
