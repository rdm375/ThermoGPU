#include "thermomodel/detail/cubic_kernel.hpp"
#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using thermomodel::detail::CubicRoots;
using thermomodel::detail::solve_monic_cubic;

struct CubicCase {
    double c2;
    double c1;
    double c0;
    double B;
    int category;
};

enum {
    BASIC = 0,
    METHANE = 1,
    REPEATED = 2,
    NEAR_DEGENERATE = 3,
    DISCRIMINANT = 4
};

__global__ void evaluate_cubics(
    const CubicCase* cases,
    CubicRoots* results,
    int count)
{
    const int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count) {
        const auto& c = cases[i];

        results[i] = solve_monic_cubic(
            c.c2, c.c1, c.c0
        );
    }
}

static void check_cuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA failure [%s]: %s\n",
            operation,
            cudaGetErrorString(error)
        );
        std::exit(EXIT_FAILURE);
    }
}

static void add_case(
    std::vector<CubicCase>& cases,
    double c2,
    double c1,
    double c0,
    double B,
    int category)
{
    cases.push_back({c2,c1,c0,B,category});
}

static void add_roots(
    std::vector<CubicCase>& cases,
    double x,
    double y,
    double z,
    int category)
{
    add_case(
        cases,
        -(x+y+z),
        x*y+x*z+y*z,
        -x*y*z,
        0.0,
        category
    );
}

static std::vector<CubicCase> generate_cases()
{
    using namespace thermomodel::detail;

    std::vector<CubicCase> cases;
    cases.reserve(122000);

    // Original eight smoke tests.
    constexpr std::array<std::array<double,3>,8> basic{{
        {{0.0,0.0,-1.0}},
        {{0.0,-1.0,0.0}},
        {{-6.0,11.0,-6.0}},
        {{-3.0,3.0,-1.0}},
        {{0.0,0.0,0.0}},
        {{1.0,1.0,1.0}},
        {{-1.0,-1.0,1.0}},
        {{-0.9,0.1,-0.001}}
    }};

    for (const auto& c : basic)
        add_case(cases,c[0],c[1],c[2],0.0,BASIC);

    // Methane PR operating-domain sweep.
    const auto q = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    constexpr int nT = 401;
    constexpr int nP = 301;

    constexpr double Tmin = 100.0;
    constexpr double Tmax = 1000.0;
    constexpr double pmin = 100.0;
    constexpr double pmax = 100e6;

    const double logpmin = std::log(pmin);
    const double logpmax = std::log(pmax);

    for (int i = 0; i < nT; ++i) {
        const double T =
            Tmin+(Tmax-Tmin)*i/(nT-1);

        const double a = pr_attraction(q,T).a;

        for (int j = 0; j < nP; ++j) {
            const double fraction =
                static_cast<double>(j)/(nP-1);

            const double p = std::exp(
                logpmin+
                fraction*(logpmax-logpmin)
            );

            const double A =
                a*p/(q.R*q.R*T*T);

            const double B =
                q.b*p/(q.R*T);

            const double c2 = B-1;
            const double c1 = A-3*B*B-2*B;
            const double c0 =
                -A*B+B*B+B*B*B;

            add_case(
                cases,c2,c1,c0,B,METHANE
            );
        }
    }

    // Repeated-root cases.
    constexpr std::array<double,9> values{{
        -100.0,-10.0,-1.0,-0.1,
        0.0,0.1,1.0,10.0,100.0
    }};

    for (double x : values) {
        add_roots(cases,x,x,x,REPEATED);

        for (double y : values)
            add_roots(cases,x,x,y,REPEATED);
    }

    // Near-degenerate cases.
    constexpr std::array<double,7> centers{{
        -10.0,-1.0,-0.1,0.0,
        0.1,1.0,10.0
    }};

    constexpr std::array<double,12> offsets{{
        -1e-5,-1e-7,-1e-9,
        -1e-11,-1e-13,-1e-15,
         1e-15,1e-13,1e-11,
         1e-9,1e-7,1e-5
    }};

    for (double x : centers) {
        for (double delta : offsets) {
            add_roots(
                cases,x,x+delta,x+2*delta,
                NEAR_DEGENERATE
            );

            add_roots(
                cases,x,x,x+delta,
                NEAR_DEGENERATE
            );

            add_roots(
                cases,x,x+delta,x+1.0,
                NEAR_DEGENERATE
            );
        }
    }

    // Discriminant boundary perturbations.
    add_case(
        cases,0.0,-3.0,2.0,0.0,DISCRIMINANT
    );

    double below = 2.0;
    double above = 2.0;

    for (int i = 0; i < 128; ++i) {
        below = std::nextafter(
            below,
            -std::numeric_limits<double>::infinity()
        );

        above = std::nextafter(
            above,
            std::numeric_limits<double>::infinity()
        );

        add_case(
            cases,0.0,-3.0,below,0.0,DISCRIMINANT
        );

        add_case(
            cases,0.0,-3.0,above,0.0,DISCRIMINANT
        );
    }

    return cases;
}

static int physical_count(
    const CubicRoots& roots,
    double B)
{
    int count = 0;

    for (int i = 0; i < roots.count; ++i) {
        if (std::isfinite(roots.values[i]) &&
            roots.values[i] > B*(1.0+1e-12))
            ++count;
    }

    return count;
}

static double selected_root(
    const CubicRoots& roots,
    double B,
    bool vapor)
{
    double selected = 0.0;
    bool found = false;

    for (int i = 0; i < roots.count; ++i) {
        const double z = roots.values[i];

        if (!std::isfinite(z) ||
            !(z > B*(1.0+1e-12)))
            continue;

        if (!found) {
            selected = z;
            found = true;
        } else if (vapor) {
            selected = std::max(selected,z);
        } else {
            selected = std::min(selected,z);
        }
    }

    return found
        ? selected
        : std::numeric_limits<double>::quiet_NaN();
}

int main()
{
    const auto cases = generate_cases();

    const int count =
        static_cast<int>(cases.size());

    std::vector<CubicRoots> gpu_results(count);

    CubicCase* d_cases = nullptr;
    CubicRoots* d_results = nullptr;

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_cases),
            count*sizeof(CubicCase)
        ),
        "cudaMalloc cases"
    );

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&d_results),
            count*sizeof(CubicRoots)
        ),
        "cudaMalloc results"
    );

    check_cuda(
        cudaMemcpy(
            d_cases,
            cases.data(),
            count*sizeof(CubicCase),
            cudaMemcpyHostToDevice
        ),
        "copy cases"
    );

    constexpr int block_size = 256;

    const int blocks =
        (count+block_size-1)/block_size;

    evaluate_cubics<<<blocks,block_size>>>(
        d_cases,d_results,count
    );

    check_cuda(
        cudaGetLastError(),
        "kernel launch"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "kernel execution"
    );

    check_cuda(
        cudaMemcpy(
            gpu_results.data(),
            d_results,
            count*sizeof(CubicRoots),
            cudaMemcpyDeviceToHost
        ),
        "copy results"
    );

    cudaFree(d_cases);
    cudaFree(d_results);

    long long root_comparisons = 0;
    long long count_mismatches = 0;
    long long value_mismatches = 0;
    long long physical_count_mismatches = 0;
    long long selection_mismatches = 0;
    long long nonfinite = 0;

    double max_absolute_error = 0.0;
    double max_scaled_error = 0.0;

    int worst_case = -1;
    int worst_root = -1;

    for (int i = 0; i < count; ++i) {
        const auto& c = cases[i];

        const auto cpu = solve_monic_cubic(
            c.c2,c.c1,c.c0
        );

        const auto& gpu = gpu_results[i];

        if (cpu.count != gpu.count) {
            ++count_mismatches;

            if (count_mismatches <= 10) {
                std::fprintf(
                    stderr,
                    "ROOT COUNT MISMATCH "
                    "case=%d category=%d "
                    "CPU=%d GPU=%d "
                    "c2=%.17g c1=%.17g c0=%.17g\n",
                    i,c.category,
                    cpu.count,gpu.count,
                    c.c2,c.c1,c.c0
                );
            }
        } else {
            for (int j = 0; j < cpu.count; ++j) {
                const double x = cpu.values[j];
                const double y = gpu.values[j];

                ++root_comparisons;

                if (!std::isfinite(x) ||
                    !std::isfinite(y)) {
                    ++nonfinite;
                    continue;
                }

                const double absolute_error =
                    std::abs(x-y);

                const double scaled_error =
                    absolute_error /
                    std::max(1.0,std::abs(x));

                if (absolute_error > max_absolute_error)
                    max_absolute_error = absolute_error;

                if (scaled_error > max_scaled_error) {
                    max_scaled_error = scaled_error;
                    worst_case = i;
                    worst_root = j;
                }

                if (scaled_error > 1e-12) {
                    ++value_mismatches;

                    if (value_mismatches <= 10) {
                        std::fprintf(
                            stderr,
                            "ROOT VALUE MISMATCH "
                            "case=%d root=%d "
                            "CPU=%.17g GPU=%.17g "
                            "scaled=%.17g\n",
                            i,j,x,y,scaled_error
                        );
                    }
                }
            }
        }

        // Physical-root diagnostics apply to methane PR cases.
        if (c.category == METHANE) {
            const int nc = physical_count(cpu,c.B);
            const int ng = physical_count(gpu,c.B);

            if (nc != ng)
                ++physical_count_mismatches;

            for (bool vapor : {false,true}) {
                const double x =
                    selected_root(cpu,c.B,vapor);

                const double y =
                    selected_root(gpu,c.B,vapor);

                if (std::isnan(x) && std::isnan(y))
                    continue;

                if (!std::isfinite(x) ||
                    !std::isfinite(y)) {
                    ++selection_mismatches;
                    continue;
                }

                const double scaled_error =
                    std::abs(x-y) /
                    std::max(1.0,std::abs(x));

                if (scaled_error > 1e-12)
                    ++selection_mismatches;
            }
        }
    }

    const bool pass =
        count_mismatches == 0 &&
        value_mismatches == 0 &&
        physical_count_mismatches == 0 &&
        selection_mismatches == 0 &&
        nonfinite == 0;

    std::printf(
        "Cubic CUDA parity: %s\n"
        "Cases: %d\n"
        "Root comparisons: %lld\n"
        "Root-count mismatches: %lld\n"
        "Root-value mismatches: %lld\n"
        "Physical-root-count mismatches: %lld\n"
        "Physical-root-selection mismatches: %lld\n"
        "Nonfinite comparisons: %lld\n"
        "Maximum absolute root error: %.17g\n"
        "Maximum scaled root error: %.17g\n"
        "Worst case: %d\n"
        "Worst root: %d\n",
        pass ? "PASS" : "FAIL",
        count,
        root_comparisons,
        count_mismatches,
        value_mismatches,
        physical_count_mismatches,
        selection_mismatches,
        nonfinite,
        max_absolute_error,
        max_scaled_error,
        worst_case,
        worst_root
    );

    if (worst_case >= 0) {
        const auto& c = cases[worst_case];

        std::printf(
            "Worst coefficients: "
            "c2=%.17g c1=%.17g c0=%.17g B=%.17g "
            "category=%d\n",
            c.c2,c.c1,c.c0,c.B,c.category
        );
    }

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
