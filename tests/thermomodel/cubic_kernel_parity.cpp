#include "thermomodel/detail/cubic_kernel.hpp"
#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <vector>

namespace {

// Original production cubic solver, retained as the reference.
std::vector<double> original_cubic(
    double c2, double c1, double c0)
{
    const double q=(3*c1-c2*c2)/9;
    const double r=(9*c2*c1-27*c0-2*c2*c2*c2)/54;
    const double disc=q*q*q+r*r;

    std::vector<double> roots;

    if(disc>0) {
        const double sd=std::sqrt(disc);
        const double u=std::cbrt(r+sd);
        const double v=std::cbrt(r-sd);
        roots.push_back(-c2/3+u+v);
    } else if(q<0) {
        const double arg=std::clamp(
            r/std::sqrt(-q*q*q),-1.0,1.0);
        const double angle=std::acos(arg)/3;

        for(int k=0;k<3;++k)
            roots.push_back(
                2*std::sqrt(-q)*
                std::cos(angle-2*std::numbers::pi*k/3)
                -c2/3);
    } else {
        roots.push_back(-c2/3);
    }

    std::sort(roots.begin(),roots.end());

    roots.erase(
        std::unique(
            roots.begin(),roots.end(),
            [](double a,double b) {
                return std::abs(a-b) <
                    1e-11*std::max({
                        1.0,std::abs(a),std::abs(b)
                    });
            }),
        roots.end()
    );

    return roots;
}

struct Diagnostics {
    long long cases = 0;
    long long comparisons = 0;
    long long count_mismatches = 0;
    long long value_mismatches = 0;
    long long nonfinite = 0;

    double max_absolute_error = 0.0;
    double max_scaled_error = 0.0;

    double worst_c2 = 0.0;
    double worst_c1 = 0.0;
    double worst_c0 = 0.0;

    double worst_expected = 0.0;
    double worst_actual = 0.0;

    const char* worst_category = "none";
};

void compare_case(
    Diagnostics& d,
    double c2,
    double c1,
    double c0,
    const char* category)
{
    using thermomodel::detail::solve_monic_cubic;

    ++d.cases;

    const auto expected = original_cubic(c2,c1,c0);
    const auto actual = solve_monic_cubic(c2,c1,c0);

    if (actual.count != static_cast<int>(expected.size())) {
        ++d.count_mismatches;

        if (d.count_mismatches <= 10) {
            std::fprintf(
                stderr,
                "COUNT MISMATCH [%s] "
                "c2=%.17g c1=%.17g c0=%.17g "
                "reference=%zu extracted=%d\n",
                category,
                c2,c1,c0,
                expected.size(),
                actual.count
            );
        }

        return;
    }

    for (int i = 0; i < actual.count; ++i) {
        const double reference = expected[i];
        const double candidate = actual.values[i];

        ++d.comparisons;

        if (!std::isfinite(reference) ||
            !std::isfinite(candidate)) {
            ++d.nonfinite;
            continue;
        }

        const double absolute_error =
            std::abs(candidate-reference);

        const double scale =
            std::max(1.0,std::abs(reference));

        const double scaled_error =
            absolute_error/scale;

        if (absolute_error > d.max_absolute_error)
            d.max_absolute_error = absolute_error;

        if (scaled_error > d.max_scaled_error) {
            d.max_scaled_error = scaled_error;

            d.worst_c2 = c2;
            d.worst_c1 = c1;
            d.worst_c0 = c0;

            d.worst_expected = reference;
            d.worst_actual = candidate;

            d.worst_category = category;
        }

        if (scaled_error > 1e-12) {
            ++d.value_mismatches;

            if (d.value_mismatches <= 10) {
                std::fprintf(
                    stderr,
                    "VALUE MISMATCH [%s] "
                    "c2=%.17g c1=%.17g c0=%.17g "
                    "reference=%.17g extracted=%.17g "
                    "scaled_error=%.17g\n",
                    category,
                    c2,c1,c0,
                    reference,candidate,scaled_error
                );
            }
        }
    }
}

// Construct a monic cubic from three prescribed roots.
void compare_roots(
    Diagnostics& d,
    double x,
    double y,
    double z,
    const char* category)
{
    const double c2 = -(x+y+z);
    const double c1 = x*y+x*z+y*z;
    const double c0 = -x*y*z;

    compare_case(d,c2,c1,c0,category);
}

void basic_cases(Diagnostics& d)
{
    const std::array<std::array<double,3>,8> cases{{
        {{0.0, 0.0, -1.0}},
        {{0.0, -1.0, 0.0}},
        {{-6.0, 11.0, -6.0}},
        {{-3.0, 3.0, -1.0}},
        {{0.0, 0.0, 0.0}},
        {{1.0, 1.0, 1.0}},
        {{-1.0, -1.0, 1.0}},
        {{-0.9, 0.1, -0.001}}
    }};

    for (const auto& c : cases)
        compare_case(d,c[0],c[1],c[2],"basic");
}

void methane_pr_sweep(Diagnostics& d)
{
    using namespace thermomodel::detail;

    const auto q = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    // Includes subcritical, near-critical and supercritical T.
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

            const double p =
                std::exp(
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

            compare_case(
                d,c2,c1,c0,"methane-pr"
            );
        }
    }
}

void repeated_root_cases(Diagnostics& d)
{
    constexpr std::array<double,9> values{{
        -100.0,-10.0,-1.0,-0.1,
        0.0,0.1,1.0,10.0,100.0
    }};

    for (double x : values) {
        compare_roots(d,x,x,x,"triple-root");

        for (double y : values) {
            compare_roots(d,x,x,y,"double-root");
        }
    }
}

void near_degenerate_cases(Diagnostics& d)
{
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
            compare_roots(
                d,x,x+delta,x+2*delta,
                "near-triple"
            );

            compare_roots(
                d,x,x,x+delta,
                "near-double"
            );

            compare_roots(
                d,x,x+delta,x+1.0,
                "near-discriminant"
            );
        }
    }
}

void discriminant_boundary_cases(Diagnostics& d)
{
    // Directly perturb coefficients of x^3 - 3x + 2,
    // which has roots -2, 1, 1.
    const double c2 = 0.0;
    const double c1 = -3.0;
    const double c0 = 2.0;

    compare_case(
        d,c2,c1,c0,"discriminant-exact"
    );

    double below = c0;
    double above = c0;

    for (int i = 0; i < 128; ++i) {
        below = std::nextafter(
            below,
            -std::numeric_limits<double>::infinity()
        );

        above = std::nextafter(
            above,
            std::numeric_limits<double>::infinity()
        );

        compare_case(
            d,c2,c1,below,"discriminant-below"
        );

        compare_case(
            d,c2,c1,above,"discriminant-above"
        );
    }
}

}

int main()
{
    Diagnostics d;

    basic_cases(d);
    methane_pr_sweep(d);
    repeated_root_cases(d);
    near_degenerate_cases(d);
    discriminant_boundary_cases(d);

    const bool pass =
        d.count_mismatches == 0 &&
        d.value_mismatches == 0 &&
        d.nonfinite == 0;

    std::printf(
        "Cubic CPU reference parity: %s\n"
        "Cases: %lld\n"
        "Root comparisons: %lld\n"
        "Root-count mismatches: %lld\n"
        "Value mismatches: %lld\n"
        "Nonfinite comparisons: %lld\n"
        "Maximum absolute root error: %.17g\n"
        "Maximum scaled root error: %.17g\n"
        "Worst category: %s\n"
        "Worst coefficients: "
        "c2=%.17g c1=%.17g c0=%.17g\n"
        "Worst reference root: %.17g\n"
        "Worst extracted root: %.17g\n",
        pass ? "PASS" : "FAIL",
        d.cases,
        d.comparisons,
        d.count_mismatches,
        d.value_mismatches,
        d.nonfinite,
        d.max_absolute_error,
        d.max_scaled_error,
        d.worst_category,
        d.worst_c2,
        d.worst_c1,
        d.worst_c0,
        d.worst_expected,
        d.worst_actual
    );

    return pass ? 0 : 1;
}
