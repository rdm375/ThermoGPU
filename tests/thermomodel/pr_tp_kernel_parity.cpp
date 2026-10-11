#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace thermomodel::detail;

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
        std::max({1.0, std::fabs(a), std::fabs(b)});
}

int main()
{
    const auto parameters = make_parameters();

    constexpr int nt = 301;
    constexpr int np = 301;

    std::size_t cases = 0;
    std::size_t successful = 0;
    std::size_t comparisons = 0;
    std::size_t violations = 0;

    double max_scaled = 0.0;

    for (int i = 0; i < nt; ++i) {
        const double T =
            100.0 + (1200.0-100.0)*i/(nt-1);

        for (int j = 0; j < np; ++j) {
            const double fraction =
                static_cast<double>(j)/(np-1);

            const double p =
                100.0*std::pow(1.0e6, fraction);

            for (const auto policy : {
                PRTPRootPolicy::Vapor,
                PRTPRootPolicy::Liquid,
                PRTPRootPolicy::RejectMultiple
            }) {
                const auto actual =
                    pr_evaluate_Tp_polynomial(
                        parameters,T,p,policy
                    );

                ++cases;

                if (actual.status != PRTPStatus::Success)
                    continue;

                ++successful;

                const auto reference =
                    pr_evaluate_Tvp_polynomial(
                        parameters,T,actual.selected_v,p
                    );

                if (reference.status != PRStateStatus::Success) {
                    ++violations;
                    continue;
                }

                const double a[] = {
                    actual.state.T,
                    actual.state.p,
                    actual.state.rho,
                    actual.state.Z,
                    actual.state.u,
                    actual.state.h,
                    actual.state.s,
                    actual.state.cv,
                    actual.state.cp,
                    actual.state.sound_speed,
                    actual.state.drho_dp_T,
                    actual.state.drho_dT_p,
                    actual.state.dh_dp_T,
                    actual.state.dh_dT_p,
                    actual.state.ds_dp_T,
                    actual.state.ds_dT_p
                };

                const double b[] = {
                    reference.T,
                    reference.p,
                    reference.rho,
                    reference.Z,
                    reference.u,
                    reference.h,
                    reference.s,
                    reference.cv,
                    reference.cp,
                    reference.sound_speed,
                    reference.drho_dp_T,
                    reference.drho_dT_p,
                    reference.dh_dp_T,
                    reference.dh_dT_p,
                    reference.ds_dp_T,
                    reference.ds_dT_p
                };

                for (int k = 0; k < 16; ++k) {
                    ++comparisons;

                    const double error =
                        scaled_error(a[k],b[k]);

                    max_scaled =
                        std::max(max_scaled,error);

                    if (error != 0.0)
                        ++violations;
                }
            }
        }
    }

    std::cout << std::setprecision(17)
              << "G2e.3 CPU cases: " << cases << '\n'
              << "G2e.3 successful states: " << successful << '\n'
              << "G2e.3 comparisons: " << comparisons << '\n'
              << "G2e.3 violations: " << violations << '\n'
              << "G2e.3 maximum scaled error: "
              << max_scaled << '\n';

    return violations == 0 ? 0 : 1;
}
