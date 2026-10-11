#include "thermomodel/detail/pr_state_kernel.hpp"
#include "thermomodel/detail/pr_polynomial_state_kernel.hpp"
#include "thermomodel/detail/polynomial_cp_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace thermomodel::detail;

int main() {
    const auto coefficients = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
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

    const double b = coefficients.b;
    const double R = coefficients.R;

    const double vmin = b*(1.0+1e-8);
    const double vmax = 1000.0*b;

    std::size_t cases = 0;
    std::size_t successful = 0;
    std::size_t unstable = 0;
    std::size_t nonphysical = 0;
    std::size_t comparisons = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;

    std::size_t composition_comparisons = 0;
    std::size_t composition_mismatches = 0;
    std::size_t composition_status_mismatches = 0;

    double max_absolute = 0.0;
    double max_scaled = 0.0;

    for (int i = 0; i < nt; ++i) {
        const double T =
            200.0+(1000.0-200.0)*i/(nt-1);

        const PRIdealCaloricState caloric{
            polynomial_cp(polynomial,T),
            polynomial_enthalpy_delta(
                polynomial,T,Tref
            ),
            polynomial_entropy_delta(
                polynomial,T,Tref
            )
        };

        for (int j = 0; j < nv; ++j) {
            const double fraction =
                static_cast<double>(j)/(nv-1);

            const double v =
                vmin*std::pow(vmax/vmin,fraction);

            const auto derivatives =
                pr_pressure_derivatives(
                    coefficients,T,v
                );

            const double p = derivatives.p;

            // Production rejects pressure-unstable roots.
            const double pv = derivatives.pv;
            const double pt = derivatives.pt;

            const auto actual =
                pr_evaluate_Tvp(
                    parameters,caloric,T,v,p
                );

            ++cases;

            const auto composed =
                pr_evaluate_Tvp_polynomial(
                    composed_parameters,T,v,p
                );

            if (composed.status != actual.status) {
                ++composition_status_mismatches;
            } else if (actual.status == PRStateStatus::Success) {
                const double reference_state[] = {
                    actual.T,actual.p,actual.rho,actual.Z,
                    actual.u,actual.h,actual.s,
                    actual.cv,actual.cp,actual.sound_speed,
                    actual.drho_dp_T,actual.drho_dT_p,
                    actual.dh_dp_T,actual.dh_dT_p,
                    actual.ds_dp_T,actual.ds_dT_p
                };

                const double composed_state[] = {
                    composed.T,composed.p,composed.rho,composed.Z,
                    composed.u,composed.h,composed.s,
                    composed.cv,composed.cp,composed.sound_speed,
                    composed.drho_dp_T,composed.drho_dT_p,
                    composed.dh_dp_T,composed.dh_dT_p,
                    composed.ds_dp_T,composed.ds_dT_p
                };

                for (int k = 0; k < 16; ++k) {
                    ++composition_comparisons;

                    if (!std::isfinite(reference_state[k]) ||
                        !std::isfinite(composed_state[k]) ||
                        reference_state[k] != composed_state[k])
                        ++composition_mismatches;
                }
            }

            if (!(std::isfinite(pv) && pv < 0.0 &&
                  std::isfinite(pt))) {
                ++unstable;

                if (actual.status !=
                    PRStateStatus::UnstableRoot)
                    ++mismatches;

                continue;
            }

            const auto residual =
                pr_residual_properties(
                    coefficients,T,v,p
                );

            const double Z = p*v/(R*T);
            const double cv0 = caloric.cp-R;
            const double cv = cv0+residual.cvres;
            const double cp = cv-T*pt*pt/pv;

            const double dv_dT = -pt/pv;
            const double dv_dp = 1/pv;
            const double c2 = -v*v*pv*cp/cv;

            if (!(std::isfinite(cv) && cv > 0.0 &&
                  std::isfinite(cp) && cp > cv &&
                  std::isfinite(c2) && c2 > 0.0)) {
                ++nonphysical;

                if (actual.status !=
                    PRStateStatus::NonphysicalProperties)
                    ++mismatches;

                continue;
            }

            if (actual.status != PRStateStatus::Success) {
                ++mismatches;
                continue;
            }

            ++successful;

            const double h =
                caloric.enthalpy_delta
                +residual.ur+p*v-R*T;

            const double s =
                caloric.entropy_thermal_delta
                -R*std::log(p/pref)+residual.sres;

            const double reference[] = {
                T,p,1/v,Z,
                h-p*v,h,s,
                cv,cp,std::sqrt(c2),
                -dv_dp/(v*v),
                -dv_dT/(v*v),
                v-T*dv_dT,
                cp,
                -dv_dT,
                cp/T
            };

            const double result[] = {
                actual.T,actual.p,actual.rho,actual.Z,
                actual.u,actual.h,actual.s,
                actual.cv,actual.cp,actual.sound_speed,
                actual.drho_dp_T,
                actual.drho_dT_p,
                actual.dh_dp_T,
                actual.dh_dT_p,
                actual.ds_dp_T,
                actual.ds_dT_p
            };

            for (int k = 0; k < 16; ++k) {
                ++comparisons;

                if (!std::isfinite(reference[k]) ||
                    !std::isfinite(result[k])) {
                    ++nonfinite;
                    continue;
                }

                const double absolute =
                    std::abs(result[k]-reference[k]);

                const double scaled =
                    absolute/std::max(
                        1.0,std::abs(reference[k])
                    );

                max_absolute =
                    std::max(max_absolute,absolute);

                max_scaled =
                    std::max(max_scaled,scaled);

                if (result[k] != reference[k])
                    ++mismatches;
            }
        }
    }

    const bool passed =
        mismatches == 0 &&
        nonfinite == 0 &&
        composition_mismatches == 0 &&
        composition_status_mismatches == 0;

    std::cout << std::setprecision(17)
              << "PR state CPU parity: "
              << (passed ? "PASS" : "FAIL") << '\n'
              << "Cases: " << cases << '\n'
              << "Successful states: " << successful << '\n'
              << "Unstable states: " << unstable << '\n'
              << "Nonphysical states: " << nonphysical << '\n'
              << "Comparisons: " << comparisons << '\n'
              << "Mismatches: " << mismatches << '\n'
              << "Nonfinite: " << nonfinite << '\n'
              << "Maximum absolute error: "
              << max_absolute << '\n'
              << "Maximum scaled error: "
              << max_scaled << '\n'
              << "Composition comparisons: "
              << composition_comparisons << '\n'
              << "Composition mismatches: "
              << composition_mismatches << '\n'
              << "Composition status mismatches: "
              << composition_status_mismatches << '\n';

    return passed ? 0 : 1;
}
