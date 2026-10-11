#include "thermomodel/detail/pr_residual_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace thermomodel::detail;

int main() {
    const auto coefficients = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    constexpr int nt = 401;
    constexpr int nv = 301;

    constexpr double sqrt2 = 1.4142135623730950488;

    const double Tmin = 100.0;
    const double Tmax = 1000.0;

    const double R = coefficients.R;
    const double b = coefficients.b;

    const double vmin = b * (1.0 + 1e-8);
    const double vmax = 1000.0 * b;

    std::size_t cases = 0;
    std::size_t comparisons = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
    std::size_t skipped_entropy = 0;

    double max_absolute = 0.0;
    double max_scaled = 0.0;

    for (int i = 0; i < nt; ++i) {
        const double T =
            Tmin + (Tmax-Tmin)*i/(nt-1);

        for (int j = 0; j < nv; ++j) {
            const double fraction =
                static_cast<double>(j)/(nv-1);

            const double v =
                vmin * std::pow(vmax/vmin, fraction);

            const auto attraction =
                pr_attraction(coefficients, T);

            const double a = attraction.a;
            const double ap = attraction.ap;
            const double app = attraction.app;

            const double p =
                pr_pressure_derivatives(
                    coefficients, T, v
                ).p;

            // Independent transcription of production expressions.
            const double logq =
                std::log(
                    (v+(1+sqrt2)*b) /
                    (v+(1-sqrt2)*b)
                );

            const double ur_ref =
                (T*ap-a)*logq/(2*sqrt2*b);

            const double cvres_ref =
                T*app*logq/(2*sqrt2*b);

            const bool valid_entropy =
                std::isfinite(p) && p > 0.0;

            double sres_ref = 0.0;

            if (valid_entropy) {
                const double Z = p*v/(R*T);

                sres_ref =
                    R*std::log(Z*(v-b)/v)
                    + ap*logq/(2*sqrt2*b);
            } else {
                ++skipped_entropy;
            }

            const auto result =
                pr_residual_properties(
                    coefficients, T, v, p
                );

            const double reference[3] = {
                ur_ref, sres_ref, cvres_ref
            };

            const double actual[3] = {
                result.ur, result.sres, result.cvres
            };

            ++cases;

            for (int k = 0; k < 3; ++k) {
                if (k == 1 && !valid_entropy)
                    continue;

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

                max_scaled =
                    std::max(max_scaled, scaled);

                if (actual[k] != reference[k])
                    ++mismatches;
            }
        }
    }

    const bool passed =
        mismatches == 0 && nonfinite == 0;

    std::cout << std::setprecision(17)
              << "PR residual CPU parity: "
              << (passed ? "PASS" : "FAIL") << '\n'
              << "Cases: " << cases << '\n'
              << "Comparisons: " << comparisons << '\n'
              << "Skipped entropy comparisons: "
              << skipped_entropy << '\n'
              << "Value mismatches: " << mismatches << '\n'
              << "Nonfinite comparisons: " << nonfinite << '\n'
              << "Maximum absolute error: "
              << max_absolute << '\n'
              << "Maximum scaled error: "
              << max_scaled << '\n';

    return passed ? 0 : 1;
}
