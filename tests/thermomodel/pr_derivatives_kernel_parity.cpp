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

    const double Tmin = 100.0;
    const double Tmax = 1000.0;

    const double b = coefficients.b;

    // Sample admissible volumes logarithmically.
    const double vmin = b * (1.0 + 1e-8);
    const double vmax = 1000.0 * b;

    std::size_t cases = 0;
    std::size_t comparisons = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;

    double max_abs_error = 0.0;
    double max_scaled_error = 0.0;

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

            const double R = coefficients.R;
            const double a = attraction.a;
            const double ap = attraction.ap;

            // Independent transcription of production expressions.
            const double D =
                v*v + 2*b*v - b*b;

            const double p_ref =
                R*T/(v-b) - a/D;

            const double pv_ref =
                -R*T/((v-b)*(v-b))
                + a*(2*v+2*b)/(D*D);

            const double pt_ref =
                R/(v-b) - ap/D;

            const auto result =
                pr_pressure_derivatives(coefficients, T, v);

            const double reference[3] = {
                p_ref, pv_ref, pt_ref
            };

            const double actual[3] = {
                result.p, result.pv, result.pt
            };

            ++cases;

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

                max_abs_error =
                    std::max(max_abs_error, absolute);

                max_scaled_error =
                    std::max(max_scaled_error, scaled);

                if (actual[k] != reference[k])
                    ++mismatches;
            }
        }
    }

    std::cout << std::setprecision(17)
              << "PR derivative CPU parity: "
              << ((mismatches == 0 && nonfinite == 0)
                  ? "PASS" : "FAIL") << '\n'
              << "Cases: " << cases << '\n'
              << "Comparisons: " << comparisons << '\n'
              << "Value mismatches: " << mismatches << '\n'
              << "Nonfinite comparisons: " << nonfinite << '\n'
              << "Maximum absolute error: "
              << max_abs_error << '\n'
              << "Maximum scaled error: "
              << max_scaled_error << '\n';

    return (mismatches == 0 && nonfinite == 0) ? 0 : 1;
}
