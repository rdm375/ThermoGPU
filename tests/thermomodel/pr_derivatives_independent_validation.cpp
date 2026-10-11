#include "thermomodel/detail/pr_coefficients_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

using thermomodel::detail::PRCoefficientParameters;
using thermomodel::detail::pr_pressure_derivatives;

// Independent pressure implementation.
//
// Deliberately does not call pr_pressure_derivatives().
// Attraction is calculated directly from the PR parameters.
double independent_pressure(
    const PRCoefficientParameters& c,
    double T,
    double v
) {
    const double sqrt_Tr = std::sqrt(T / c.Tc);

    const double alpha_factor =
        1.0 + c.kappa * (1.0 - sqrt_Tr);

    const double a =
        c.a0 * alpha_factor * alpha_factor;

    const double D =
        v*v + 2.0*c.b*v - c.b*c.b;

    return c.R*T/(v-c.b) - a/D;
}

template <typename Function>
double five_point_derivative(
    Function f,
    double x,
    double h
) {
    return (
        f(x-2.0*h)
        - 8.0*f(x-h)
        + 8.0*f(x+h)
        - f(x+2.0*h)
    ) / (12.0*h);
}

struct DerivativeStatistics {
    std::size_t comparisons = 0;
    std::size_t failures = 0;
    std::size_t nonfinite = 0;

    double maximum_error = 0.0;
    double maximum_relative_error = 0.0;

    double worst_T = 0.0;
    double worst_v = 0.0;
};

void validate_derivative(
    DerivativeStatistics& stats,
    double analytical,
    double numerical,
    double T,
    double v,
    double tolerance
) {
    ++stats.comparisons;

    if (!std::isfinite(analytical) ||
        !std::isfinite(numerical)) {
        ++stats.nonfinite;
        ++stats.failures;
        return;
    }

    const double absolute =
        std::abs(analytical-numerical);

    const double relative =
        absolute / std::max(1.0, std::abs(analytical));

    stats.maximum_error =
        std::max(stats.maximum_error, absolute);

    if (relative > stats.maximum_relative_error) {
        stats.maximum_relative_error = relative;
        stats.worst_T = T;
        stats.worst_v = v;
    }

    if (relative > tolerance)
        ++stats.failures;
}

void print_statistics(
    const char* name,
    const DerivativeStatistics& stats
) {
    std::cout
        << name << '\n'
        << "  Comparisons: " << stats.comparisons << '\n'
        << "  Failures: " << stats.failures << '\n'
        << "  Nonfinite: " << stats.nonfinite << '\n'
        << "  Maximum absolute error: "
        << stats.maximum_error << '\n'
        << "  Maximum scaled error: "
        << stats.maximum_relative_error << '\n'
        << "  Worst T: " << stats.worst_T << '\n'
        << "  Worst v: " << stats.worst_v << '\n';
}

} // namespace

int main() {
    using namespace thermomodel::detail;

    const auto coefficients = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    constexpr std::array<double, 9> temperatures = {
        110.0,
        140.0,
        170.0,
        190.56,
        220.0,
        300.0,
        400.0,
        600.0,
        900.0
    };

    constexpr std::array<double, 10> volume_multipliers = {
        1.05,
        1.2,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        20.0,
        100.0,
        500.0
    };

    // Dimensionless relative step sizes.
    constexpr std::array<double, 6> step_sizes = {
        1e-2,
        3e-3,
        1e-3,
        3e-4,
        1e-4,
        3e-5
    };

    constexpr double tolerance = 1e-7;

    DerivativeStatistics volume_stats;
    DerivativeStatistics temperature_stats;

    std::size_t states = 0;
    std::size_t unstable_volume = 0;
    std::size_t unstable_temperature = 0;

    for (double T : temperatures) {
        for (double multiplier : volume_multipliers) {
            const double v = multiplier * coefficients.b;

            ++states;

            const auto analytical =
                pr_pressure_derivatives(coefficients, T, v);

            const auto pressure_at_volume =
                [&](double trial_v) {
                    return independent_pressure(
                        coefficients,
                        T,
                        trial_v
                    );
                };

            const auto pressure_at_temperature =
                [&](double trial_T) {
                    return independent_pressure(
                        coefficients,
                        trial_T,
                        v
                    );
                };

            double best_volume_error =
                std::numeric_limits<double>::infinity();

            double best_temperature_error =
                std::numeric_limits<double>::infinity();

            double best_volume_value =
                std::numeric_limits<double>::quiet_NaN();

            double best_temperature_value =
                std::numeric_limits<double>::quiet_NaN();

            std::array<double, step_sizes.size()> volume_errors{};
            std::array<double, step_sizes.size()> temperature_errors{};

            volume_errors.fill(
                std::numeric_limits<double>::infinity()
            );

            temperature_errors.fill(
                std::numeric_limits<double>::infinity()
            );

            for (std::size_t k = 0; k < step_sizes.size(); ++k) {
                const double fraction = step_sizes[k];

                const double hv = fraction * v;
                const double hT = fraction * T;

                if (v-2.0*hv > coefficients.b) {
                    const double numerical =
                        five_point_derivative(
                            pressure_at_volume,
                            v,
                            hv
                        );

                    if (std::isfinite(numerical)) {
                        const double error =
                            std::abs(numerical-analytical.pv) /
                            std::max(1.0, std::abs(analytical.pv));

                        volume_errors[k] = error;

                        if (error < best_volume_error) {
                            best_volume_error = error;
                            best_volume_value = numerical;
                        }
                    }
                }

                if (T-2.0*hT > 0.0) {
                    const double numerical =
                        five_point_derivative(
                            pressure_at_temperature,
                            T,
                            hT
                        );

                    if (std::isfinite(numerical)) {
                        const double error =
                            std::abs(numerical-analytical.pt) /
                            std::max(1.0, std::abs(analytical.pt));

                        temperature_errors[k] = error;

                        if (error < best_temperature_error) {
                            best_temperature_error = error;
                            best_temperature_value = numerical;
                        }
                    }
                }
            }

            // Require a useful numerical differentiation result.
            validate_derivative(
                volume_stats,
                analytical.pv,
                best_volume_value,
                T,
                v,
                tolerance
            );

            validate_derivative(
                temperature_stats,
                analytical.pt,
                best_temperature_value,
                T,
                v,
                tolerance
            );

            // Diagnostic: did at least two adjacent step sizes
            // produce independently acceptable estimates?
            bool volume_stable = false;
            bool temperature_stable = false;

            for (std::size_t k = 1; k < step_sizes.size(); ++k) {
                if (volume_errors[k-1] < tolerance &&
                    volume_errors[k] < tolerance) {
                    volume_stable = true;
                }

                if (temperature_errors[k-1] < tolerance &&
                    temperature_errors[k] < tolerance) {
                    temperature_stable = true;
                }
            }

            if (!volume_stable)
                ++unstable_volume;

            if (!temperature_stable)
                ++unstable_temperature;
        }
    }

    std::cout << std::setprecision(17);

    print_statistics("Volume derivative", volume_stats);
    print_statistics("Temperature derivative", temperature_stats);

    std::cout
        << "States: " << states << '\n'
        << "Unstable volume states: "
        << unstable_volume << '\n'
        << "Unstable temperature states: "
        << unstable_temperature << '\n';

    const bool passed =
        volume_stats.failures == 0 &&
        temperature_stats.failures == 0 &&
        unstable_volume == 0 &&
        unstable_temperature == 0;

    std::cout
        << "PR independent derivative validation: "
        << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
