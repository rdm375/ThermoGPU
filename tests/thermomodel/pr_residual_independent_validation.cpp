#include "thermomodel/detail/pr_residual_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace thermomodel::detail;

namespace {

template <typename Function>
double derivative_5point(
    Function function,
    double x,
    double h
) {
    return (
        function(x-2*h)
        - 8.0*function(x-h)
        + 8.0*function(x+h)
        - function(x+2*h)
    ) / (12.0*h);
}

double scaled_error(double actual, double expected) {
    return std::abs(actual-expected) /
        std::max(1.0, std::abs(expected));
}

} // namespace

int main() {
    const auto coefficients = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    const double b = coefficients.b;

    const std::array<double, 10> temperatures = {
        110.0, 130.0, 160.0, 190.0, 220.0,
        280.0, 350.0, 450.0, 650.0, 900.0
    };

    const std::array<double, 10> volume_factors = {
        1.05, 1.2, 1.5, 2.0, 3.0,
        5.0, 10.0, 20.0, 50.0, 100.0
    };

    const std::array<double, 7> relative_steps = {
        1e-2, 3e-3, 1e-3, 3e-4,
        1e-4, 3e-5, 1e-5
    };

    std::size_t states = 0;
    std::size_t failures = 0;
    std::size_t nonfinite = 0;

    double maximum_volume_error = 0.0;
    double maximum_temperature_error = 0.0;

    constexpr double tolerance = 1e-8;

    for (double T : temperatures) {
        for (double factor : volume_factors) {
            const double v = factor*b;

            const auto pressure =
                pr_pressure_derivatives(
                    coefficients, T, v
                );

            // Pressure is only an argument to the residual
            // kernel's entropy calculation. The internal-energy
            // and cv residuals do not depend on it.
            const double p = pressure.p;

            const auto residual =
                pr_residual_properties(
                    coefficients, T, v, p
                );

            const double volume_reference =
                T*pressure.pt-pressure.p;

            const double temperature_reference =
                residual.cvres;

            const auto ur_at_volume = [&](double volume) {
                return pr_residual_properties(
                    coefficients,
                    T,
                    volume,
                    p
                ).ur;
            };

            const auto ur_at_temperature = [&](double temperature) {
                return pr_residual_properties(
                    coefficients,
                    temperature,
                    v,
                    p
                ).ur;
            };

            double best_volume_error =
                std::numeric_limits<double>::infinity();

            double best_temperature_error =
                std::numeric_limits<double>::infinity();

            for (double relative_step : relative_steps) {
                const double hv = relative_step*v;
                const double hT = relative_step*T;

                // Five-point stencil must remain in v>b.
                if (v-2.0*hv > b) {
                    const double derivative =
                        derivative_5point(
                            ur_at_volume, v, hv
                        );

                    if (std::isfinite(derivative)) {
                        best_volume_error =
                            std::min(
                                best_volume_error,
                                scaled_error(
                                    derivative,
                                    volume_reference
                                )
                            );
                    }
                }

                if (T-2.0*hT > 0.0) {
                    const double derivative =
                        derivative_5point(
                            ur_at_temperature, T, hT
                        );

                    if (std::isfinite(derivative)) {
                        best_temperature_error =
                            std::min(
                                best_temperature_error,
                                scaled_error(
                                    derivative,
                                    temperature_reference
                                )
                            );
                    }
                }
            }

            ++states;

            if (!std::isfinite(best_volume_error) ||
                !std::isfinite(best_temperature_error)) {
                ++nonfinite;
                continue;
            }

            maximum_volume_error =
                std::max(
                    maximum_volume_error,
                    best_volume_error
                );

            maximum_temperature_error =
                std::max(
                    maximum_temperature_error,
                    best_temperature_error
                );

            if (best_volume_error > tolerance ||
                best_temperature_error > tolerance) {
                ++failures;

                std::cerr << std::setprecision(17)
                          << "Failure: T=" << T
                          << " v=" << v
                          << " volume_error="
                          << best_volume_error
                          << " temperature_error="
                          << best_temperature_error
                          << '\n';
            }
        }
    }

    const bool passed =
        failures == 0 && nonfinite == 0;

    std::cout << std::setprecision(17)
              << "PR residual independent validation: "
              << (passed ? "PASS" : "FAIL") << '\n'
              << "States: " << states << '\n'
              << "Failures: " << failures << '\n'
              << "Nonfinite: " << nonfinite << '\n'
              << "Maximum volume derivative scaled error: "
              << maximum_volume_error << '\n'
              << "Maximum temperature derivative scaled error: "
              << maximum_temperature_error << '\n';

    return passed ? 0 : 1;
}
