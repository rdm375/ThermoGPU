#include "thermomodel/caloric.hpp"
#include "thermomodel/detail/polynomial_cp_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace {

bool compare(const char* name, double expected, double actual)
{
    const double error = std::abs(expected - actual);

    const double tolerance =
        8.0 * 2.2204460492503131e-16 *
        std::max({1.0, std::abs(expected), std::abs(actual)});

    if (error > tolerance || !std::isfinite(actual)) {
        std::fprintf(stderr,
                     "%s: expected %.17g, got %.17g, error %.17g\n",
                     name, expected, actual, error);
        return false;
    }

    return true;
}

} // namespace

int main()
{
    constexpr std::array<double, 5> nasa = {
         5.14987613,
        -1.36709788e-2,
         4.91800599e-5,
        -4.84743026e-8,
         1.66693956e-11
    };

    constexpr double Ru = 8.31446261815324;
    constexpr double MW = 0.016043;
    constexpr double scale = 1000.0;

    std::array<double, 5> coefficients{};

    double power = 1.0;

    for (int k = 0; k < 5; ++k) {
        coefficients[k] = nasa[k] * (Ru / MW) * power;
        power *= scale;
    }

    thermomodel::PolynomialCp reference(coefficients, scale);

    thermomodel::detail::PolynomialCpKernel kernel{};

    for (int k = 0; k < 5; ++k)
        kernel.a[k] = coefficients[k];

    kernel.scale = scale;

    constexpr double Tref = 298.15;

    bool passed = true;

    for (int i = 0; i <= 800; ++i) {
        const double T = 200.0 + i;

        passed &= compare(
            "cp",
            reference.cp(T),
            thermomodel::detail::polynomial_cp(kernel, T)
        );

        passed &= compare(
            "enthalpy",
            reference.enthalpy_delta(T, Tref),
            thermomodel::detail::polynomial_enthalpy_delta(
                kernel, T, Tref)
        );

        passed &= compare(
            "entropy",
            reference.entropy_thermal_delta(T, Tref),
            thermomodel::detail::polynomial_entropy_delta(
                kernel, T, Tref)
        );
    }

    if (!passed)
        return 1;

    std::puts("Polynomial caloric kernel parity: PASS (801 states)");

    return 0;
}
