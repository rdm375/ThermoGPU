#pragma once

#include "thermomodel/caloric.hpp"
#include <array>
#include <cstddef>
#include <memory>

namespace thermomodel {

// Methane NASA 7-coefficient low-temperature cp/R polynomial,
// applicable over 200–1000 K.
//
// The first five NASA coefficients describe cp/R as a
// polynomial in temperature measured in kelvin.
//
// PolynomialCp uses powers of T/1000 and mass-specific SI units.
class MethaneNasaLowCp final : public CaloricModel {
public:
    explicit MethaneNasaLowCp(std::array<double, 5> coefficients)
        : polynomial_(coefficients, 1000.0) {}

    double minimum_temperature() const noexcept override {
        return 200.0;
    }

    double maximum_temperature() const noexcept override {
        return 1000.0;
    }

    double cp(double T) const noexcept override {
        return polynomial_.cp(T);
    }

    double enthalpy_delta(double T, double ref) const noexcept override {
        return polynomial_.enthalpy_delta(T, ref);
    }

    double entropy_thermal_delta(double T, double ref) const noexcept override {
        return polynomial_.entropy_thermal_delta(T, ref);
    }

    double minimum_cp(double Tmin, double Tmax) const noexcept override {
        return polynomial_.minimum_cp(Tmin, Tmax);
    }

private:
    PolynomialCp polynomial_;
};

inline std::shared_ptr<const CaloricModel>
make_methane_nasa_low_cp()
{
    constexpr double Ru = 8.31446261815324;
    constexpr double MW = 0.016043;
    constexpr double scale = 1000.0;

    constexpr std::array<double, 5> nasa = {
         5.14987613,
        -1.36709788e-2,
         4.91800599e-5,
        -4.84743026e-8,
         1.66693956e-11
    };

    std::array<double, 5> mass_coefficients{};

    double power = 1.0;
    for (std::size_t k = 0; k < nasa.size(); ++k) {
        mass_coefficients[k] = nasa[k] * (Ru / MW) * power;
        power *= scale;
    }

    return std::make_shared<MethaneNasaLowCp>(
        mass_coefficients
    );
}

} // namespace thermomodel
