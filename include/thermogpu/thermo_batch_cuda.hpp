#pragma once

#include "thermogpu/thermo_batch.hpp"
#include "thermomodel/peng_robinson.hpp"

#include <array>

namespace thermogpu::thermo {

// Pure-fluid Peng-Robinson with polynomial ideal-gas heat capacity.
//
// cp(T) = sum_{k=0}^4 a[k] * (T / caloric_scale)^k
//
// All thermodynamic properties use mass-specific SI units.
struct CUDAPRParameters {
    thermomodel::PRParameters pr{};

    std::array<double, 5> caloric_coefficients{
        2200.0, 0.0, 0.0, 0.0, 0.0
    };

    double caloric_scale = 1000.0;
};

BatchResult evaluate_batch_cuda(
    const CUDAPRParameters& parameters,
    const BatchInput& input
);

} // namespace thermogpu::thermo
