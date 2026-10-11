// SPDX-License-Identifier: GPL-3.0-or-later
// Frozen methane reference from ThermoGPU.
//
// Source:
// https://github.com/rdm375/ThermoGPU/blob/main/
// tests/reference/data/methane_moderate_pressure.csv
//
// ThermoPack 2.2.3 controlled validation:
// tests/reference/test_thermopack_controlled.cpp

#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/caloric.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

bool close(double got, double expected,
           double rtol = 2e-13,
           double atol = 2e-15)
{
    return std::isfinite(got) &&
           std::abs(got - expected) <=
               atol + rtol * std::abs(expected);
}

int check(const char* name, double got, double expected)
{
    const double error = std::abs(got - expected);

    std::cout << std::setprecision(17)
              << name
              << "\n  ThermoModel: " << got
              << "\n  ThermoGPU:   " << expected
              << "\n  abs error:   " << error
              << '\n';

    if (!close(got, expected)) {
        std::cerr << "FAIL: " << name << '\n';
        return 1;
    }

    return 0;
}

} // namespace

int main()
{
    // ThermoGPU controlled methane pseudo-component.
    thermomodel::PRParameters params;
    params.Tc = 190.56;
    params.pc = 4599200.0;
    params.acentric = 0.01142;
    params.molar_mass = 0.016043;

    // ThermoGPU's Z and density do not depend on the
    // ideal-gas caloric model.
    auto caloric = std::make_shared<
        thermomodel::ConstantCp>(2200.0);

    thermomodel::PengRobinson model(caloric, params);

    const auto result = model.evaluate_Tp(300.0, 5000000.0);

    if (!result) {
        std::cerr << "FAIL: Peng-Robinson evaluation\n";
        return 1;
    }

    const auto& state = result.value;

    int failures = 0;

    failures += check(
        "Compressibility factor Z",
        state.Z,
        0.90183535320862873
    );

    failures += check(
        "Density [kg/m^3]",
        state.rho,
        35.659306418579362
    );

    std::cout << (failures ? "FAIL" : "PASS")
              << " ThermoGPU frozen methane reference\n";

    return failures ? 1 : 0;
}
