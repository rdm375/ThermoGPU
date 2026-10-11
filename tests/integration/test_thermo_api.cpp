// SPDX-License-Identifier: GPL-3.0-or-later
 // Copyright (C) 2026 Richard Myers

#include "thermogpu/thermo.hpp"
#include "thermomodel/caloric.hpp"

#include <cmath>
#include <iostream>
#include <memory>

int main()
{
    namespace thermo = thermogpu::thermo;

    auto caloric =
        std::make_shared<thermomodel::ConstantCp>(2200.0);

    thermo::PRParameters parameters;

    thermo::PengRobinson model(caloric, parameters);

    const double T = 300.0;
    const double p = 5.0e6;

    const thermo::Result<thermo::State> result =
        model.evaluate_Tp(T, p);

    if (!result) {
        std::cerr << "Thermodynamic evaluation failed\n";
        return 1;
    }

    const auto& state = result.value;

    if (!std::isfinite(state.Z) ||
        !std::isfinite(state.rho) ||
        !std::isfinite(state.h) ||
        !std::isfinite(state.s) ||
        !std::isfinite(state.cp) ||
        !std::isfinite(state.cv) ||
        !std::isfinite(state.sound_speed)) {

        std::cerr << "Non-finite thermodynamic property\n";
        return 1;
    }

    const auto derivatives =
        model.derivatives_Tp(T, p);

    if (!derivatives) {
        std::cerr << "Thermodynamic derivatives failed\n";
        return 1;
    }

    std::cout
        << "Z = " << state.Z << '\n'
        << "rho = " << state.rho << " kg/m^3\n"
        << "h = " << state.h << " J/kg\n"
        << "s = " << state.s << " J/(kg K)\n"
        << "cp = " << state.cp << " J/(kg K)\n"
        << "cv = " << state.cv << " J/(kg K)\n"
        << "sound speed = "
        << state.sound_speed << " m/s\n";

    return 0;
}
