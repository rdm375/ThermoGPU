// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace thermogpu;

namespace {
const char* classification_name(RootClassification c) {
    switch (c) {
        case RootClassification::one_real: return "one real root";
        case RootClassification::three_real: return "three real roots";
        default: return "ambiguous";
    }
}
}

int main() {
    const Component methane{"methane",190.564,4599200.0,0.01142,0.01604246};
    const Component ethane{"ethane",305.322,4872200.0,0.0995,0.03006904};
    const Component propane{"propane",369.83,4248000.0,0.1523,0.04409562};
    const Component nitrogen{"nitrogen",126.192,3395800.0,0.0372,0.0280134};
    const Component co2{"carbon_dioxide",304.1282,7377300.0,0.22394,0.0440095};

    const Mixture gas{{methane,ethane,propane,nitrogen,co2},
                      {0.80,0.08,0.04,0.04,0.04},
                      std::vector<double>(25,0.0)};
    const State state{8.0e6,320.0};
    const auto result = evaluate_mixture(gas,state);
    const auto roots = solve_z_cubic(result.A,result.B);

    std::cout << std::setprecision(15);
    std::cout << "ThermoGPU scalar Peng-Robinson evaluation\n\n";
    std::cout << "State\n"
              << "  P = " << state.pressure_Pa << " Pa\n"
              << "  T = " << state.temperature_K << " K\n\n";

    std::cout << "Composition\n";
    for (std::size_t i=0;i<gas.size();++i)
        std::cout << "  " << std::left << std::setw(18) << gas.components[i].name
                  << std::right << gas.mole_fractions[i] << '\n';

    std::cout << "\nMixture\n"
              << "  molecular weight = " << result.molar_mass_kg_per_mol << " kg/mol\n"
              << "  a_m              = " << result.a_m << '\n'
              << "  b_m              = " << result.b_m << " m^3/mol\n"
              << "  A                = " << result.A << '\n'
              << "  B                = " << result.B << "\n\n";

    std::cout << "Cubic solution\n"
              << "  classification   = " << classification_name(roots.classification) << '\n'
              << "  real root count  = " << roots.count << '\n';
    for (std::size_t i=0;i<roots.count;++i)
        std::cout << "  root[" << i << "]          = " << roots.values[i] << '\n';
    std::cout << "  selected Z       = " << result.Z << '\n'
              << "  selection status = " << (result.root_selection_ambiguous ? "AMBIGUOUS (largest root selected)" : "unambiguous") << '\n'
              << "  cubic residual   = " << result.cubic_residual << "\n\n";

    std::cout << "Properties\n"
              << "  density          = " << result.density_kg_per_m3 << " kg/m^3\n\n";

    std::cout << "Fugacity coefficients\n";
    std::cout << "  " << std::left << std::setw(18) << "component" << std::right
              << std::setw(22) << "ln(phi)" << std::setw(22) << "phi" << '\n';
    for (std::size_t i=0;i<gas.size();++i)
        std::cout << "  " << std::left << std::setw(18) << gas.components[i].name << std::right
                  << std::setw(22) << result.ln_phi[i]
                  << std::setw(22) << std::exp(result.ln_phi[i]) << '\n';
}
