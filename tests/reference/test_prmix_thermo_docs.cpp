// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace thermogpu;

namespace {
bool rel_close(double got, double ref, double rtol, double atol = 0.0) {
    return std::abs(got-ref) <= atol + rtol*std::abs(ref);
}
}

int main() {
    // External regression case from thermo 0.6.1 PRMIX documentation:
    // N2/CH4 = 0.5/0.5, T=115 K, P=1 MPa, kij=0.
    // https://thermo.readthedocs.io/thermo.eos_mix.html
    // Published: V_l=3.6257362939e-05 m3/mol, V_g=7.0066592313e-04 m3/mol
    // fugacities_l=[793860.83821, 73468.552253] Pa
    // fugacities_g=[436530.92470, 358114.63827] Pa
    const Component nitrogen{"nitrogen",126.1,33.94e5,0.04,0.0280134};
    const Component methane{"methane",190.6,46.04e5,0.011,0.01604246};
    const Mixture mix{{nitrogen,methane},{0.5,0.5},{0.0,0.0,0.0,0.0}};
    const State state{1.0e6,115.0};

    const auto result = evaluate_mixture(mix,state);
    const auto roots = solve_z_cubic(result.A,result.B);
    int failures = 0;
    auto check = [&](bool ok, const char* what, double got, double ref) {
        if (!ok) {
            std::cerr << std::setprecision(17) << "FAIL " << what
                      << ": got=" << got << " ref=" << ref << '\n';
            ++failures;
        }
    };

    if (roots.count != 3) {
        std::cerr << "FAIL expected three real PR roots, got " << roots.count << '\n';
        ++failures;
    } else {
        const double Vl = roots.values[0]*gas_constant*state.temperature_K/state.pressure_Pa;
        const double Vg = roots.values[2]*gas_constant*state.temperature_K/state.pressure_Pa;
        // The documentation values are rounded and its constants are not guaranteed
        // identical to ThermoGPU's, so tolerances reflect an external regression test,
        // not a same-code bitwise comparison.
        check(rel_close(Vg,7.0066592313e-4,2e-7),"vapor molar volume",Vg,7.0066592313e-4);
        check(rel_close(Vl,3.6257362939e-5,1e-4),"liquid molar volume",Vl,3.6257362939e-5);
    }

    const double fg_n2 = 0.5*state.pressure_Pa*std::exp(result.ln_phi[0]);
    const double fg_ch4 = 0.5*state.pressure_Pa*std::exp(result.ln_phi[1]);
    check(rel_close(fg_n2,436530.92470,1e-6),"N2 vapor fugacity",fg_n2,436530.92470);
    check(rel_close(fg_ch4,358114.63827,5e-6),"CH4 vapor fugacity",fg_ch4,358114.63827);
    check(std::abs(result.cubic_residual)<1e-12,"selected-root cubic residual",result.cubic_residual,0.0);

    if (!failures) {
        std::cout << "PASS external PRMIX regression (thermo 0.6.1 documentation)\n";
        std::cout << std::setprecision(15)
                  << "  Z_g      = " << result.Z << '\n'
                  << "  f_N2     = " << fg_n2 << " Pa\n"
                  << "  f_CH4    = " << fg_ch4 << " Pa\n";
    }
    return failures ? 1 : 0;
}
