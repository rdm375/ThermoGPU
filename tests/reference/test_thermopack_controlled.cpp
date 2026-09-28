// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using namespace thermogpu;

namespace {
bool close(double got, double ref, double rtol=5e-14, double atol=5e-15) {
    return std::abs(got-ref) <= atol + rtol*std::abs(ref);
}

int check(const std::string& name, double got, double ref) {
    if (close(got, ref)) return 0;
    std::cerr << std::setprecision(17)
              << "FAIL " << name << "\n  ThermoGPU " << got
              << "\n  ThermoPack " << ref
              << "\n  delta      " << (got-ref) << '\n';
    return 1;
}

Component methane(){return {"methane",190.56,4.5992e6,0.01142,0.016043};}
Component ethane(){return {"ethane",305.32,4.872e6,0.09950,0.030070};}
}

int main() {
    int failures=0;

    // ThermoPack 2.2.3, PR/vdW/Classic, genuine PSEUDO slot initialized with
    // exactly the ThermoGPU component constants. Vapor root at 300 K, 5 MPa.
    {
        const auto r=evaluate_pure(methane(), {5.0e6,300.0});
        failures += check("pure methane Z", r.Z, 9.0183535320862862e-01);
        failures += check("pure methane lnphi", r.ln_phi, -1.0383023649485131e-01);
    }

    // ThermoPack 2.2.3 controlled CH4/C2H6 pseudo-component case, PR/vdW/Classic,
    // z=(0.8,0.2), kij12=0, vapor root at 300 K, 5 MPa.
    {
        Mixture m{{methane(),ethane()},{0.8,0.2},{0.0,0.0,0.0,0.0}};
        const auto r=evaluate_mixture(m,{5.0e6,300.0});
        failures += check("binary Z", r.Z, 8.5350705215058376e-01);
        failures += check("binary methane lnphi", r.ln_phi[0], -9.9050653862969157e-02);
        failures += check("binary ethane lnphi", r.ln_phi[1], -3.6002198232155225e-01);
    }

    std::cout << (failures ? "FAIL" : "PASS")
              << " controlled ThermoPack external validation\n";
    return failures ? 1 : 0;
}
