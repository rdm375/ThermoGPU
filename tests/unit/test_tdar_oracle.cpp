// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include "thermogpu/mixture_benchmark.hpp"
#include "thermogpu/tdar_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace thermogpu;

namespace {
bool close(double a, double b, double atol, double rtol) {
    return std::abs(a-b) <= atol + rtol*std::max(std::abs(a), std::abs(b));
}
}

int main() {
    const Component methane{"methane",190.564,4599200.0,0.01142,0.01604246};
    const State state{5.0e6,300.0};
    int fail = 0;
    auto check = [&](bool ok, const char* message) {
        if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++fail; }
    };

    const auto center = evaluate_pure(methane, state);
    const auto r = evaluate_z_oracle(methane, state);
    check(r.Z == center.Z, "oracle Z is the production pure-component Z");
    check(r.root_classification == center.root_classification, "root classification is preserved");
    check(r.cubic_residual == center.cubic_residual, "cubic residual is preserved");
    check(std::isfinite(r.dZ_dT) && std::isfinite(r.dZ_dP), "finite derivatives");
    check(r.dZ_dT > 0.0, "methane Z increases with T at reference state");
    check(r.dZ_dP < 0.0, "methane Z decreases with P at reference state");

    // Step-halving convergence: the default stencil should agree closely with
    // a stencil using half the relative step at this smooth, one-root state.
    const auto half = evaluate_z_oracle(methane, state, 5.0e-6, 5.0e-6);
    check(close(r.dZ_dT, half.dZ_dT, 1e-10, 2e-8), "dZ/dT step-halving agreement");
    check(close(r.dZ_dP, half.dZ_dP, 1e-16, 2e-8), "dZ/dP step-halving agreement");

    // The distinct family is a nested prefix of the canonical C5 composition.
    const double expected[5][5] = {
        {1.0,0,0,0,0},
        {10.0/11.0,1.0/11.0,0,0,0},
        {20.0/23.0,2.0/23.0,1.0/23.0,0,0},
        {5.0/6.0,1.0/12.0,1.0/24.0,1.0/24.0,0},
        {0.80,0.08,0.04,0.04,0.04}
    };
    for (std::size_t n=1; n<=5; ++n) {
        const auto mix=make_distinct_benchmark_mixture(n);
        check(mix.size()==n, "distinct mixture component count");
        for (std::size_t i=0; i<n; ++i)
            check(close(mix.mole_fractions[i],expected[n-1][i],1e-15,1e-15),
                  "distinct mixture mole fraction");
        const auto z=evaluate_z_oracle(mix,state);
        check(std::isfinite(z.Z), "distinct mixture finite Z");
    }
    const auto c5=evaluate_z_oracle(make_distinct_benchmark_mixture(5),state);
    check(close(c5.Z,0.8645668727595472,1e-14,1e-14),
          "distinct C5 reference Z at 300 K, 5 MPa");

    // A small supercritical rectangle intended for the first TDAR experiment
    // must stay on the smooth one-real-root branch.
    for (double T : {250.0, 275.0, 300.0, 325.0, 350.0}) {
        for (double P : {1.0e6, 3.25e6, 5.5e6, 7.75e6, 10.0e6}) {
            const auto q = evaluate_z_oracle(methane, {P,T});
            check(q.root_classification == RootClassification::one_real,
                  "TDAR reference rectangle remains one-real-root");
            check(std::abs(q.cubic_residual) < 1e-12,
                  "TDAR reference rectangle cubic residual");
        }
    }

    return fail ? 1 : 0;
}
