// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>

using namespace thermogpu;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void require_same(double a, double b, const std::string& message) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    if (std::abs(a-b) > 2.0e-14*scale)
        throw std::runtime_error(message + ": " + std::to_string(a) + " vs " + std::to_string(b));
}
Mixture binary() {
    return {
        {{"methane",190.56,4.5992e6,0.01142,0.016043},
         {"ethane",305.32,4.872e6,0.09950,0.030070}},
        {0.8,0.2},
        {0.0,0.0,0.0,0.0}
    };
}
MixtureBatch campaign(std::size_t n) {
    MixtureBatch b;
    b.pressure_Pa.reserve(n);
    b.temperature_K.reserve(n);
    b.mole_fractions.reserve(2*n);
    for (std::size_t s=0; s<n; ++s) {
        // Deterministic coverage: P 0.1--12 MPa, T 240--420 K, x_CH4 0.05--0.95.
        const double fp = static_cast<double>((s * 37) % 997) / 996.0;
        const double ft = static_cast<double>((s * 61) % 991) / 990.0;
        const double fx = static_cast<double>((s * 83) % 983) / 982.0;
        const double x = 0.05 + 0.90*fx;
        b.pressure_Pa.push_back(1.0e5 + fp*11.9e6);
        b.temperature_K.push_back(240.0 + ft*180.0);
        b.mole_fractions.push_back(x);
        b.mole_fractions.push_back(1.0-x);
    }
    return b;
}
}

int main() {
    try {
        constexpr std::size_t nstate = 1000;
        const Mixture m = binary();
        const MixtureBatch b = campaign(nstate);
        const auto scalar = evaluate_mixture_batch_scalar(m, b);
        const auto parallel = evaluate_mixture_batch_openmp(m, b);

        require(parallel.state_count == nstate, "OpenMP state count mismatch");
        require(parallel.component_count == 2, "OpenMP component count mismatch");
        for (std::size_t s=0; s<nstate; ++s) {
            require_same(parallel.Z[s], scalar.Z[s], "OpenMP/scalar Z mismatch at state " + std::to_string(s));
            require_same(parallel.density_kg_per_m3[s], scalar.density_kg_per_m3[s],
                         "OpenMP/scalar density mismatch at state " + std::to_string(s));
            require_same(parallel.cubic_residual[s], scalar.cubic_residual[s],
                         "OpenMP/scalar residual mismatch at state " + std::to_string(s));
            require(parallel.root_classification[s] == scalar.root_classification[s],
                    "OpenMP/scalar root classification mismatch at state " + std::to_string(s));
            require(parallel.root_selection_ambiguous[s] == scalar.root_selection_ambiguous[s],
                    "OpenMP/scalar ambiguity mismatch at state " + std::to_string(s));
            for (std::size_t i=0; i<2; ++i)
                require_same(parallel.ln_phi[2*s+i], scalar.ln_phi[2*s+i],
                             "OpenMP/scalar ln_phi mismatch at state " + std::to_string(s));
        }
        std::cout << "M4 OpenMP differential tests passed: " << nstate
                  << " states, max_threads=" << omp_get_max_threads() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
