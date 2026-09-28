// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/peng_robinson.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace thermogpu;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void require_same(double a, double b, const std::string& message) {
    const double scale = std::max({1.0, std::abs(a), std::abs(b)});
    if (std::abs(a-b) > 2.0e-14*scale) throw std::runtime_error(message);
}
Mixture binary() {
    return {
        {{"methane",190.56,4.5992e6,0.01142,0.016043},
         {"ethane",305.32,4.872e6,0.09950,0.030070}},
        {0.8,0.2},
        {0.0,0.0,0.0,0.0}
    };
}
}

int main() {
    try {
        Mixture m = binary();
        MixtureBatch batch{
            {1.0e5, 5.0e6, 8.0e6, 2.5e6},
            {300.0, 300.0, 340.0, 260.0},
            {1.0,0.0,
             0.8,0.2,
             0.5,0.5,
             0.2,0.8}
        };
        const auto br = evaluate_mixture_batch_scalar(m, batch);
        require(br.state_count == 4 && br.component_count == 2, "batch result dimensions");
        require(br.ln_phi.size() == 8, "flattened ln_phi size");

        for (std::size_t s=0; s<batch.size(); ++s) {
            m.mole_fractions = {batch.mole_fractions[2*s], batch.mole_fractions[2*s+1]};
            const auto scalar = evaluate_mixture(m, {batch.pressure_Pa[s], batch.temperature_K[s]});
            require_same(br.Z[s], scalar.Z, "batch/scalar Z mismatch");
            require_same(br.density_kg_per_m3[s], scalar.density_kg_per_m3, "batch/scalar density mismatch");
            require_same(br.cubic_residual[s], scalar.cubic_residual, "batch/scalar residual mismatch");
            require(br.root_classification[s] == scalar.root_classification, "batch/scalar root classification mismatch");
            require(static_cast<bool>(br.root_selection_ambiguous[s]) == scalar.root_selection_ambiguous,
                    "batch/scalar ambiguity mismatch");
            for (std::size_t i=0; i<2; ++i)
                require_same(br.ln_phi[2*s+i], scalar.ln_phi[i], "batch/scalar ln_phi mismatch");
        }

        const auto empty = evaluate_mixture_batch_scalar(binary(), MixtureBatch{});
        require(empty.state_count == 0 && empty.Z.empty() && empty.ln_phi.empty(), "empty batch behavior");

        bool threw = false;
        try { evaluate_mixture_batch_scalar(binary(), {{1e5,2e5},{300.0},{1.0,0.0,1.0,0.0}}); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "mismatched P/T arrays must throw");

        threw = false;
        try { evaluate_mixture_batch_scalar(binary(), {{1e5},{300.0},{0.7,0.2}}); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "non-normalized batch composition must throw");

        std::cout << "M3 scalar batch tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
