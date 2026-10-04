// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers
#include "thermogpu/mixture_benchmark.hpp"
#include <array>
#include <stdexcept>

namespace thermogpu {
Mixture make_fixed_benchmark_mixture(std::size_t n) {
    constexpr Component base[] = {
        {"methane", 190.564, 4599200.0, 0.01142, 0.01604246},
        {"ethane", 305.322, 4872200.0, 0.0995, 0.03006904},
        {"propane", 369.83, 4248000.0, 0.1523, 0.04409562},
        {"nitrogen", 126.192, 3395800.0, 0.0372, 0.0280134},
    };
    constexpr double z4[] = {0.70, 0.12, 0.08, 0.10};
    if (n != 1 && n != 2 && n != 4 && n != 8 && n != 16)
        throw std::invalid_argument("benchmark component count must be one of 1,2,4,8,16");
    Mixture m;
    m.components.reserve(n); m.mole_fractions.reserve(n);
    if (n == 1) {
        m.components.push_back(base[0]); m.mole_fractions.push_back(1.0);
    } else if (n == 2) {
        m.components.push_back(base[0]); m.components.push_back(base[1]);
        m.mole_fractions = {0.85, 0.15};
    } else {
        const std::size_t copies = n / 4;
        for (std::size_t k=0; k<copies; ++k)
            for (std::size_t i=0; i<4; ++i) {
                m.components.push_back(base[i]);
                m.mole_fractions.push_back(z4[i] / static_cast<double>(copies));
            }
    }
    m.binary_interactions.assign(n*n, 0.0);
    validate_mixture(m);
    return m;
}
}
