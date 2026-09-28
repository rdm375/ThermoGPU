// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include "component.hpp"
#include <cstddef>
#include <vector>

namespace thermogpu {

struct Mixture {
    std::vector<Component> components;
    std::vector<double> mole_fractions;
    // Dense row-major n x n matrix.  V1.0 keeps this explicit even when all kij are zero.
    std::vector<double> binary_interactions;

    [[nodiscard]] std::size_t size() const noexcept { return components.size(); }
    [[nodiscard]] double kij(std::size_t i, std::size_t j) const;
};

void validate_mixture(const Mixture& mixture, double composition_tolerance = 1.0e-12);

} // namespace thermogpu
