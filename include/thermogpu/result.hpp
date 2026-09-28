// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include <array>
#include <cstddef>
#include <vector>

namespace thermogpu {

enum class RootClassification { one_real, three_real, ambiguous };

struct CubicRoots {
    std::array<double,3> values{};
    std::size_t count{};
    RootClassification classification{RootClassification::ambiguous};
};

struct PureResult {
    double A{}, B{}, a_alpha{}, b{}, Z{}, density_kg_per_m3{}, ln_phi{};
    double cubic_residual{};
    RootClassification root_classification{RootClassification::ambiguous};
};

struct MixtureResult {
    double A{}, B{}, a_m{}, b_m{}, molar_mass_kg_per_mol{};
    double Z{}, density_kg_per_m3{}, cubic_residual{};
    std::vector<double> ln_phi;
    RootClassification root_classification{RootClassification::ambiguous};
    bool root_selection_ambiguous{};
};

} // namespace thermogpu
