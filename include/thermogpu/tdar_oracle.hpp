// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once

#include "component.hpp"
#include "mixture.hpp"
#include "result.hpp"
#include "state.hpp"

namespace thermogpu {

struct ZOracleResult {
    double Z{};
    double dZ_dT{};
    double dZ_dP{};
    double cubic_residual{};
    RootClassification root_classification{RootClassification::ambiguous};
};

// Evaluate Z and centered finite-difference derivatives for a pure component.
// Relative steps are scaled by max(|coordinate|, 1); callers may override them
// for convergence studies. The center and all stencil states must be positive.
ZOracleResult evaluate_z_oracle(
    const Component& component,
    const State& state,
    double relative_temperature_step = 1.0e-5,
    double relative_pressure_step = 1.0e-5);

// Same oracle contract for a fixed-composition mixture.
ZOracleResult evaluate_z_oracle(
    const Mixture& mixture,
    const State& state,
    double relative_temperature_step = 1.0e-5,
    double relative_pressure_step = 1.0e-5);

} // namespace thermogpu
