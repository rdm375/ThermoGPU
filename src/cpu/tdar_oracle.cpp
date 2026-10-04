// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/tdar_oracle.hpp"
#include "thermogpu/peng_robinson.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace thermogpu {
namespace {

double step(double x, double relative_step) {
    if (!(relative_step > 0.0) || !std::isfinite(relative_step)) {
        throw std::invalid_argument("finite-difference relative step must be finite and positive");
    }
    return relative_step * std::max(std::abs(x), 1.0);
}

} // namespace

ZOracleResult evaluate_z_oracle(
    const Component& component,
    const State& state,
    double relative_temperature_step,
    double relative_pressure_step) {

    const double hT = step(state.temperature_K, relative_temperature_step);
    const double hP = step(state.pressure_Pa, relative_pressure_step);
    if (!(state.temperature_K > hT) || !(state.pressure_Pa > hP)) {
        throw std::invalid_argument("finite-difference stencil crosses non-positive T or P");
    }

    const auto center = evaluate_pure(component, state);
    const auto t_minus = evaluate_pure(component, {state.pressure_Pa, state.temperature_K - hT});
    const auto t_plus  = evaluate_pure(component, {state.pressure_Pa, state.temperature_K + hT});
    const auto p_minus = evaluate_pure(component, {state.pressure_Pa - hP, state.temperature_K});
    const auto p_plus  = evaluate_pure(component, {state.pressure_Pa + hP, state.temperature_K});

    return {
        center.Z,
        (t_plus.Z - t_minus.Z) / (2.0 * hT),
        (p_plus.Z - p_minus.Z) / (2.0 * hP),
        center.cubic_residual,
        center.root_classification,
    };
}


ZOracleResult evaluate_z_oracle(
    const Mixture& mixture, const State& state,
    double relative_temperature_step, double relative_pressure_step) {
    const double hT = step(state.temperature_K, relative_temperature_step);
    const double hP = step(state.pressure_Pa, relative_pressure_step);
    if (!(state.temperature_K > hT) || !(state.pressure_Pa > hP))
        throw std::invalid_argument("finite-difference stencil crosses non-positive T or P");
    const auto center = evaluate_mixture(mixture, state);
    const auto tm = evaluate_mixture(mixture, {state.pressure_Pa, state.temperature_K-hT});
    const auto tp = evaluate_mixture(mixture, {state.pressure_Pa, state.temperature_K+hT});
    const auto pm = evaluate_mixture(mixture, {state.pressure_Pa-hP, state.temperature_K});
    const auto pp = evaluate_mixture(mixture, {state.pressure_Pa+hP, state.temperature_K});
    return {center.Z, (tp.Z-tm.Z)/(2*hT), (pp.Z-pm.Z)/(2*hP),
            center.cubic_residual, center.root_classification};
}

} // namespace thermogpu
