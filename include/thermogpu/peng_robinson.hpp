// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include "component.hpp"
#include "diagnostics.hpp"
#include "mixture.hpp"
#include "result.hpp"
#include "state.hpp"

namespace thermogpu {
inline constexpr double gas_constant = 8.31446261815324;
double kappa(double acentric_factor);
double alpha(const Component&, double temperature_K);
double a_parameter(const Component&);
double b_parameter(const Component&);
CubicRoots solve_z_cubic(double A, double B);
double cubic_residual(double Z, double A, double B);
PureResult evaluate_pure(const Component&, const State&);
MixtureResult evaluate_mixture(const Mixture&, const State&);
MixtureDiagnostics evaluate_mixture_diagnostics(const Mixture&, const State&);
} // namespace thermogpu
