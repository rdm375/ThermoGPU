// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include "result.hpp"
#include <vector>

namespace thermogpu {

struct MixtureDiagnostics {
    MixtureResult result;
    std::vector<double> alpha;
    std::vector<double> a;
    std::vector<double> a_alpha;
    std::vector<double> b;
    std::vector<double> sum_z_aij;
    CubicRoots roots;
};

} // namespace thermogpu
