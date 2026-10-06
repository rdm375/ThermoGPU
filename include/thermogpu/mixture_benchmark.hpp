// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers
#pragma once
#include "mixture.hpp"
#include <cstddef>

namespace thermogpu {
// Deterministic fixed-composition mixtures for the TDAR/direct-PR crossover study.
// C8/C16 duplicate the C4 pseudo-species while splitting their mole fractions;
// with kij=0 this preserves the C4 thermodynamics while increasing generic PR work.
Mixture make_fixed_benchmark_mixture(std::size_t component_count);

// Nested genuinely-distinct mixtures for studying how physical mixture
// complexity changes the geometry of Z(T,P).  C1..C5 are prefixes of the
// canonical five-component composition (0.80,0.08,0.04,0.04,0.04),
// renormalized at each prefix.
Mixture make_distinct_benchmark_mixture(std::size_t component_count);
}
