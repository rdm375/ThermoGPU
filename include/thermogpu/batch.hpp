// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once
#include "mixture.hpp"
#include "result.hpp"
#include <cstddef>
#include <vector>

namespace thermogpu {

// Structure-of-arrays batch layout.  Composition is flattened state-major:
// mole_fractions[state * component_count + component].
struct MixtureBatch {
    std::vector<double> pressure_Pa;
    std::vector<double> temperature_K;
    std::vector<double> mole_fractions;

    [[nodiscard]] std::size_t size() const noexcept { return pressure_Pa.size(); }
};

struct MixtureBatchResult {
    std::size_t state_count{};
    std::size_t component_count{};
    std::vector<double> Z;
    std::vector<double> density_kg_per_m3;
    std::vector<double> cubic_residual;
    std::vector<double> ln_phi; // state-major, same indexing as input composition
    std::vector<RootClassification> root_classification;
    std::vector<unsigned char> root_selection_ambiguous;
};

void validate_batch(const Mixture& mixture, const MixtureBatch& batch,
                    double composition_tolerance = 1.0e-12);
MixtureBatchResult evaluate_mixture_batch_scalar(const Mixture& mixture,
                                                 const MixtureBatch& batch);
#ifdef THERMOGPU_HAS_OPENMP
MixtureBatchResult evaluate_mixture_batch_openmp(const Mixture& mixture,
                                                 const MixtureBatch& batch);
#endif
#ifdef THERMOGPU_HAS_CUDA
MixtureBatchResult evaluate_mixture_batch_cuda(const Mixture& mixture,
                                               const MixtureBatch& batch);
#endif

} // namespace thermogpu
