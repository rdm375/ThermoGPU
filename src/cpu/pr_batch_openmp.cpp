// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/peng_robinson.hpp"
#include <omp.h>

namespace thermogpu {

MixtureBatchResult evaluate_mixture_batch_openmp(const Mixture& mixture,
                                                 const MixtureBatch& batch) {
    Mixture invariant = mixture;
    if (invariant.mole_fractions.size() != invariant.size())
        invariant.mole_fractions.assign(invariant.size(), 0.0);
    if (!invariant.mole_fractions.empty()) {
        invariant.mole_fractions.assign(invariant.size(), 0.0);
        invariant.mole_fractions[0] = 1.0;
    }
    validate_mixture(invariant);
    validate_batch(mixture, batch);

    const std::size_t nstate = batch.size();
    const std::size_t ncomp = mixture.size();
    MixtureBatchResult out;
    out.state_count = nstate;
    out.component_count = ncomp;
    out.Z.resize(nstate);
    out.density_kg_per_m3.resize(nstate);
    out.cubic_residual.resize(nstate);
    out.ln_phi.resize(nstate*ncomp);
    out.root_classification.resize(nstate);
    out.root_selection_ambiguous.resize(nstate);

    // Each iteration owns its Mixture copy and writes only its state's output
    // slots.  Component definitions and kij are copied from the immutable input.
#pragma omp parallel for schedule(static)
    for (std::size_t s = 0; s < nstate; ++s) {
        Mixture work = mixture;
        work.mole_fractions.resize(ncomp);
        for (std::size_t i = 0; i < ncomp; ++i)
            work.mole_fractions[i] = batch.mole_fractions[s*ncomp+i];

        const MixtureResult r = evaluate_mixture(
            work, State{batch.pressure_Pa[s], batch.temperature_K[s]});
        out.Z[s] = r.Z;
        out.density_kg_per_m3[s] = r.density_kg_per_m3;
        out.cubic_residual[s] = r.cubic_residual;
        out.root_classification[s] = r.root_classification;
        out.root_selection_ambiguous[s] =
            static_cast<unsigned char>(r.root_selection_ambiguous);
        for (std::size_t i = 0; i < ncomp; ++i)
            out.ln_phi[s*ncomp+i] = r.ln_phi[i];
    }
    return out;
}

} // namespace thermogpu
