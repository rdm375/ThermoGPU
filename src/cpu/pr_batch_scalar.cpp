// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/peng_robinson.hpp"
#include <cmath>
#include <stdexcept>

namespace thermogpu {

void validate_batch(const Mixture& mixture, const MixtureBatch& batch,
                    double composition_tolerance) {
    const std::size_t ncomp = mixture.size();
    if (ncomp == 0) throw std::invalid_argument("batch mixture must contain at least one component");
    if (batch.temperature_K.size() != batch.size())
        throw std::invalid_argument("batch pressure and temperature arrays must have equal length");
    if (batch.mole_fractions.size() != batch.size() * ncomp)
        throw std::invalid_argument("batch composition array must contain state_count * component_count values");
    if (!(composition_tolerance >= 0.0) || !std::isfinite(composition_tolerance))
        throw std::invalid_argument("invalid batch composition tolerance");

    for (std::size_t s = 0; s < batch.size(); ++s) {
        const double P = batch.pressure_Pa[s];
        const double T = batch.temperature_K[s];
        if (!std::isfinite(P) || !std::isfinite(T) || P <= 0.0 || T <= 0.0)
            throw std::invalid_argument("batch P and T values must be finite and positive");

        double sum = 0.0;
        for (std::size_t i = 0; i < ncomp; ++i) {
            const double z = batch.mole_fractions[s*ncomp+i];
            if (!std::isfinite(z) || z < 0.0 || z > 1.0)
                throw std::invalid_argument("batch mole fractions must be finite and in [0,1]");
            sum += z;
        }
        if (std::abs(sum - 1.0) > composition_tolerance)
            throw std::invalid_argument("each batch composition must sum to one; no silent normalization is performed");
    }
}

MixtureBatchResult evaluate_mixture_batch_scalar(const Mixture& mixture,
                                                 const MixtureBatch& batch) {
    // Validate the invariant component/kij part once.  Batch compositions are
    // validated separately because they intentionally vary by state.
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

    Mixture work = mixture;
    work.mole_fractions.resize(ncomp);
    for (std::size_t s = 0; s < nstate; ++s) {
        for (std::size_t i = 0; i < ncomp; ++i)
            work.mole_fractions[i] = batch.mole_fractions[s*ncomp+i];
        const MixtureResult r = evaluate_mixture(work, State{batch.pressure_Pa[s], batch.temperature_K[s]});
        out.Z[s] = r.Z;
        out.density_kg_per_m3[s] = r.density_kg_per_m3;
        out.cubic_residual[s] = r.cubic_residual;
        out.root_classification[s] = r.root_classification;
        out.root_selection_ambiguous[s] = static_cast<unsigned char>(r.root_selection_ambiguous);
        for (std::size_t i = 0; i < ncomp; ++i)
            out.ln_phi[s*ncomp+i] = r.ln_phi[i];
    }
    return out;
}

} // namespace thermogpu
