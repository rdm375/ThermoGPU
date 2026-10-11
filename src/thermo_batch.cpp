// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/thermo_batch.hpp"

#include <limits>
#include <stdexcept>

namespace thermogpu::thermo {

BatchResult evaluate_batch_scalar(
    const Model& model,
    const BatchInput& input
) {
    const std::size_t n = input.size();

    if (input.pressure_Pa.size() != n) {
        throw std::invalid_argument(
            "temperature and pressure batch sizes differ"
        );
    }

    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    BatchResult result;

    result.Z.assign(n, nan);
    result.rho.assign(n, nan);
    result.h.assign(n, nan);
    result.s.assign(n, nan);
    result.u.assign(n, nan);
    result.cp.assign(n, nan);
    result.cv.assign(n, nan);
    result.sound_speed.assign(n, nan);

    result.phase.assign(n, thermomodel::Phase::Unknown);
    result.error.assign(n, thermomodel::ErrorCode::None);
    result.message.resize(n);

    for (std::size_t i = 0; i < n; ++i) {

        const auto evaluated = model.evaluate_Tp(
            input.temperature_K[i],
            input.pressure_Pa[i]
        );

        result.error[i] = evaluated.error;
        result.message[i] = evaluated.message;

        if (!evaluated) {
            continue;
        }

        const auto& state = evaluated.value;

        result.Z[i] = state.Z;
        result.rho[i] = state.rho;
        result.h[i] = state.h;
        result.s[i] = state.s;
        result.u[i] = state.u;
        result.cp[i] = state.cp;
        result.cv[i] = state.cv;
        result.sound_speed[i] = state.sound_speed;
        result.phase[i] = state.phase;
    }

    return result;
}

} // namespace thermogpu::thermo
