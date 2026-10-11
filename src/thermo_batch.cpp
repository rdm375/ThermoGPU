// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/thermo_batch.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef THERMOGPU_HAS_OPENMP
#include <omp.h>
#endif

namespace thermogpu::thermo {

namespace {

std::size_t validate_input(const BatchInput& input)
{
    const std::size_t n = input.size();

    if (input.pressure_Pa.size() != n) {
        throw std::invalid_argument(
            "temperature and pressure batch sizes differ"
        );
    }

    return n;
}

BatchResult make_result(std::size_t n)
{
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

    return result;
}

void evaluate_one(
    const Model& model,
    const BatchInput& input,
    BatchResult& result,
    std::size_t i
)
{
    const auto evaluated = model.evaluate_Tp(
        input.temperature_K[i],
        input.pressure_Pa[i]
    );

    result.error[i] = evaluated.error;
    result.message[i] = evaluated.message;

    if (!evaluated) {
        return;
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

} // anonymous namespace

BatchResult evaluate_batch_scalar(
    const Model& model,
    const BatchInput& input
)
{
    const std::size_t n = validate_input(input);

    BatchResult result = make_result(n);

    for (std::size_t i = 0; i < n; ++i) {
        evaluate_one(model, input, result, i);
    }

    return result;
}

#ifdef THERMOGPU_HAS_OPENMP

BatchResult evaluate_batch_openmp(
    const Model& model,
    const BatchInput& input
)
{
    const std::size_t n = validate_input(input);

    BatchResult result = make_result(n);

    // OpenMP's canonical loop uses a signed integer type.
    if (n > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
        throw std::length_error("thermodynamic batch too large");
    }

    const auto count = static_cast<std::int64_t>(n);

    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < count; ++i) {
        evaluate_one(
            model,
            input,
            result,
            static_cast<std::size_t>(i)
        );
    }

    return result;
}

#endif

} // namespace thermogpu::thermo
