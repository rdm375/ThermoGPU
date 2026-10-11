// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/thermo_batch.hpp"
#include "thermomodel/caloric.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare(
    const thermogpu::thermo::BatchResult& a,
    const thermogpu::thermo::BatchResult& b
)
{
    require(a.size() == b.size(), "batch size mismatch");

    for (std::size_t i = 0; i < a.size(); ++i) {

        require(a.error[i] == b.error[i], "error code mismatch");
        require(a.message[i] == b.message[i], "error message mismatch");
        require(a.phase[i] == b.phase[i], "phase mismatch");

        if (!a.success(i)) {
            require(std::isnan(a.Z[i]), "scalar failure not NaN");
            require(std::isnan(b.Z[i]), "OpenMP failure not NaN");
            continue;
        }

        require(a.Z[i] == b.Z[i], "Z mismatch");
        require(a.rho[i] == b.rho[i], "density mismatch");
        require(a.h[i] == b.h[i], "enthalpy mismatch");
        require(a.s[i] == b.s[i], "entropy mismatch");
        require(a.u[i] == b.u[i], "internal energy mismatch");
        require(a.cp[i] == b.cp[i], "cp mismatch");
        require(a.cv[i] == b.cv[i], "cv mismatch");
        require(
            a.sound_speed[i] == b.sound_speed[i],
            "sound speed mismatch"
        );
    }
}

} // anonymous namespace

int main()
{
    namespace thermo = thermogpu::thermo;

    auto caloric =
        std::make_shared<thermomodel::ConstantCp>(2200.0);

    thermo::PengRobinson model(caloric);

    constexpr std::size_t n = 10000;

    thermo::BatchInput input;
    input.temperature_K.resize(n);
    input.pressure_Pa.resize(n);

    for (std::size_t i = 0; i < n; ++i) {
        input.temperature_K[i] =
            200.0 + 300.0 * static_cast<double>(i % 101) / 100.0;

        input.pressure_Pa[i] =
            1.0e5 + 19.9e6 * static_cast<double>(i % 97) / 96.0;
    }

    // Deliberately introduce independent failures.
    input.temperature_K[17] = -1.0;
    input.temperature_K[211] = 5000.0;
    input.pressure_Pa[499] = -1.0;

    const auto scalar =
        thermo::evaluate_batch_scalar(model, input);

    for (int threads : {1, 2, 4, 8}) {
        omp_set_num_threads(threads);

        const auto parallel =
            thermo::evaluate_batch_openmp(model, input);

        compare(scalar, parallel);
    }

    // Empty batches must also work.
    const auto empty =
        thermo::evaluate_batch_openmp(model, {});

    require(empty.size() == 0, "empty batch mismatch");

    // Dimension mismatch must be rejected before parallel execution.
    bool caught = false;

    try {
        thermo::BatchInput bad{
            {300.0, 320.0},
            {5.0e6}
        };

        (void)thermo::evaluate_batch_openmp(model, bad);

    } catch (const std::invalid_argument&) {
        caught = true;
    }

    require(caught, "dimension mismatch not rejected");

    return 0;
}
