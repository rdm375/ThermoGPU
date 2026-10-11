// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/thermo_batch.hpp"
#include "thermomodel/caloric.hpp"

#include <stdexcept>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

int main()
{
    namespace thermo = thermogpu::thermo;

    auto caloric =
        std::make_shared<thermomodel::ConstantCp>(2200.0);

    thermo::PengRobinson model(caloric);

    // Empty batch.
    {
        const auto result =
            thermo::evaluate_batch_scalar(model, {});

        require(result.size() == 0, "batch validation failure");
    }

    // Valid and invalid states in one batch.
    {
        thermo::BatchInput input{
            {300.0, 320.0, -1.0, 5000.0, 350.0},
            {5.0e6, 8.0e6, 1.0e5, 1.0e5, -1.0}
        };

        const auto batch =
            thermo::evaluate_batch_scalar(model, input);

        require(batch.size() == input.size(), "batch validation failure");

        for (std::size_t i = 0; i < input.size(); ++i) {

            const auto scalar = model.evaluate_Tp(
                input.temperature_K[i],
                input.pressure_Pa[i]
            );

            require(batch.error[i] == scalar.error, "batch validation failure");
            require(batch.message[i] == scalar.message, "batch validation failure");

            if (!scalar) {
                require(std::isnan(batch.Z[i]), "batch validation failure");
                require(std::isnan(batch.rho[i]), "batch validation failure");
                require(std::isnan(batch.h[i]), "batch validation failure");
                require(std::isnan(batch.s[i]), "batch validation failure");
                continue;
            }

            const auto& state = scalar.value;

            require(batch.Z[i] == state.Z, "batch validation failure");
            require(batch.rho[i] == state.rho, "batch validation failure");
            require(batch.h[i] == state.h, "batch validation failure");
            require(batch.s[i] == state.s, "batch validation failure");
            require(batch.u[i] == state.u, "batch validation failure");
            require(batch.cp[i] == state.cp, "batch validation failure");
            require(batch.cv[i] == state.cv, "batch validation failure");
            require(batch.sound_speed[i] == state.sound_speed, "batch validation failure");
            require(batch.phase[i] == state.phase, "batch validation failure");
        }

        require(batch.success(0), "batch validation failure");
        require(batch.success(1), "batch validation failure");

        require(
            batch.error[2] ==
            thermomodel::ErrorCode::InvalidInput
        , "batch validation failure");

        require(
            batch.error[3] ==
            thermomodel::ErrorCode::OutOfRange
        , "batch validation failure");

        require(
            batch.error[4] ==
            thermomodel::ErrorCode::InvalidInput
        , "batch validation failure");
    }

    // Mismatched input lengths.
    {
        bool caught = false;

        try {
            thermo::BatchInput input{
                {300.0, 320.0},
                {5.0e6}
            };

            (void)thermo::evaluate_batch_scalar(model, input);

        } catch (const std::invalid_argument&) {
            caught = true;
        }

        require(caught, "batch validation failure");
    }

    return 0;
}
