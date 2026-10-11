// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#pragma once

#include "thermogpu/thermo.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace thermogpu::thermo {

struct BatchInput {
    std::vector<double> temperature_K;
    std::vector<double> pressure_Pa;

    [[nodiscard]]
    std::size_t size() const noexcept {
        return temperature_K.size();
    }
};

struct BatchResult {
    std::vector<double> Z;
    std::vector<double> rho;
    std::vector<double> h;
    std::vector<double> s;
    std::vector<double> u;
    std::vector<double> cp;
    std::vector<double> cv;
    std::vector<double> sound_speed;

    std::vector<thermomodel::Phase> phase;
    std::vector<thermomodel::ErrorCode> error;
    std::vector<std::string> message;

    [[nodiscard]]
    std::size_t size() const noexcept {
        return Z.size();
    }

    [[nodiscard]]
    bool success(std::size_t i) const {
        return error.at(i) == thermomodel::ErrorCode::None;
    }
};

BatchResult evaluate_batch_scalar(
    const Model& model,
    const BatchInput& input
);

} // namespace thermogpu::thermo
