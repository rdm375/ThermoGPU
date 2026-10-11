#pragma once
#include "model.hpp"
namespace thermomodel {
struct FlashOptions {
    double Tmin = 100.0, Tmax = 3000.0;
    double relative_tolerance = 1e-11;
    int max_iterations = 200;
};
// Generic, single-phase, monotonic-in-temperature inversion at fixed pressure.
// A two-phase backend requires a specialized equilibrium flash.
Result<State> flash_ph(const Model& model, double p, double h, const Composition& x = {}, FlashOptions options = {});
Result<State> flash_ps(const Model& model, double p, double s, const Composition& x = {}, FlashOptions options = {});
} // namespace thermomodel
