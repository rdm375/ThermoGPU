#pragma once
#include <limits>
namespace thermomodel {
enum class Phase { Vapor, Liquid, TwoPhase, Supercritical, Unknown };
struct State {
    double T = std::numeric_limits<double>::quiet_NaN(); // K
    double p = std::numeric_limits<double>::quiet_NaN(); // Pa
    double rho = std::numeric_limits<double>::quiet_NaN(); // kg/m3
    double h = std::numeric_limits<double>::quiet_NaN(); // J/kg
    double s = std::numeric_limits<double>::quiet_NaN(); // J/(kg K)
    double u = std::numeric_limits<double>::quiet_NaN(); // J/kg
    double cp = std::numeric_limits<double>::quiet_NaN(); // J/(kg K)
    double cv = std::numeric_limits<double>::quiet_NaN(); // J/(kg K)
    double Z = std::numeric_limits<double>::quiet_NaN();
    double sound_speed = std::numeric_limits<double>::quiet_NaN(); // m/s; ideal gas only v0.1
    Phase phase = Phase::Unknown;
};
} // namespace thermomodel
