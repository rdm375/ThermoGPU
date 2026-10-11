#pragma once
#include "result.hpp"
#include "state.hpp"
#include <string>
#include <vector>
namespace thermomodel {
struct Composition { std::vector<double> mole_fractions; };
struct Capabilities { bool evaluate_Tp=false; bool evaluate_Trho=false; bool single_phase_ph_flash=false; bool single_phase_ps_flash=false; bool analytic_derivatives=false; };
struct Metadata { std::string name, version, reference_convention; double Tmin, Tmax, pmin, pmax; };
class Model {
public:
    virtual ~Model() = default;
    virtual Result<State> evaluate_Tp(double T, double p, const Composition& x = {}) const = 0;
    virtual Result<State> evaluate_Trho(double T, double rho, const Composition& x = {}) const = 0;
    virtual Capabilities capabilities() const noexcept = 0;
    virtual Metadata metadata() const = 0;
};
} // namespace thermomodel
