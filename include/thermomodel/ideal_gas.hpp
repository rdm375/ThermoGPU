#pragma once
#include "derivatives.hpp"
namespace thermomodel {
struct IdealGasParameters {
    double R = 287.05; // J/(kg K)
    double cp = 1004.5; // J/(kg K)
    double Tref = 298.15; // K
    double pref = 101325.0; // Pa
    double Tmin = 100.0, Tmax = 3000.0;
    double pmin = 100.0, pmax = 1e9;
};
class IdealGas final : public Model, public DifferentiableModel {
public:
    explicit IdealGas(IdealGasParameters params = {});
    Result<State> evaluate_Tp(double T, double p, const Composition& x = {}) const override;
    Result<State> evaluate_Trho(double T, double rho, const Composition& x = {}) const override;
    Capabilities capabilities() const noexcept override { return {true,true,true,true,true}; }
    Metadata metadata() const override;
    Result<Derivatives> derivatives_Tp(double T,double p,const Composition& x={}) const override;
    const IdealGasParameters& parameters() const noexcept { return params_; }
private:
    IdealGasParameters params_;
};
} // namespace thermomodel
