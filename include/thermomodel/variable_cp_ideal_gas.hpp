#pragma once
#include "caloric.hpp"
#include "derivatives.hpp"
#include <memory>
namespace thermomodel {
struct VariableCpParameters {
    double R=287.05, Tref=298.15, pref=101325.0;
    double Tmin=100.0,Tmax=3000.0,pmin=100.0,pmax=1e9;
};
class VariableCpIdealGas final : public Model, public DifferentiableModel {
public:
    VariableCpIdealGas(std::shared_ptr<const CaloricModel> caloric, VariableCpParameters params={});
    Result<State> evaluate_Tp(double T,double p,const Composition& x={}) const override;
    Result<State> evaluate_Trho(double T,double rho,const Composition& x={}) const override;
    Result<Derivatives> derivatives_Tp(double T,double p,const Composition& x={}) const override;
    Capabilities capabilities() const noexcept override {return {true,true,true,true,true};}
    Metadata metadata() const override;
private:
    std::shared_ptr<const CaloricModel> caloric_;
    VariableCpParameters params_;
};
} // namespace thermomodel
