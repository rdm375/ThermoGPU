#pragma once
#include "caloric.hpp"
#include "derivatives.hpp"
#include <memory>
namespace thermomodel {
enum class PRRootPolicy { Vapor, Liquid, RejectMultiple };
struct PRParameters {
    // Methane example defaults. SI mass-specific basis.
    double molar_mass = 0.016043; // kg/mol
    double Tc = 190.56;           // K
    double pc = 4599200.0;         // Pa
    double acentric = 0.01142;
    double Tref = 298.15, pref = 101325.0;
    double Tmin = 100.0, Tmax = 1200.0;
    double pmin = 100.0, pmax = 100e6;
    PRRootPolicy root_policy = PRRootPolicy::Vapor;
};
class PengRobinson final : public Model, public DifferentiableModel {
public:
    // Caloric model is the *ideal-gas* cp(T) on a mass basis.
    explicit PengRobinson(std::shared_ptr<const CaloricModel> ideal_cp,
                          PRParameters params = {});
    Result<State> evaluate_Tp(double T,double p,const Composition& x={}) const override;
    Result<State> evaluate_Trho(double T,double rho,const Composition& x={}) const override;
    Result<Derivatives> derivatives_Tp(double T,double p,const Composition& x={}) const override;
    Capabilities capabilities() const noexcept override {return {true,true,false,false,true};}
    Metadata metadata() const override;
    const PRParameters& parameters() const noexcept {return params_;}
private:
    struct Local { State state; Derivatives deriv; };
    Result<Local> from_volume(double T,double v,double p) const;
    std::shared_ptr<const CaloricModel> caloric_;
    PRParameters params_;
    double R_,a0_,b_,kappa_;
};
} // namespace thermomodel
