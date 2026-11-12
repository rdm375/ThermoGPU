#pragma once
#include "caloric.hpp"
#include "derivatives.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>
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

// Fixed-composition multicomponent Peng-Robinson model. Component cp values
// are ideal-gas mass-specific constants; k_ij is dense, symmetric, row-major.
struct PRMixtureComponent {
    std::string name;
    double mole_fraction = 0.0;
    double molar_mass = 0.0; // kg/mol
    double Tc = 0.0;         // K
    double pc = 0.0;         // Pa
    double acentric = 0.0;
    double ideal_cp = 0.0;   // J/(kg K)
};

class PengRobinsonMixture final : public Model, public DifferentiableModel {
public:
    PengRobinsonMixture(std::vector<PRMixtureComponent> components,
                        std::vector<double> binary_interactions = {},
                        double Tref = 298.15, double pref = 101325.0,
                        double Tmin = 150.0, double Tmax = 1000.0,
                        double pmin = 100.0, double pmax = 100.0e6,
                        PRRootPolicy root_policy = PRRootPolicy::Vapor);
    Result<State> evaluate_Tp(double T,double p,const Composition& x={}) const override;
    Result<State> evaluate_Trho(double T,double rho,const Composition& x={}) const override;
    Result<Derivatives> derivatives_Tp(double T,double p,const Composition& x={}) const override;
    Capabilities capabilities() const noexcept override { return {true,true,false,false,true}; }
    Metadata metadata() const override;
    const std::vector<PRMixtureComponent>& components() const noexcept { return components_; }
    double mixture_molar_mass() const noexcept { return molar_mass_; }
private:
    struct Local { State state; Derivatives deriv; };
    struct MixAtT { double a, ap, app, b; };
    MixAtT parameters_at(double T) const;
    Result<Local> from_volume(double T,double v,double p) const;
    bool composition_matches(const Composition&) const;
    std::vector<PRMixtureComponent> components_;
    std::vector<double> kij_;
    double Tref_,pref_,Tmin_,Tmax_,pmin_,pmax_,molar_mass_,gas_constant_,ideal_cp_;
    PRRootPolicy root_policy_;
};
} // namespace thermomodel
