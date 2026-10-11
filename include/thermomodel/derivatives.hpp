#pragma once
#include "model.hpp"
namespace thermomodel {
struct Derivatives {
    double drho_dp_T, drho_dT_p;
    double dh_dp_T, dh_dT_p;
    double ds_dp_T, ds_dT_p;
};
// Optional analytic derivative contract; independent of EOS representation.
class DifferentiableModel {
public:
    virtual ~DifferentiableModel() = default;
    virtual Result<Derivatives> derivatives_Tp(double T,double p,const Composition& x = {}) const = 0;
};
} // namespace thermomodel
