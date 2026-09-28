#pragma once
#include "mixture.hpp"
#include "state.hpp"
#include <cstddef>
namespace thermogpu {
struct CudaPrDiagnostic {
  std::size_t component_count{};
  double kappa[32]{}, alpha[32]{}, a[32]{}, a_alpha[32]{}, b[32]{}, sum_z_aij[32]{};
  double aij[32*32]{};
  double a_m{}, b_m{}, molar_mass{}, A{}, B{};
  double cubic_a{}, cubic_b{}, cubic_c{}, p{}, q{}, discriminant{}, trig_arg{}, theta{};
  double roots[3]{};
  int root_count{};
  double Z{}, density{}, cubic_residual{};
  double ln_phi[32]{};
};
CudaPrDiagnostic diagnose_mixture_cuda(const Mixture&, const State&);
}
