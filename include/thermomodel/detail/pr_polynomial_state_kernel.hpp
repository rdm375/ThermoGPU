#pragma once

#include "thermomodel/detail/polynomial_cp_kernel.hpp"
#include "thermomodel/detail/pr_state_kernel.hpp"

#if defined(__CUDACC__)
#define THERMOMODEL_PR_POLYNOMIAL_HD __host__ __device__
#else
#define THERMOMODEL_PR_POLYNOMIAL_HD
#endif

namespace thermomodel::detail {

struct PRPolynomialStateParameters {
    PRStateKernelParameters pr;
    PolynomialCpKernel caloric;
    double Tref;
};

THERMOMODEL_PR_POLYNOMIAL_HD inline PRStateKernelResult
pr_evaluate_Tvp_polynomial(
    const PRPolynomialStateParameters& parameters,
    double T,
    double v,
    double p)
{
    const PRIdealCaloricState ideal{
        polynomial_cp(parameters.caloric, T),

        polynomial_enthalpy_delta(
            parameters.caloric,
            T,
            parameters.Tref
        ),

        polynomial_entropy_delta(
            parameters.caloric,
            T,
            parameters.Tref
        )
    };

    return pr_evaluate_Tvp(
        parameters.pr,
        ideal,
        T,
        v,
        p
    );
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_POLYNOMIAL_HD
