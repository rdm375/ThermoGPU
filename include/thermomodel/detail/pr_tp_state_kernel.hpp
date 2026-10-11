#pragma once

#include "thermomodel/detail/pr_tp_root_kernel.hpp"
#include "thermomodel/detail/pr_polynomial_state_kernel.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_PR_TP_HD __host__ __device__
#else
#define THERMOMODEL_PR_TP_HD
#endif

namespace thermomodel::detail {

struct PRTPResult {
    PRStateKernelResult state{};
    PRTPStatus status = PRTPStatus::InvalidInput;

    double selected_Z = 0.0;
    double selected_v = 0.0;

    int admissible_roots = 0;
};

THERMOMODEL_PR_TP_HD inline PRTPResult
pr_evaluate_Tp_polynomial(
    const PRPolynomialStateParameters& parameters,
    double T,
    double p,
    PRTPRootPolicy policy = PRTPRootPolicy::Vapor)
{
    PRTPResult result{};

    const auto selected = pr_select_Tp_root(parameters.pr.coefficients,T,p,policy);
    result.status = selected.status;
    result.admissible_roots = selected.admissible_roots;
    result.selected_Z = selected.selected_Z;
    result.selected_v = selected.selected_v;
    if (selected.status != PRTPStatus::Success) return result;

    result.state = pr_evaluate_Tvp_polynomial(
        parameters, T, selected.selected_v, p
    );


    if (result.state.status == PRStateStatus::UnstableRoot) {
        result.status = PRTPStatus::UnstableRoot;
        return result;
    }

    if (result.state.status != PRStateStatus::Success) {
        result.status = PRTPStatus::NonphysicalProperties;
        return result;
    }

    result.status = PRTPStatus::Success;
    return result;
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_TP_HD
