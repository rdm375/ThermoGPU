#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_HD __host__ __device__
#else
#define THERMOMODEL_HD
#endif

namespace thermomodel::detail {

struct PolynomialCpKernel {
    double a[5];
    double scale;
};

THERMOMODEL_HD inline double
polynomial_cp(const PolynomialCpKernel& q, double T)
{
    const double t = T / q.scale;

    return ((((q.a[4] * t + q.a[3]) * t
              + q.a[2]) * t + q.a[1]) * t + q.a[0]);
}

THERMOMODEL_HD inline double
polynomial_enthalpy_delta(
    const PolynomialCpKernel& q,
    double T,
    double ref)
{
    const double t = T / q.scale;
    const double r = ref / q.scale;

    double tpow = t;
    double rpow = r;
    double sum = 0.0;

    for (int k = 0; k < 5; ++k) {
        sum += q.a[k] * (tpow - rpow) / (k + 1);
        #if defined(__CUDA_ARCH__)
        tpow = __dmul_rn(tpow, t);
#else
        tpow *= t;
#endif
        #if defined(__CUDA_ARCH__)
        rpow = __dmul_rn(rpow, r);
#else
        rpow *= r;
#endif
    }

    return q.scale * sum;
}

THERMOMODEL_HD inline double
polynomial_entropy_delta(
    const PolynomialCpKernel& q,
    double T,
    double ref)
{
    const double t = T / q.scale;
    const double r = ref / q.scale;

    double sum = q.a[0] * ::log(T / ref);
    double tpow = t;
    double rpow = r;

    for (int k = 1; k < 5; ++k) {
        #if defined(__CUDA_ARCH__)
        const double term =
            q.a[k] * (tpow - rpow) / k;
        sum = __dadd_rn(sum, term);
#else
        sum += q.a[k] * (tpow - rpow) / k;
#endif
        tpow *= t;
        rpow *= r;
    }

    return sum;
}

} // namespace thermomodel::detail

#undef THERMOMODEL_HD
