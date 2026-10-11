#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_CUBIC_HD __host__ __device__
#else
#define THERMOMODEL_CUBIC_HD
#endif

namespace thermomodel::detail {

// Explicitly rounded arithmetic for cancellation-sensitive expressions.
// CUDA device intrinsics prevent multiplication/addition contraction.
// CPU operations retain the reference implementation's arithmetic order.

THERMOMODEL_CUBIC_HD inline double cubic_mul_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}

THERMOMODEL_CUBIC_HD inline double cubic_add_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}

THERMOMODEL_CUBIC_HD inline double cubic_sub_rn(double a, double b)
{
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, -b);
#else
    return a - b;
#endif
}

struct CubicRoots {
    double values[3]{};
    int count = 0;
};

THERMOMODEL_CUBIC_HD inline CubicRoots
solve_monic_cubic(double c2, double c1, double c0)
{
    constexpr double pi = 3.14159265358979323846;

    // Preserve separate rounding in the depressed-cubic coefficients.
    const double q = cubic_sub_rn(
        cubic_mul_rn(3.0, c1),
        cubic_mul_rn(c2, c2)
    ) / 9.0;

    const double r_numerator = cubic_sub_rn(
        cubic_sub_rn(
            cubic_mul_rn(
                cubic_mul_rn(9.0, c2),
                c1
            ),
            cubic_mul_rn(27.0, c0)
        ),
        cubic_mul_rn(
            cubic_mul_rn(
                cubic_mul_rn(2.0, c2),
                c2
            ),
            c2
        )
    );

    const double r = r_numerator / 54.0;

    const double disc = cubic_add_rn(
        cubic_mul_rn(cubic_mul_rn(q, q), q),
        cubic_mul_rn(r, r)
    );

    CubicRoots roots{};

    if (disc > 0) {
        const double sd = ::sqrt(disc);
        const double u = ::cbrt(r+sd);
        const double v = ::cbrt(r-sd);

        roots.values[roots.count++] = -c2/3+u+v;
    } else if (q < 0) {
        double arg = r/::sqrt(-q*q*q);

        if (arg < -1.0) arg = -1.0;
        if (arg > 1.0) arg = 1.0;

        const double angle = ::acos(arg)/3;

        for (int k = 0; k < 3; ++k) {
            roots.values[roots.count++] =
                2*::sqrt(-q)*::cos(angle-2*pi*k/3)-c2/3;
        }
    } else {
        roots.values[roots.count++] = -c2/3;
    }

    // Sort at most three roots.
    for (int i = 1; i < roots.count; ++i) {
        const double value = roots.values[i];
        int j = i;

        while (j > 0 && roots.values[j-1] > value) {
            roots.values[j] = roots.values[j-1];
            --j;
        }

        roots.values[j] = value;
    }

    // Preserve the original relative deduplication criterion.
    int unique_count = 0;

    for (int i = 0; i < roots.count; ++i) {
        if (unique_count == 0) {
            roots.values[unique_count++] = roots.values[i];
            continue;
        }

        const double previous = roots.values[unique_count-1];
        const double current = roots.values[i];

        double scale = ::fmax(1.0, ::fabs(previous));
        scale = ::fmax(scale, ::fabs(current));

        if (::fabs(current-previous) >= 1e-11*scale)
            roots.values[unique_count++] = current;
    }

    roots.count = unique_count;
    return roots;
}

} // namespace thermomodel::detail

#undef THERMOMODEL_CUBIC_HD
