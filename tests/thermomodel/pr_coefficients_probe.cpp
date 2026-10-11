#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <cstdio>
#include <initializer_list>

int main()
{
    using namespace thermomodel::detail;

    const auto q = make_pr_coefficients(
        0.016043,
        190.56,
        4599200.0,
        0.01142
    );

    std::printf("R     = %.17g\n", q.R);
    std::printf("a0    = %.17g\n", q.a0);
    std::printf("b     = %.17g\n", q.b);
    std::printf("kappa = %.17g\n", q.kappa);

    for (double T : {200.0, 250.0, 300.0, 400.0, 600.0, 1000.0}) {
        const auto a = pr_attraction(q, T);

        std::printf(
            "T=%.17g a=%.17g ap=%.17g app=%.17g\n",
            T, a.a, a.ap, a.app
        );
    }

    return 0;
}
