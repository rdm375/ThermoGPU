#include "thermogpu/peng_robinson.hpp"
#include "thermomodel/peng_robinson.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

double scaled_error(double a, double b) {
    return std::abs(a - b) /
           std::max({1.0, std::abs(a), std::abs(b)});
}

}

int main() {
    try {
        constexpr double Tc = 190.56;
        constexpr double pc = 4599200.0;
        constexpr double omega = 0.01142;
        constexpr double molar_mass = 0.016043;

        const thermogpu::Component methane{
            "methane", Tc, pc, omega, molar_mass
        };

        thermomodel::PRParameters parameters;
        parameters.Tc = Tc;
        parameters.pc = pc;
        parameters.acentric = omega;
        parameters.molar_mass = molar_mass;
        parameters.root_policy = thermomodel::PRRootPolicy::Vapor;

        thermomodel::PengRobinson model(
            std::make_shared<thermomodel::ConstantCp>(2200.0),
            parameters
        );

        std::size_t compared = 0;
        std::size_t model_rejected = 0;
        std::size_t gpu_rejected = 0;
        std::size_t both_rejected = 0;
        std::size_t violations = 0;

        double max_Z_error = 0.0;
        double max_density_error = 0.0;

        double worst_Z_T = 0.0;
        double worst_Z_p = 0.0;

        constexpr double tolerance = 1.0e-10;

        for (int i = 0; i <= 250; ++i) {
            const double T = 150.0 + i;

            for (int j = 0; j <= 200; ++j) {
                const double p = 1.0e5 + j * 1.0e5;

                bool gpu_ok = false;
                thermogpu::PureResult gpu_result{};

                try {
                    gpu_result = thermogpu::evaluate_pure(
                        methane, {p, T}
                    );
                    gpu_ok = true;
                } catch (const std::exception&) {
                    gpu_ok = false;
                }

                const auto model_result = model.evaluate_Tp(T, p);
                const bool model_ok = static_cast<bool>(model_result);

                if (!gpu_ok && !model_ok) {
                    ++both_rejected;
                    continue;
                }

                if (!gpu_ok) {
                    ++gpu_rejected;
                    continue;
                }

                if (!model_ok) {
                    ++model_rejected;
                    continue;
                }

                ++compared;

                const double eZ = scaled_error(
                    gpu_result.Z, model_result.value.Z
                );

                const double erho = scaled_error(
                    gpu_result.density_kg_per_m3,
                    model_result.value.rho
                );

                if (eZ > max_Z_error) {
                    max_Z_error = eZ;
                    worst_Z_T = T;
                    worst_Z_p = p;
                }

                max_density_error =
                    std::max(max_density_error, erho);

                if (!std::isfinite(eZ) ||
                    !std::isfinite(erho) ||
                    eZ > tolerance ||
                    erho > tolerance) {
                    ++violations;
                }
            }
        }

        std::cout << std::setprecision(17);

        std::cout << "Compared states:       " << compared << '\n';
        std::cout << "ThermoModel rejected:  " << model_rejected << '\n';
        std::cout << "ThermoGPU rejected:    " << gpu_rejected << '\n';
        std::cout << "Both rejected:         " << both_rejected << '\n';
        std::cout << "Numerical violations:  " << violations << '\n';
        std::cout << "Maximum scaled Z error: "
                  << max_Z_error << '\n';
        std::cout << "Maximum scaled density error: "
                  << max_density_error << '\n';
        std::cout << "Worst Z state: T="
                  << worst_Z_T << " K, p="
                  << worst_Z_p << " Pa\n";

        return violations == 0 ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
