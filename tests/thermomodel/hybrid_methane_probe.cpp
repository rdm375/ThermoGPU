#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/methane_caloric.hpp"
#include "thermomodel/derivatives.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: hybrid_methane_probe T p\n";
        return 2;
    }

    const double T = std::strtod(argv[1], nullptr);
    const double p = std::strtod(argv[2], nullptr);

    thermomodel::PengRobinson model(
        thermomodel::make_methane_nasa_low_cp()
    );

    auto state = model.evaluate_Tp(T, p);
    auto deriv = model.derivatives_Tp(T, p);

    if (!state || !deriv) {
        std::cerr << "ThermoModel evaluation failed\n";
        return 1;
    }

    // Derivatives aggregate ordering matches src/peng_robinson.cpp:
    // drho/dp|T, drho/dT|p, dh/dp|T, dh/dT|p,
    // ds/dp|T, ds/dT|p.
    const auto& [
        drho_dp, drho_dT, dh_dp, dh_dT, ds_dp, ds_dT
    ] = deriv.value;

    std::cout << std::setprecision(17)
              << state.value.cp << ","
              << state.value.cv << ","
              << state.value.sound_speed << ","
              << drho_dp << ","
              << drho_dT << ","
              << dh_dp << ","
              << ds_dp << "\n";
    return 0;
}
