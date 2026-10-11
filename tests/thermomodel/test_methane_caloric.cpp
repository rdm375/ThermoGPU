#include "thermomodel/methane_caloric.hpp"
#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/flash.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace thermomodel;

namespace {

constexpr double R = 8.31446261815324 / 0.016043;

constexpr std::array<double,5> A = {
    5.14987613,
    -1.36709788e-2,
    4.91800599e-5,
    -4.84743026e-8,
    1.66693956e-11
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void near(double actual, double expected,
          double rtol, double atol, const char* message) {
    if (!std::isfinite(actual) ||
        std::abs(actual-expected) >
            atol + rtol*std::abs(expected)) {
        std::cerr << message << ": actual=" << actual
                  << " expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}

double reference_cp(double T) {
    double result = 0.0;
    double power = 1.0;

    for (double coefficient : A) {
        result += coefficient*power;
        power *= T;
    }

    return R*result;
}

double reference_dh(double T, double ref) {
    double result = 0.0;
    double tpower = T;
    double rpower = ref;

    for (std::size_t k=0; k<A.size(); ++k) {
        result += A[k]*(tpower-rpower)/(k+1);
        tpower *= T;
        rpower *= ref;
    }

    return R*result;
}

double reference_ds(double T, double ref) {
    double result = A[0]*std::log(T/ref);
    double tpower = T;
    double rpower = ref;

    for (std::size_t k=1; k<A.size(); ++k) {
        result += A[k]*(tpower-rpower)/k;
        tpower *= T;
        rpower *= ref;
    }

    return R*result;
}

}

int main() {
    try {
        auto caloric = make_methane_nasa_low_cp();

        require(caloric->minimum_temperature()==200.0,
                "minimum caloric temperature");
        require(caloric->maximum_temperature()==1000.0,
                "maximum caloric temperature");

        constexpr double Tref = 298.15;

        for (double T : {
            200.0, 220.0, 300.0, 400.0,
            600.0, 800.0, 1000.0
        }) {
            near(caloric->cp(T),
                 reference_cp(T),
                 2e-13, 1e-9, "ideal cp");

            near(caloric->enthalpy_delta(T,Tref),
                 reference_dh(T,Tref),
                 2e-12, 1e-7, "ideal dh");

            near(caloric->entropy_thermal_delta(T,Tref),
                 reference_ds(T,Tref),
                 2e-12, 1e-9, "ideal ds");
        }

        PengRobinson model(caloric);

        auto metadata = model.metadata();

        require(metadata.Tmin==200.0,
                "PR lower temperature bound");
        require(metadata.Tmax==1000.0,
                "PR upper temperature bound");

        for (double T : {199.999,1000.001}) {
            require(!model.evaluate_Tp(T,5e6),
                    "Tp must reject invalid temperature");

            require(!model.evaluate_Trho(T,35.0),
                    "Trho must reject invalid temperature");

            require(!model.derivatives_Tp(T,5e6),
                    "derivatives must reject invalid temperature");
        }

        constexpr double pressure = 5e6;

        auto state = model.evaluate_Tp(300.0,pressure);
        require(bool(state), "reference state evaluation");

        auto ph = flash_ph(
            model,pressure,state.value.h
        );
        require(bool(ph), "ph flash");
        near(ph.value.T,300.0,1e-10,1e-7,
             "ph flash temperature");

        auto ps = flash_ps(
            model,pressure,state.value.s
        );
        require(bool(ps), "ps flash");
        near(ps.value.T,300.0,1e-10,1e-7,
             "ps flash temperature");

        std::cout
            << "PASS methane NASA caloric validation\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
