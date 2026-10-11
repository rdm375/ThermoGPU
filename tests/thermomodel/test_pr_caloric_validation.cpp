#include "thermomodel/peng_robinson.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace thermomodel;

namespace {

int checks = 0;
int failures = 0;

void compare(const std::string& name,
             double actual, double expected,
             double rtol, double atol = 1e-9)
{
    ++checks;

    const double error = std::abs(actual - expected);
    const double limit = atol + rtol * std::abs(expected);

    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        error > limit) {
        ++failures;
        std::cerr << std::setprecision(17)
                  << "FAIL " << name
                  << " actual=" << actual
                  << " expected=" << expected
                  << " error=" << error
                  << " limit=" << limit << '\n';
    }
}

State state(const PengRobinson& model, double T, double p)
{
    auto r = model.evaluate_Tp(T, p);
    if (!r)
        throw std::runtime_error(
            "evaluate_Tp failed: " + r.message);
    return r.value;
}

Derivatives derivatives(const PengRobinson& model,
                        double T, double p)
{
    auto r = model.derivatives_Tp(T, p);
    if (!r)
        throw std::runtime_error(
            "derivatives_Tp failed: " + r.message);
    return r.value;
}

void validate(const PengRobinson& model,
              double T, double p)
{
    const State z = state(model, T, p);
    const Derivatives d = derivatives(model, T, p);

    const double dt = 1e-3;
    const double dp = 100.0;

    const State tp = state(model, T + dt, p);
    const State tm = state(model, T - dt, p);
    const State pp = state(model, T, p + dp);
    const State pm = state(model, T, p - dp);

    auto dT = [&](double State::*member) {
        return (tp.*member - tm.*member) / (2 * dt);
    };

    auto dP = [&](double State::*member) {
        return (pp.*member - pm.*member) / (2 * dp);
    };

    // Analytic derivatives versus centered finite differences.
    compare("drho/dT|p", d.drho_dT_p,
            dT(&State::rho), 3e-6, 1e-9);

    compare("drho/dp|T", d.drho_dp_T,
            dP(&State::rho), 3e-6, 1e-12);

    compare("dh/dT|p", d.dh_dT_p,
            dT(&State::h), 3e-6);

    compare("dh/dp|T", d.dh_dp_T,
            dP(&State::h), 3e-6, 1e-9);

    compare("ds/dT|p", d.ds_dT_p,
            dT(&State::s), 3e-6);

    compare("ds/dp|T", d.ds_dp_T,
            dP(&State::s), 3e-6, 1e-10);

    // Caloric identities.
    compare("h = u + pv", z.h,
            z.u + p / z.rho, 2e-12);

    compare("cp = dh/dT|p", z.cp,
            dT(&State::h), 3e-6);

    compare("ds/dT|p = cp/T", d.ds_dT_p,
            z.cp / T, 2e-12);

    // Maxwell identities in mass-specific coordinates.
    const double v = 1.0 / z.rho;
    const double dv_dT_p = -d.drho_dT_p /
                           (z.rho * z.rho);

    compare("ds/dp|T = -dv/dT|p",
            d.ds_dp_T, -dv_dT_p,
            2e-10, 1e-12);

    compare("dh/dp|T = v - T dv/dT|p",
            d.dh_dp_T, v - T * dv_dT_p,
            2e-10, 1e-10);

    // Independently reconstruct cp-cv using the
    // EOS derivatives inferred from density.
    const double dv_dp_T = -d.drho_dp_T /
                           (z.rho * z.rho);

    const double pv = 1.0 / dv_dp_T;
    const double pt = -dv_dT_p / dv_dp_T;

    const double cp_minus_cv = -T * pt * pt / pv;

    compare("cp-cv identity", z.cp - z.cv,
            cp_minus_cv, 3e-10);

    // Isentropic sound speed identity.
    const double c2 = -v * v * pv * z.cp / z.cv;

    compare("sound speed squared",
            z.sound_speed * z.sound_speed,
            c2, 3e-10);

    // T,p -> T,rho round trip.
    auto roundtrip = model.evaluate_Trho(T, z.rho);

    if (!roundtrip) {
        ++failures;
        std::cerr << "FAIL Trho round trip: "
                  << roundtrip.message << '\n';
    } else {
        compare("Trho pressure", roundtrip.value.p,
                p, 2e-10, 1e-4);
        compare("Trho enthalpy", roundtrip.value.h,
                z.h, 2e-10);
        compare("Trho entropy", roundtrip.value.s,
                z.s, 2e-10);
    }
}

} // namespace

int main()
{
    try {
        PRParameters params;
        params.Tc = 190.56;
        params.pc = 4599200.0;
        params.acentric = 0.01142;
        params.molar_mass = 0.016043;

        PengRobinson model(
            std::make_shared<ConstantCp>(2200.0),
            params
        );

        for (double T : {220.0, 300.0, 400.0, 600.0}) {
            for (double p : {1e5, 1e6, 5e6, 10e6}) {
                validate(model, T, p);
            }
        }

        std::cout << "PR caloric validation: "
                  << checks << " comparisons, "
                  << failures << " failures\n";

        return failures ? 1 : 0;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
