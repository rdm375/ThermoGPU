// SPDX-License-Identifier: GPL-3.0-or-later
// ThermoGPU G4.6c: CUDA workspace execution-time decomposition.

#include "thermogpu/thermo_batch_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace thermogpu::thermo;
using Clock = std::chrono::steady_clock;

// Reused verbatim from the G4.6a backend benchmark.
BatchInput make_input(std::size_t n, const std::string& workload) {
    BatchInput input;
    input.temperature_K.resize(n);
    input.pressure_Pa.resize(n);
    for (std::size_t i=0; i<n; ++i) {
        const double x = n > 1 ? static_cast<double>(i)/static_cast<double>(n-1) : 0.5;
        if (workload=="uniform") {
            input.temperature_K[i]=320.0;
            input.pressure_Pa[i]=8.0e6;
        } else if (workload=="smooth") {
            input.temperature_K[i]=300.0+35.0*x+4.0*std::sin(6.283185307179586*x);
            input.pressure_Pa[i]=4.0e6+5.0e6*x+2.0e5*std::sin(12.566370614359172*x);
        } else if (workload=="diverse") {
            // Deterministic, inexpensive index scrambling. All states inside PR domain.
            const double a = static_cast<double>((i*104729ULL)%1000003ULL)/1000003.0;
            const double b = static_cast<double>((i*130363ULL+17ULL)%1000033ULL)/1000033.0;
            input.temperature_K[i]=220.0+600.0*a;
            input.pressure_Pa[i]=1.0e5+3.0e7*b;
        } else throw std::invalid_argument("unknown workload");
    }
    return input;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());

    const auto n = values.size();

    return n % 2
        ? values[n / 2]
        : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

struct Sample {
    double total_ms{};
    double pack_ms{};
    double h2d_ms{};
    double kernel_wall_ms{};
    double kernel_event_ms{};
    double d2h_ms{};
    double conversion_ms{};
    double result_init_ms{};
    double error_translation_ms{};
    double numeric_phase_ms{};
    double checksum{};
};

Sample measure(
    CUDAThermoWorkspace& workspace,
    const BatchInput& input)
{
    const auto start = Clock::now();

    auto profile = workspace.profile(input);

    const auto stop = Clock::now();

    Sample s;

    s.total_ms =
        std::chrono::duration<double, std::milli>(
            stop - start).count();

    s.pack_ms = profile.pack_ms;
    s.h2d_ms = profile.h2d_ms;
    s.kernel_wall_ms = profile.kernel_wall_ms;
    s.kernel_event_ms = profile.kernel_event_ms;
    s.d2h_ms = profile.d2h_ms;
    s.conversion_ms = profile.conversion_ms;
    s.result_init_ms = profile.result_init_ms;
    s.error_translation_ms = profile.error_translation_ms;
    s.numeric_phase_ms = profile.numeric_phase_ms;

    const std::size_t n = input.size();

    if (profile.result.Z.size() != n ||
        profile.result.error.size() != n)
        throw std::runtime_error("Incorrect result size");

    const std::size_t middle = n / 2;

    if (profile.result.error[middle] != thermomodel::ErrorCode::None ||
        !std::isfinite(profile.result.Z[middle]))
        throw std::runtime_error("Invalid thermodynamic result");

    s.checksum = profile.result.Z[middle];

    return s;
}

double field_median(
    const std::vector<Sample>& samples,
    double Sample::*field)
{
    std::vector<double> values;
    values.reserve(samples.size());

    for (const auto& sample : samples)
        values.push_back(sample.*field);

    return median(std::move(values));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const std::string path =
            argc > 1 ? argv[1] : "g46c-profile.csv";

        if (argc > 2)
            throw std::invalid_argument(
                "usage: thermogpu_thermo_profile_benchmark [output.csv]");

        constexpr int repeats = 9;

        std::ofstream csv(path);

        if (!csv)
            throw std::runtime_error("Cannot open CSV output");

        csv << std::setprecision(12);

        csv
            << "workload,batch_size,repeat,total_ms,"
            << "pack_ms,h2d_ms,"
            << "kernel_wall_ms,kernel_event_ms,"
            << "d2h_ms,conversion_ms,"
            << "result_init_ms,error_translation_ms,"
            << "numeric_phase_ms,checksum\n";

        // Identical configuration construction to G4.6a.
        CUDAPRParameters config;

        for (const auto& workload :
             {"uniform", "smooth", "diverse"})
        {
            for (std::size_t n :
                 {10000ULL, 100000ULL, 1000000ULL})
            {
                const auto input = make_input(n, workload);

                CUDAThermoWorkspace workspace(config, n);

                // Warm up before recording measurements.
                for (int i = 0; i < 3; ++i)
                    (void)measure(workspace, input);

                std::vector<Sample> samples;
                samples.reserve(repeats);

                for (int repeat = 0; repeat < repeats; ++repeat)
                {
                    const Sample s = measure(workspace, input);

                    samples.push_back(s);

                    csv
                        << workload << ','
                        << n << ','
                        << repeat << ','
                        << s.total_ms << ','
                        << s.pack_ms << ','
                        << s.h2d_ms << ','
                        << s.kernel_wall_ms << ','
                        << s.kernel_event_ms << ','
                        << s.d2h_ms << ','
                        << s.conversion_ms << ','
                        << s.result_init_ms << ','
                        << s.error_translation_ms << ','
                        << s.numeric_phase_ms << ','
                        << s.checksum << '\n';
                }

                std::cout
                    << std::fixed << std::setprecision(6)
                    << workload
                    << " N=" << n
                    << " total="
                    << field_median(samples, &Sample::total_ms)
                    << " kernel="
                    << field_median(samples, &Sample::kernel_event_ms)
                    << " H2D="
                    << field_median(samples, &Sample::h2d_ms)
                    << " D2H="
                    << field_median(samples, &Sample::d2h_ms)
                    << " conversion="
                    << field_median(samples, &Sample::conversion_ms)
                    << " ms\n";
            }
        }

        std::cout << "CSV: " << path << '\n';

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
