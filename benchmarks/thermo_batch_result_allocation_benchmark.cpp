// SPDX-License-Identifier: GPL-3.0-or-later
// ThermoGPU G4.6c.1: BatchResult allocation and initialization.

#include "thermogpu/thermo_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using thermogpu::thermo::BatchResult;
using Clock = std::chrono::steady_clock;

volatile double sink = 0.0;

double median(std::vector<double> samples)
{
    std::sort(samples.begin(), samples.end());

    const auto n = samples.size();

    return n % 2
        ? samples[n / 2]
        : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

void initialize(BatchResult& result, std::size_t n)
{
    const double nan =
        std::numeric_limits<double>::quiet_NaN();

    result.Z.assign(n, nan);
    result.rho.assign(n, nan);
    result.h.assign(n, nan);
    result.s.assign(n, nan);
    result.u.assign(n, nan);
    result.cp.assign(n, nan);
    result.cv.assign(n, nan);
    result.sound_speed.assign(n, nan);

    result.phase.assign(n, thermomodel::Phase::Unknown);
    result.error.assign(n, thermomodel::ErrorCode::None);

    result.message.resize(n);

    for (auto& message : result.message)
        message.clear();
}

template<class F>
double measure(F&& operation, int repeats)
{
    std::vector<double> samples;
    samples.reserve(repeats);

    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();

        operation();

        const auto stop = Clock::now();

        samples.push_back(
            std::chrono::duration<double, std::milli>(
                stop - start).count()
        );
    }

    return median(std::move(samples));
}

void run(std::size_t n, int repeats)
{
    // A: Fresh allocation and initialization.
    const double fresh = measure([&] {
        BatchResult result;
        initialize(result, n);
        sink = sink + static_cast<double>(result.size());
    }, repeats);

    // B: Reuse existing allocations but reinitialize contents.
    BatchResult reusable;
    initialize(reusable, n);

    const double reused = measure([&] {
        initialize(reusable, n);
        sink = sink + static_cast<double>(reusable.size());
    }, repeats);

    // C: Reuse existing storage without reinitialization.
    const double retained = measure([&] {
        sink = sink + static_cast<double>(reusable.size());
    }, repeats);

    std::cout
        << std::setw(10) << n
        << std::setw(16) << fresh
        << std::setw(16) << reused
        << std::setw(16) << retained
        << std::setw(16) << fresh / reused
        << '\n';
}

} // namespace

int main()
{
    constexpr int repeats = 15;

    std::cout << std::fixed << std::setprecision(6);

    std::cout
        << std::setw(10) << "N"
        << std::setw(16) << "fresh_ms"
        << std::setw(16) << "reused_ms"
        << std::setw(16) << "retained_ms"
        << std::setw(16) << "fresh/reused"
        << '\n';

    for (std::size_t n :
         {10000ULL, 100000ULL, 1000000ULL})
    {
        run(n, repeats);
    }

    std::cout << "sink=" << sink << '\n';
}
