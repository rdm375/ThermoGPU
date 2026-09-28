# Benchmarking

`thermogpu_benchmark` characterizes the M4 scalar and OpenMP batch backends using the same deterministic CH4/C2H6 workload. It is a characterization tool, not a correctness test; correctness remains gated by CTest differential tests.

The default sweep uses batch sizes `1, 10, 100, 1000, 10000, 100000, 1000000`. For each size the executable measures the scalar batch backend followed by OpenMP at 1, 2, 4, and 8 threads when those thread counts are available. Each configuration is warmed up and then calibrated so that a timed sample targets approximately 200 ms. The calibrated repetition count preserves repeated batch-call overhead while making even small batches long enough for stable timing. Nine samples are collected; the median is the primary timing, with minimum, maximum, and median absolute deviation (MAD) reported as dispersion diagnostics. Validation and checksum work are outside the timed region.

Reported `Speedup` is relative to the scalar batch backend at the same batch size and uses median time per batch call. `MAD%` is the median absolute deviation divided by the median, expressed as a percentage; it provides a robust indication of run-to-run timing variability. Keeping scalar and OpenMP-1 as separate rows exposes OpenMP parallel-region overhead. On an SMT CPU, comparing the physical-core count with the logical-CPU count also shows whether simultaneous multithreading adds useful throughput.

Build and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTHERMOGPU_ENABLE_OPENMP=ON
cmake --build build -j"$(nproc)"
./build/thermogpu_benchmark
```

Specific batch sizes can be supplied on the command line, for example:

```bash
./build/thermogpu_benchmark 1000 10000 100000
```

These M4 CPU timings are end-to-end calls to the current batch API, including result allocation and the per-state work performed by each backend. M5 CUDA measurements separately report resident and end-to-end timings so host/device transfer, allocation, and launch costs are not hidden. The workload-dependent scalar/OpenMP/CUDA crossover is reported below.

## Focused scalar/OpenMP crossover sweep

The default decade sweep is intentionally broad. To resolve the small-batch crossover more closely, use:

```bash
./build/thermogpu_benchmark --crossover
```

This measures batch sizes `10, 15, 20, 25, 30, 40, 50, 60, 75, 100`. The crossover should be identified from the measured speedups, not assumed from the grid: it lies between the largest tested batch size where the relevant OpenMP backend has `Speedup < 1` and the smallest tested size where it has `Speedup > 1`. If a row has `MAD% > 10`, the benchmark marks it `UNSTABLE`; repeat that point before using it to bracket a crossover or support a performance conclusion.

The crossover is backend- and machine-specific. In particular, OpenMP-1 is retained as an overhead diagnostic rather than an expected performance winner. For the Dell Precision 7710 / i7-6920HQ characterization, the broad M4 sweep showed that four physical cores provide nearly all of the useful CPU parallelism at large batch sizes, while eight SMT threads add only a few percent; the focused sweep is intended to locate where the four-core backend first amortizes its parallel-region overhead relative to scalar execution.

## Linear performance model

The focused `--crossover` run also fits a simple total-call-time model to the stable median measurements:

```text
T(N) = a + b N
```

where `a` estimates fixed per-call overhead and `b` estimates marginal time per state over the measured crossover range. Rows marked `UNSTABLE` (`MAD% > 10`) are excluded from the fit. The report includes the number of fitted points, `a` in microseconds, `b` in ns/state, coefficient of determination (`R^2`), RMS residual, and the intersection of each OpenMP fitted line with the scalar fitted line.

The fitted intersection is a compact estimate of crossover, not a replacement for the raw measurements. The measured bracket remains authoritative if the linear model has poor fit quality or if the fitted intersection lies outside the sampled range. This distinction is important because batch composition, allocation costs, cache effects, OpenMP scheduling, and CPU frequency behavior can make real execution depart from a two-parameter linear model.

## M5 CUDA baseline timing

When built with `THERMOGPU_ENABLE_CUDA=ON`, the benchmark adds two CUDA rows at every requested batch size. `CUDA-res` constructs and uploads a fixed batch before timing, then measures repeated synchronized kernel executions with the batch and outputs resident on the device; its final download/checksum is outside the timed region. `CUDA-e2e` times the ordinary public `evaluate_mixture_batch_cuda()` call, including device allocation, invariant/state H2D copies, kernel launch/execution, synchronization, and D2H result copies. This distinction prevents PCIe/allocation overhead from being hidden while also exposing the throughput of the baseline kernel itself.

The resident path uses the same `CudaBatchWorkspace` and PR kernel as the ordinary CUDA API; it is not a separate benchmark-only implementation. As with CPU timing, validation/checksum work is excluded. CUDA rows use the scalar median at the same batch size for the reported speedup. The default decade sweep is appropriate for first characterization; after observing the CPU/CUDA crossover, run explicit intermediate sizes to bracket it more closely.


## Focused CPU/CUDA crossover sweep

After the broad M5 decade sweep, use:

```bash
./build/thermogpu_benchmark --gpu-crossover
```

This measures `100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000` states. The grid spans both transitions observed in the broad characterization: resident CUDA versus the CPU envelope and end-to-end CUDA versus the CPU envelope.

The ordinary benchmark rows retain `Speedup` relative to scalar for continuity with earlier M4/M5 results. After the detailed rows, `--gpu-crossover` prints a dedicated summary in which the CPU reference is the fastest stable measured CPU backend at each batch size (`scalar`, `OMP2`, `OMP4`, or `OMP8`). OpenMP rows with `MAD% > 10` are excluded from this CPU envelope so an unstable measurement cannot create a false crossover. The scalar row remains the fallback reference.

In the summary, `res/CPU` and `e2e/CPU` are CPU-envelope time divided by the corresponding CUDA time, so values greater than one mean CUDA is faster. Crossover is bracketed directly from adjacent measured points around one. No affine GPU model is fitted at this stage because the broad M5 sweep shows distinct latency-, utilization-, and throughput-dominated CUDA regimes.

## M5 measured CUDA characterization — Dell Precision 7710 / Quadro M3000M

The M5 characterization used a Dell Precision 7710 with an Intel i7-6920HQ (4 physical cores / 8 hardware threads) and NVIDIA Quadro M3000M. The benchmark uses the deterministic CH4/C2H6 workload described above; these crossover values are machine- and workload-specific and are not hard-coded dispatch thresholds.

The focused heterogeneous sweep measured resident CUDA below the fastest stable CPU backend at 250 states (`res/CPU = 0.916`) and above it at 300 states (`res/CPU = 1.110`). Therefore the authoritative measured resident crossover bracket is **250–300 states**. Linear interpolation may be used descriptively to say "about 275 states", but the measured bracket is the result.

A second narrow sweep (`--gpu-e2e-crossover`) resolved the end-to-end transition with 200-state spacing from 3200 through 4800 states. CUDA remained slower than the fastest stable CPU backend at 3800 states (`CPU = 104.1 ns/state`, `CUDA-e2e = 108.8 ns/state`, `e2e/CPU = 0.957`) and was faster at 4000 states (`CPU = 106.8 ns/state`, `CUDA-e2e = 104.6 ns/state`, `e2e/CPU = 1.021`). Every sampled point from 4000 through 4800 remained above unity. The authoritative measured end-to-end crossover bracket is therefore **3800–4000 states**. Linear interpolation of the timing difference between those adjacent measurements gives approximately **3936 states**, reported descriptively as **about 3900 states**; the measured bracket remains authoritative.

At 10,000 states, resident CUDA measured 16.5 ns/state versus 103.2 ns/state for the fastest stable CPU row, a **6.261x** speedup over the CPU envelope. End-to-end CUDA measured 66.6 ns/state, a **1.548x** speedup over that envelope. The resident timings also show a roughly 34 microsecond execution floor from approximately 100 through 1000 states before transitioning toward a throughput-dominated regime. For this reason M5 does not force a single affine model across the CUDA measurements; the raw measured crossover brackets remain authoritative.

The distinction between `CUDA-res` and `CUDA-e2e` is architectural, not cosmetic. A larger GPU-resident simulation that repeatedly invokes EOS work without returning all state to the host can approach the resident regime; an isolated one-shot EOS call must pay the end-to-end allocation and transfer costs measured by `CUDA-e2e`.

### CUDA 12.4 host-toolchain configuration

On the characterized system, `/usr/bin/c++` selected GCC 15 while CUDA 12.4 selected GCC 13 as nvcc's supported host compiler. Mixing GCC-15-compiled C++/OpenMP objects with the GCC 13 library search path contributed by CUDA caused the final link to select the GCC 13 development `libstdc++`, which lacks the `__cxa_call_terminate@@CXXABI_1.3.15` symbol required by the GCC 15 object. CUDA 12.4 also explicitly rejects GCC versions later than 13 unless nvcc is forced with `--allow-unsupported-compiler`; ThermoGPU does not use that unsupported override.

For the reproducible CUDA 12.4 configuration used for M5, configure both the project C++ compiler and CUDA host compiler as GCC 13:

```bash
rm -rf build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DTHERMOGPU_ENABLE_OPENMP=ON \
    -DTHERMOGPU_ENABLE_CUDA=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

This is a documented toolchain requirement for the tested CUDA 12.4 environment, not a claim that ThermoGPU intrinsically requires GCC 13 with all CUDA releases. Compiler support should be revisited when the CUDA toolkit changes.

The narrow end-to-end crossover can be reproduced with:

```bash
./build/thermogpu_benchmark --gpu-e2e-crossover \
    | tee benchmark-M5-gpu-e2e-crossover-M3000M.txt
```

## M5 acceptance summary

For the characterized Dell Precision 7710 / i7-6920HQ / Quadro M3000M system:

- resident CUDA crosses the measured CPU envelope between **250 and 300 states** (about **275** by local interpolation);
- end-to-end CUDA crosses between **3800 and 4000 states** (about **3900**, or **3936** by local interpolation);
- at 10,000 states, resident CUDA is **6.261x** faster and end-to-end CUDA is **1.548x** faster than the fastest stable measured CPU backend;
- CUDA timing exhibits distinct latency/utilization/throughput regimes, so M5 intentionally does not fit one affine model across the complete GPU range; and
- these are machine- and workload-specific characterization results, not universal runtime dispatch thresholds.

Final acceptance should use the homogeneous CUDA 12.4 / GCC 13 configuration above, run the complete CTest suite, then capture one final default broad benchmark. The committed benchmark logs are experimental evidence; correctness remains gated by the differential tests.
