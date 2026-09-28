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

These M4 CPU timings are end-to-end calls to the current batch API, including result allocation and the per-state work performed by each backend. Later CUDA benchmarks will separately report kernel-only and end-to-end timings so host/device transfer and launch costs are not hidden. The principal V1 result will be the workload-dependent scalar/OpenMP/CUDA crossover.

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
