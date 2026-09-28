# Benchmarking

`thermogpu_benchmark` characterizes the M4 scalar and OpenMP batch backends using the same deterministic CH4/C2H6 workload. It is a characterization tool, not a correctness test; correctness remains gated by CTest differential tests.

The default sweep uses batch sizes `1, 10, 100, 1000, 10000, 100000, 1000000`. For each size the executable measures the scalar batch backend followed by OpenMP at 1, 2, 4, and 8 threads when those thread counts are available. Each configuration is warmed up, then timed for five samples; the reported value is the median. Small batches are repeated within each sample to reduce timer noise. Validation and checksum work are outside the timed region.

Reported `Speedup` is relative to the scalar batch backend at the same batch size. Keeping scalar and OpenMP-1 as separate rows exposes OpenMP parallel-region overhead. On an SMT CPU, comparing the physical-core count with the logical-CPU count also shows whether simultaneous multithreading adds useful throughput.

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
