# M7 — Heterogeneous performance analysis

M7 converts the M4/M5 timing work into a reproducible heterogeneous performance study and adds the V1.0 axis that was still missing: **component-count scaling**. The numerical problem is unchanged: all backends execute the same V1 Peng–Robinson calculation, and M6 remains the numerical-equivalence gate.

## Questions

1. How do scalar CPU, OpenMP, resident CUDA, and end-to-end CUDA throughput scale with batch size?
2. How does CPU/GPU crossover move as mixture component count changes?
3. How large is host/device-transfer and allocation overhead relative to a resident workload?
4. What measured CPU/GPU effects explain the component-count behavior, and which effects cannot be resolved on the available profiling hardware?

## Benchmark methodology

`thermogpu_scaling` benchmarks 1-, 2-, 3-, and 5-component mixtures using component data already shipped in `data/components.csv`. Binary interaction parameters are zero so the experiment isolates computational scaling rather than a change of physical model. The production batch grid is:

```text
100 300 1000 3000 10000 30000 100000 300000 1000000
```

Each production timing uses warm-up, calibrated repeated calls targeting about 200 ms per sample, nine samples, the median as the primary timing, and MAD as a dispersion diagnostic. CUDA is reported both as a resident workload and end-to-end through the public API. `--quick` is only a smoke test and is not performance evidence.

The retained CSV is `results/m7/precision-7710/scaling.csv`. `tools/analyze_m7.py` regenerates the throughput figures and crossover summary. Crossover brackets are measured brackets around speedup 1. The optional local interpolation is descriptive interpolation of the adjacent measured GPU-minus-best-CPU timing difference; it is not a fitted performance model.

## Dell Precision 7710 characterization

The primary M7 characterization was performed on a Dell Precision 7710 with Intel i7-6920HQ (4 physical cores / 8 hardware threads) and NVIDIA Quadro M3000M (GM204, 4 GiB). CUDA code was compiled for `sm_52`. These results are machine- and workload-specific and are not universal dispatch thresholds.

### Broad component scaling

The broad production sweep measured the following crossover brackets and maximum measured GPU speedups over the best stable CPU backend:

| Components | CUDA mode | Broad measured crossover | Max measured speedup vs best CPU |
|---:|---|---:|---:|
| 1 | resident | 100–300 | 11.195x |
| 1 | end-to-end | 3000–10000 | 2.050x |
| 2 | resident | 100–300 | 9.003x |
| 2 | end-to-end | 3000–10000 | 2.235x |
| 3 | resident | 300–1000 | 7.270x |
| 3 | end-to-end | 3000–10000 | 2.106x |
| 5 | resident | 300–1000 | 5.926x |
| 5 | end-to-end | 3000–10000 | 1.983x |

At 1,000,000 states in the five-component workload, scalar measured about 533.6 ns/state, OpenMP-8 about 173.8 ns/state, resident CUDA about 29.33 ns/state, and end-to-end CUDA about 96.83 ns/state. Resident CUDA therefore measured about 5.93x faster than the best CPU row, while end-to-end CUDA measured about 1.79x faster.

The broad maximum resident speedup decreases with component count (11.19x at one component to 5.93x at five components). That observation motivated the profiling work below; it is not by itself an explanation.

### Focused crossover sweeps

`thermogpu_scaling --crossover` resolves the transitions on a denser grid. The authoritative focused results are:

| Components | CUDA mode | Measured bracket | Local interpolation |
|---:|---|---:|---:|
| 1 | resident | 175–200 | ~191 states |
| 1 | end-to-end | 5000–5500 | ~5004 states |
| 2 | resident | 150–175 | ~171 states |
| 2 | end-to-end | 4000–4500 | ~4414 states |
| 3 | resident | 300–350 | ~328 states |
| 3 | end-to-end | 3500–4000 | ~3989 states |
| 5 | resident | 300–350 | ~339 states |
| 5 | end-to-end | 3000–3500 | ~3402 states |

The end-to-end crossover moves downward as component count increases, consistent with more compute per state amortizing fixed allocation/transfer costs. The resident crossover is non-monotonic (about 191, 171, 328, and 339 states for 1/2/3/5 components); M7 does not invent a causal explanation for that pattern.

## GPU profiling

### Controlled profiling mode

Profiling uses a separate fixed-work path rather than the calibrated benchmark loop:

```bash
./build/thermogpu_scaling --profile cuda 1 100000
./build/thermogpu_scaling --profile cuda 5 100000
```

The CUDA profile path constructs one deterministic workload, creates one reusable `CudaBatchWorkspace`, performs one warm-up, executes ten resident launches, downloads once, and emits a checksum. The warm-up plus ten measured launches therefore appears as eleven kernel instances in a Systems trace.

### Nsight Systems: one versus five components

For 100,000 states, Nsight Systems measured:

| Metric | 1 component | 5 components | 5c / 1c |
|---|---:|---:|---:|
| Median PR kernel | 1.108 ms | 3.766 ms | 3.40x |
| Mean PR kernel | 1.106 ms | 3.760 ms | 3.40x |
| Kernel instances | 11 | 11 | — |
| H2D volume | 2.4 MB | 5.6 MB | 2.33x |
| D2H volume | 4.1 MB | 7.3 MB | 1.78x |

The kernel ranges were tight: approximately 1.091–1.113 ms for one component and 3.729–3.798 ms for five components. The 3.40x kernel-time increase is therefore stable and is not an artifact of the broad benchmark's mixed workload.

Nsight Systems 2023.4 on this installation successfully collected `.qdstrm` traces, but its target-side CLI could not automatically locate the separately installed host `QdstrmImporter`. Manual import with `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter` produced valid `.nsys-rep` reports and CUDA summaries. This is an environment/tool packaging issue, not a CUDA tracing failure.

### Static CUDA resources (`ptxas -v`)

Configure with `-DTHERMOGPU_PTXAS_VERBOSE=ON` to add `-Xptxas=-v` only to CUDA compilation. The production `pr_batch_kernel` compiled for `sm_52` with:

```text
58 registers/thread
832-byte stack frame/thread
0-byte spill stores
0-byte spill loads
```

The diagnostic kernel is intentionally heavier (84 registers/thread and a 10,200-byte stack frame/thread) and is not used for production performance conclusions.

Because all component counts execute the same compiled production kernel, static register allocation does not change between the 1- and 5-component experiments. Zero spill loads/stores also excludes a register-spilling transition as the explanation for the component-count slowdown.

### Nsight Compute limitation

Nsight Compute 2024.1 initially reported `ERR_NVGPUCTRPERM` for an unprivileged run. Repeating the support test under administrative privileges passed that restriction but reported that profiling is not supported on device 0. The Quadro M3000M / GM204 therefore cannot provide the requested hardware-counter metrics with the installed Nsight Compute release.

M7 consequently does **not** claim measured achieved occupancy, memory-bandwidth utilization, warp-divergence behavior, or other unavailable GPU hardware-counter quantities. Those metrics must not be inferred from timing alone.

## CPU profiling

The fixed CPU profile modes use the same deterministic workload while bypassing benchmark calibration:

```bash
./build/thermogpu_scaling --profile scalar 1 100000
./build/thermogpu_scaling --profile scalar 5 100000
./build/thermogpu_scaling --profile omp 1 100000 8
./build/thermogpu_scaling --profile omp 5 100000 8
```

The scalar path performs one warm-up followed by 25 evaluations. The OpenMP path performs one warm-up followed by 100 evaluations. On the characterized system `perf_event_paranoid=4`, so the retained `perf stat` measurements were run with administrative privileges. Counter groups were kept small to avoid the multiplexing seen in the initial exploratory run.

### Scalar `perf stat`

| Metric | 1 component | 5 components | 5c / 1c |
|---|---:|---:|---:|
| Cycles | 2.397 B | 4.013 B | 1.674x |
| Instructions | 4.741 B | 10.364 B | 2.186x |
| Instructions/cycle | 1.978 | 2.583 | 1.306x |
| Branches | 915.2 M | 1,646.8 M | 1.799x |
| Branch misses | 0.906 M | 1.550 M | 1.710x |
| Branch-miss rate | 0.0990% | 0.0941% | lower |
| Cache references | 11.68 M | 22.54 M | 1.929x |
| Cache misses | 6.40 M | 12.52 M | 1.956x |

The generic cache events are retained only as relative evidence; their absolute miss ratio is not interpreted as a cache-hierarchy model.

### OpenMP-8 `perf stat`

| Metric | 1 component | 5 components | 5c / 1c |
|---|---:|---:|---:|
| Cycles | 24.13 B | 38.37 B | 1.590x |
| Instructions | 22.69 B | 45.49 B | 2.005x |
| Aggregate instructions/cycle | 0.940 | 1.185 | 1.261x |
| Branches | 4.605 B | 7.579 B | 1.646x |
| Branch misses | 5.170 M | 6.997 M | 1.353x |
| Branch-miss rate | 0.1123% | 0.0923% | lower |
| Wall time | 1.080 s | 1.735 s | 1.607x |

The aggregate OpenMP IPC is a process-wide multicore counter ratio and should not be compared directly with single-core scalar IPC. Its 1c-to-5c change within the same OpenMP configuration is the useful comparison.

## Interpretation

The central measured result is asymmetric component-count scaling across the CPU and GPU on this machine.

From one to five components, the scalar CPU retires about 2.19x as many instructions but consumes only about 1.67x as many cycles; IPC rises from about 1.98 to 2.58. OpenMP-8 shows the same pattern: instructions increase about 2.01x, cycles about 1.59x, and aggregate IPC rises about 26%. Branch-miss rates remain near 0.1% and decline slightly. Thus the CPU clearly performs substantially more multicomponent work, but that additional work executes more efficiently than the fixed work dominating the one-component case.

A plausible interpretation is that increasing component count shifts more execution into regular component and component-pair loops, including the Peng–Robinson mixture interaction work, while amortizing fixed per-state work such as cubic/root handling. This is an interpretation of the measured instruction/cycle behavior, not a separately measured microarchitectural decomposition.

On the GPU, the controlled resident kernel takes about 3.40x longer at five components than at one component. The same compiled kernel is used, static register allocation is unchanged at 58 registers/thread, and `ptxas` reports no spills. The GPU's proportional cost growth is therefore substantially larger than the CPU's measured cycle/time growth. This directly explains why resident CUDA speedup over the best CPU backend decreases with component count on the Precision 7710.

The measurements do **not** establish which GM204 execution resource causes the GPU's disproportionate growth. Installed Nsight Compute cannot collect the required hardware counters on this GPU. M7 therefore stops at the supported conclusion rather than attributing the behavior to occupancy, bandwidth, divergence, or another unmeasured mechanism.

## Cross-machine replication

The frozen `v0.7.0` implementation was reproduced unchanged on a Dell Precision 7680 with an Intel i9-13950HX and NVIDIA RTX 4090 Laptop GPU (Ada, compute capability 8.9) under WSL2 / Ubuntu 24.04 with CUDA 13.0. The complete configured eight-test suite passed unchanged. This is portability/reproducibility evidence across a substantially different GPU generation, CUDA toolkit, CPU, and operating environment; it is not a hardware-only controlled comparison.

Focused resident crossover brackets on the 7680 were 400–600, 500–600, 500–1200, and 500–900 states for 1/2/3/5 components respectively. These are later than the 7710 focused resident brackets even though the newer system measured much larger large-batch resident speedups. Crossover location and asymptotic throughput are therefore separate machine/workload characteristics.

The 7680 CPU/end-to-end crossover measurements require additional qualification. The i9-13950HX is physically hybrid, while WSL2 exposed 32 logical processors as 16 homogeneous SMT core pairs. The fastest stable CPU baseline also varied non-monotonically with batch size. WSL-visible `taskset` affinity worked and the computation remained deterministic, but Linux could not establish physical P-core/E-core placement. The raw measurements are retained, but focused end-to-end interpolations are not promoted as robust dispatch thresholds.

The Ada GPU also permitted Nsight Compute hardware-counter profiling that was unavailable on GM204. At 100,000 states, the one-component kernel measured about 269 us and the five-component kernel about 942 us, a ~3.50x increase. Both cases used 56 registers/thread, 75% theoretical occupancy, about 56% achieved occupancy, 1.14 waves/SM, and roughly 80% Compute (SM) throughput. DRAM throughput remained very low (~1.9–2.0% at 1c and ~1.3% at 5c).

Thus the 1c-to-5c kernel-time increase on Ada is not accompanied by increasing register pressure, occupancy collapse, changing launch geometry, DRAM saturation, or loss of overall compute utilization. The evidence supports increased per-state computational work as the primary explanation. The ~3.50x Ada duration ratio is also close to the ~3.40x Maxwell timing ratio, providing cross-platform evidence that the component-scaling behavior is associated with the PR workload/kernel structure rather than being unique to the older GPU.

The complete cross-machine synthesis and qualifications are in `results/m7/CROSS_MACHINE_ANALYSIS.md`. Raw 7680 evidence is retained under `results/m7/precision-7680/`.

## Reproduction

Build the characterized CUDA 12.4 configuration with a homogeneous GCC-13 host toolchain:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DTHERMOGPU_ENABLE_OPENMP=ON \
  -DTHERMOGPU_ENABLE_CUDA=ON
cmake --build build -j"$(nproc)"

RESULTS=results/m7/precision-7710
mkdir -p "$RESULTS"
./build/thermogpu_scaling > "$RESULTS/scaling.csv"
./build/thermogpu_scaling --crossover > "$RESULTS/crossover_scaling.csv"
python3 tools/analyze_m7.py "$RESULTS/scaling.csv" --outdir "$RESULTS"
python3 tools/analyze_m7.py "$RESULTS/crossover_scaling.csv" --outdir "$RESULTS/crossover"
```

For static CUDA resources, reconfigure with `-DTHERMOGPU_PTXAS_VERBOSE=ON` and retain the verbose build output. Profiling commands and environment notes are also summarized in `docs/benchmarking.md`.

## M7 status

M7 is complete. The primary Precision 7710 characterization was accepted and tagged `v0.7.0`; the subsequent Precision 7680 replication reproduced that frozen implementation without changing the tag. The two-machine evidence is retained separately so the replication extends rather than overwrites the original characterization.
