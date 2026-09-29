# ThermoGPU

ThermoGPU is a scientific-computing project investigating whether GPUs can materially accelerate large batches of engineering equation-of-state calculations without sacrificing numerical quality.

V1.0 targets single-phase, multicomponent Peng–Robinson (PR) calculations and compares scalar C++, OpenMP, and CUDA backends. The project is designed to produce a reproducible computational result: thermodynamic correctness is established first, optimized backends are differentially validated against that reference, and performance claims are retained with the machine-specific evidence that supports them.

**V1.0 final report:** [`docs/V1_REPORT.md`](docs/V1_REPORT.md)

The final report consolidates the numerical validation, CPU/GPU differential testing, performance characterization, cross-machine reproduction, profiling results, limitations, and engineering conclusions from the V1.0 study.

## Current status

**ThermoGPU V1.0 is complete and tagged `v1.0.0`.**

V1.0 implements and validates the scalar C++, OpenMP, and CUDA backends, completes CPU/GPU differential validation and performance characterization, and reproduces the numerical implementation on a second, substantially different CPU/GPU platform. The frozen M7 implementation remains tagged `v0.7.0`; the subsequent cross-machine evidence and V1 release integration are included in `v1.0.0`.

Implemented and validated:

- C++20 / CMake project;
- Peng–Robinson pure-component and multicomponent EOS;
- quadratic attractive mixing rule with an explicit dense symmetric `kij` matrix;
- analytic cubic solution with one-real / three-real root classification;
- numerically stabilized one-real-root Cardano evaluation;
- compressibility factor `Z`, mixture density, and component `ln(phi_i)`;
- scalar single-state and structure-of-arrays batch APIs;
- OpenMP state-parallel batch backend;
- CUDA state-parallel batch backend with reusable resident workspace;
- resident-kernel and end-to-end CUDA benchmark paths;
- invariant, limiting-case, external-reference, and cross-implementation validation;
- broad and structured CPU/CUDA differential validation;
- CPU/OpenMP/CUDA performance characterization across 1, 2, 3, and 5 components;
- cross-machine reproduction on Maxwell/native Linux and Ada/WSL2; and
- CPU `perf`, Nsight Systems, `ptxas`, and Nsight Compute profiling evidence.

## V1.0 scope

A batch consists of thermodynamic states `(P, T, z)` for a fixed component set and binary-interaction matrix. For each state ThermoGPU computes:

- compressibility factor `Z`;
- mixture density;
- component fugacity coefficients as `ln(phi_i)`;
- cubic residual;
- real-root classification; and
- root-selection ambiguity metadata.

The V1 calculation is explicitly **single phase**. The largest real PR root is selected. If the cubic has three real roots, `root_selection_ambiguous` is set rather than presenting the root choice as a phase-stability result.

V1.0 deliberately does not implement flash calculations, phase-stability analysis, saturation calculations, multiphase equilibrium, or automatic phase selection. Those are post-V1 possibilities rather than hidden assumptions in the benchmark.

See `docs/V1_SPEC.md` for the complete specification and `docs/theory.md` for the equations and units.

## Architecture

All execution backends implement the same V1 Peng–Robinson calculation.

### Scalar C++

The scalar implementation is the readable numerical reference. It is externally anchored and is the production reference for backend differential testing.

### OpenMP

The OpenMP backend parallelizes the outer state loop with static scheduling. Thermodynamic states are independent, so the backend requires no inter-state reductions or locks.

### CUDA

The CUDA backend maps one thermodynamic state to one CUDA thread. Component properties and `kij` are invariant across the batch; pressure, temperature, and state composition vary per state.

Two CUDA timing modes are intentionally kept separate:

- **resident**: reuse device storage and time repeated kernel execution; and
- **end-to-end**: exercise the public host API including allocation/data movement.

That distinction matters: fixed launch, allocation, and transfer costs can move the measured CPU/GPU crossover by orders of magnitude.

See `docs/batch_design.md` and `docs/gpu_design.md` for design details.

## Build and test

A normal CPU build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

OpenMP defaults to enabled when available. CUDA is optional and is enabled with `THERMOGPU_ENABLE_CUDA=ON`.

The characterized CUDA 12.4 environment on the Precision 7710 used GCC 13 consistently for C++ and the nvcc host compiler:

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

CUDA architecture selection may be supplied explicitly when appropriate, for example `-DCMAKE_CUDA_ARCHITECTURES=89` for the tested Ada compute-capability-8.9 target. Host-compiler compatibility must be checked against the installed CUDA toolkit rather than assuming one compiler/toolkit combination is universal.

## Batch API policy

`Mixture` stores components, mole fractions, and an explicit dense row-major `N x N` binary-interaction matrix. Mole fractions must satisfy the documented normalization tolerance; the API does **not** silently normalize them. `kij` must be finite and symmetric.

`MixtureBatch` is a structure-of-arrays workload. For `S` states and `N` components it stores:

```text
pressure_Pa[S]
temperature_K[S]
mole_fractions[S*N]
```

Compositions and component outputs are state-major, indexed as `state*N + component`. Batch results contain `Z`, density, cubic residual, root metadata, and flattened `ln_phi[S*N]`. An empty batch is valid and returns empty outputs.

## Numerical and verification policy

ThermoGPU treats unexplained numerical disagreement as something to investigate, not as a reason to widen tolerances. Constants are retained at derived precision when available, and differences caused by floating-point evaluation order, FMA, compiler transformations, or backend math are localized before they are accepted.

Validation is layered rather than delegated to one oracle.

### Unit and invariant checks

The scalar suite covers `alpha(Tc) = 1`, the low-pressure `Z -> 1` limit, cubic residuals, pure-as-mixture equivalence, finite binary/five-component results, nonzero `kij`, composition validation, and `kij` symmetry.

### Frozen Python cross-implementation reference

`tools/pr_reference.py` is separately coded and produces committed fixtures and intermediate quantities. It shares no C++ implementation code, but it was written from the same mathematical specification, so it is a **cross-implementation regression reference**, not an independent mathematical oracle.

### Independent external comparison

Controlled ThermoPack 2.2.3 pseudo-component cases provide the strongest independent third-party implementation check. After aligning component inputs and the exact PR critical constants, the controlled methane/ethane cases agree at approximately machine precision. ThermoPack remains an optional audit dependency and is not required by CMake or CTest.

### CPU/CUDA differential validation

M6 compares CUDA against the externally anchored scalar implementation over a deterministic 20,000-state CH4/C2H6 campaign spanning 0.01–20 MPa, 180–500 K, and nearly the full binary composition interval, plus a structured critical/root stress grid. `Z`, density, cubic residual, every `ln(phi_i)`, root classification, and ambiguity metadata are checked.

During this campaign, a one-ULP CPU/CUDA difference in a cubic coefficient exposed cancellation in the original one-real-root Cardano evaluation. Instead of disabling FMA or relaxing tolerances, the solver was changed to compute the larger cube-root term directly and recover the smaller term from `uv = -p/3`. The problematic residual improved from roughly `1.7e-12` to `7e-18`.

See `docs/verification.md` and `docs/m6_validation.md` for the validation hierarchy and numerical-stability work.

## Inspect a scalar EOS calculation

`thermogpu_eval` evaluates a built-in five-component natural-gas-like state at 8 MPa and 320 K with zero binary interaction coefficients:

```bash
./build/thermogpu_eval
```

The report includes the input composition, mixture molecular weight, `a_m`, `b_m`, dimensionless `A` and `B`, all real cubic roots, selected `Z`, cubic residual, density, and component `ln(phi)` / `phi` values.

## Benchmarking

`thermogpu_benchmark` provides the original scalar/OpenMP/CUDA throughput benchmark. `thermogpu_scaling` extends the experiment across 1-, 2-, 3-, and 5-component mixtures.

Production timing uses deterministic workloads, warm-up, calibrated repeated calls targeting about 200 ms per sample, nine samples, the median as the primary statistic, and median absolute deviation as a stability diagnostic. Correctness checks are outside the timed region. CUDA resident and end-to-end timings are reported separately.

Example:

```bash
./build/thermogpu_scaling > scaling.csv
python3 tools/analyze_m7.py scaling.csv --outdir results
```

`--quick` is a smoke test, not performance evidence. Crossover brackets are measured sign changes around speedup 1; local interpolation is descriptive and is not a fitted universal performance model.

See `docs/benchmarking.md` and `docs/m7_performance.md`.

## M7 performance findings

### Precision 7710: primary characterization

The primary M7 campaign used a Dell Precision 7710 with an Intel i7-6920HQ and NVIDIA Quadro M3000M (Maxwell, `sm_52`) under native Linux with CUDA 12.4.

Focused measured crossover brackets were:

| Components | Resident CUDA | End-to-end CUDA |
|---:|---:|---:|
| 1 | 175–200 states | 5000–5500 states |
| 2 | 150–175 states | 4000–4500 states |
| 3 | 300–350 states | 3500–4000 states |
| 5 | 300–350 states | 3000–3500 states |

At five components and one million states, measured times were approximately 533.6 ns/state scalar, 173.8 ns/state OpenMP-8, 29.33 ns/state resident CUDA, and 96.83 ns/state end-to-end CUDA. These are machine- and workload-specific measurements, not dispatch constants.

CPU profiling showed that 1c -> 5c roughly doubled retired instructions but increased cycles/time only about 1.6–1.7x, with improved IPC. Nsight Systems showed the resident CUDA kernel increasing about 3.40x, from approximately 1.108 ms to 3.766 ms at 100,000 states. The Maxwell profiler environment could not provide the required Nsight Compute hardware counters, so M7 did not invent an occupancy, bandwidth, or divergence explanation for that machine.

### Precision 7680: cross-machine replication

The frozen `v0.7.0` implementation was then built and tested unchanged on a Dell Precision 7680 with an Intel i9-13950HX and RTX 4090 Laptop GPU (Ada, compute capability 8.9) under WSL2 / Ubuntu 24.04 with CUDA 13.0. The complete configured eight-test suite passed unchanged.

Focused resident crossover brackets moved later on this system:

| Components | Resident crossover bracket |
|---:|---:|
| 1 | 400–600 states |
| 2 | 500–600 states |
| 3 | 500–1200 states |
| 5 | 500–900 states |

At the same time, the campaign measured much larger large-batch resident speedups. This demonstrates that **crossover location and asymptotic throughput are different characteristics**: a much faster GPU need not cross a much faster host CPU at a smaller batch size.

Precise CPU/end-to-end crossover characterization on the Precision 7680 is intentionally qualified. The i9-13950HX is physically hybrid, while WSL2 exposed 32 logical processors as 16 homogeneous SMT core pairs. The fastest stable CPU baseline also varied non-monotonically with batch size. WSL-visible affinity could be constrained, but it could not establish physical P-core/E-core placement. The raw measurements are retained; their interpolated end-to-end crossovers are not promoted as robust dispatch thresholds.

### Ada hardware-counter result

Nsight Compute on the Ada system allowed the component-scaling question to be tested directly at 100,000 states:

| Metric | 1 component | 5 components |
|---|---:|---:|
| Kernel duration | ~269 us | ~942 us |
| Relative duration | 1.00x | ~3.50x |
| Compute (SM) throughput | ~80.6% | ~80.2–80.7% |
| DRAM throughput | ~1.9–2.0% | ~1.3% |
| Registers/thread | 56 | 56 |
| Theoretical occupancy | 75% | 75% |
| Achieved occupancy | ~55.8% | ~56.1% |
| Waves/SM | 1.14 | 1.14 |

The 1c -> 5c increase therefore occurs without increasing register count, collapsing occupancy, changing launch geometry, saturating DRAM bandwidth, or losing overall compute utilization. The evidence supports the simpler explanation: each CUDA thread performs substantially more component-dependent PR work, including the quadratic mixture interaction work and component fugacity calculations.

The approximately 3.50x Ada duration ratio is close to the approximately 3.40x Maxwell timing ratio. Because the systems differ in GPU architecture, CUDA toolkit, CPU, and operating environment, this is not a hardware-only controlled comparison. It is nevertheless useful cross-platform evidence that the component-count behavior is associated with the workload/kernel structure rather than being unique to GM204.

The complete synthesis is retained in `results/m7/CROSS_MACHINE_ANALYSIS.md`; raw evidence is under `results/m7/precision-7710/` and `results/m7/precision-7680/`.

## Reproducibility notes

The repository retains machine-specific benchmark CSVs, generated plots, profiler summaries, environment records, and validation output needed to trace the published M7 conclusions back to measurements.

The two characterized environments intentionally differ. The cross-machine experiment therefore demonstrates numerical portability and repeatability of qualitative workload behavior, not a controlled hardware-only speed comparison.

Analysis and external-validation tools use a separate Python environment and are not runtime dependencies of the C++ library. See `requirements-analysis.txt` and `docs/reproducibility.md` for setup and reproducibility instructions.

## Documentation map

- `docs/V1_REPORT.md` — final V1.0 technical report and engineering conclusions
- `docs/V1_SPEC.md` — complete V1.0 scope and definition of done
- `docs/theory.md` — Peng–Robinson formulation and units
- `docs/batch_design.md` — batch representation and layout
- `docs/gpu_design.md` — CUDA backend design
- `docs/verification.md` — validation hierarchy
- `docs/m6_validation.md` — CPU/CUDA differential-validation campaign
- `docs/benchmarking.md` — benchmark and profiler workflow
- `docs/m7_performance.md` — detailed M7 performance analysis
- `docs/reproducibility.md` — build, validation, analysis, and cross-machine reproducibility
- `results/m7/CROSS_MACHINE_ANALYSIS.md` — two-machine synthesis
- `tests/reference/README.md` — reference-case provenance

## License

ThermoGPU is free software licensed under the GNU General Public License, version 3 or, at your option, any later version (`GPL-3.0-or-later`). See `LICENSE` for the full license text.
