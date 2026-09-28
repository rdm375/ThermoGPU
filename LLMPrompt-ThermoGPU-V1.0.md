# ThermoGPU

## V1.0 Project Specification

### 1. Purpose

ThermoGPU is an experimental scientific-computing project investigating GPU acceleration of thermodynamic equation-of-state calculations used in engineering simulation.

Version 1.0 will answer one narrowly defined question:

> **Can GPU execution materially accelerate large batches of single-phase Peng–Robinson mixture calculations relative to scalar and multicore CPU implementations, while maintaining engineering-quality numerical agreement?**

The project is intended to produce a reproducible computational result rather than merely demonstrate CUDA programming.

The implementation will emphasize:

- mathematical correctness,
- numerical verification,
- reproducibility,
- CPU/GPU equivalence,
- performance measurement,
- clear software architecture,
- and defensible engineering conclusions.

---

# 2. V1.0 Scope

V1.0 implements the Peng–Robinson equation of state for **single-phase multicomponent mixtures**.

Given a batch of thermodynamic states

\[
(P_s,T_s,\mathbf z_s), \qquad s=1,\ldots,N,
\]

the software will calculate:

\[
Z_s,
\]

\[
\rho_s,
\]

and component fugacity coefficients

\[
\ln\phi_{i,s}.
\]

The same mathematical model will be implemented using three execution backends:

1. scalar CPU,
2. multicore CPU using OpenMP,
3. NVIDIA GPU using CUDA.

All three implementations must solve the same defined thermodynamic problem.

---

# 3. Explicit Non-Goals

The following are **not part of V1.0**:

- vapor-liquid flash calculations,
- phase-stability analysis,
- multiphase equilibrium,
- reservoir simulation,
- pipeline simulation,
- transient flow simulation,
- transport properties,
- viscosity correlations,
- thermal conductivity,
- enthalpy or entropy calculations,
- automatic differentiation,
- EOS parameter fitting,
- alternative equations of state,
- graphical user interfaces,
- distributed computing,
- multi-GPU execution.

These may become later projects or later ThermoGPU releases.

They are not prerequisites for V1.0.

---

# 4. Thermodynamic Model

## 4.1 Peng–Robinson EOS

Implement the Peng–Robinson equation

\[
P =
\frac{RT}{v-b}
-
\frac{a\alpha}
{v(v+b)+b(v-b)}.
\]

For each pure component,

\[
a_i =
0.4572355289213822
\frac{R^2T_{c,i}^2}{P_{c,i}},
\]

\[
b_i =
0.0777960739038884
\frac{RT_{c,i}}{P_{c,i}}.
\]

Temperature dependence will use the standard Peng–Robinson alpha formulation

\[
\alpha_i(T)
=
\left[
1+\kappa_i
\left(
1-\sqrt{T/T_{c,i}}
\right)
\right]^2,
\]

with the standard acentric-factor correlation for \(\kappa_i\).

All equations, constants, units, and conventions must be explicitly documented.

---

# 5. Mixture Model

Implement conventional quadratic mixing for the attractive parameter:

\[
a_m =
\sum_i\sum_j
z_i z_j
\sqrt{a_i\alpha_i\,a_j\alpha_j}
(1-k_{ij}),
\]

and

\[
b_m=\sum_i z_i b_i.
\]

Binary interaction coefficients \(k_{ij}\) will be supported.

For the basic verification cases,

\[
k_{ij}=0
\]

may be used where appropriate.

The binary-interaction matrix must nevertheless be represented explicitly so the implementation does not assume zero interaction coefficients internally.

---

# 6. EOS Solution

For each state, construct the dimensionless Peng–Robinson parameters

\[
A =
\frac{a_mP}{R^2T^2},
\]

\[
B =
\frac{b_mP}{RT}.
\]

Solve the corresponding cubic equation in \(Z\):

\[
Z^3
-(1-B)Z^2
+(A-3B^2-2B)Z
-(AB-B^2-B^3)=0.
\]

V1.0 must provide a deterministic and documented root-selection rule appropriate for the declared single-phase calculation.

The implementation must detect and report states for which the single-phase assumption or root-selection rule is ambiguous rather than silently producing an arbitrary result.

---

# 7. Required Outputs

For every state, calculate at minimum:

### Compressibility factor

\[
Z
\]

### Mixture density

\[
\rho =
\frac{P\bar M}{ZRT},
\]

where

\[
\bar M=\sum_i z_iM_i.
\]

### Component fugacity coefficients

\[
\ln\phi_i.
\]

Intermediate quantities needed for verification should optionally be exposed by the reference implementation, including:

\[
A,\quad B,\quad a_m,\quad b_m.
\]

---

# 8. Initial Component Set

V1.0 will include thermodynamic data for a small natural-gas-oriented component set:

- methane,
- ethane,
- propane,
- nitrogen,
- carbon dioxide.

Required component properties include:

\[
T_c,\quad P_c,\quad \omega,\quad M.
\]

The source and units of every physical constant must be documented.

The architecture must permit additional components without changing the EOS implementation.

---

# 9. Units

Internally, calculations will use SI units.

At minimum:

\[
P:\mathrm{Pa}
\]

\[
T:\mathrm{K}
\]

\[
M:\mathrm{kg/mol}
\]

\[
\rho:\mathrm{kg/m^3}.
\]

Unit conversions must occur at interfaces rather than inside the EOS kernels.

No implicit field-unit conventions are permitted.

---

# 10. Reference CPU Implementation

A deliberately straightforward scalar C++ implementation will serve as the numerical reference.

Its priorities are:

1. readability,
2. correspondence with the mathematical formulation,
3. testability,
4. correctness.

Performance optimization is secondary.

The reference implementation must not share optimized computational kernels with the CUDA implementation in ways that could cause the same implementation defect to appear identically in both versions.

---

# 11. OpenMP Implementation

A multicore CPU backend will evaluate independent thermodynamic states in parallel using OpenMP.

The primary parallel decomposition will be

\[
\text{one state}\leftrightarrow\text{one independent work item}.
\]

The OpenMP implementation provides the practical CPU baseline against which GPU acceleration will be evaluated.

Benchmarks must report the CPU model and number of threads used.

---

# 12. CUDA Implementation

The CUDA backend will initially use the same state-level decomposition:

\[
\boxed{\text{one CUDA thread}\leftrightarrow\text{one thermodynamic state}}.
\]

Each thread will calculate the Peng–Robinson properties for one independent state.

V1.0 should favor clarity and correctness over aggressive GPU-specific optimization.

Subsequent optimization may investigate:

- memory layout,
- constant memory,
- shared memory,
- register pressure,
- occupancy,
- kernel fusion,
- structure-of-arrays representation,
- component-loop organization.

Optimization must be driven by profiling rather than assumption.

---

# 13. Precision

V1.0 will use IEEE-754 double precision for all primary calculations.

Single precision and mixed precision are explicitly deferred.

This establishes a high-quality numerical baseline before precision/performance tradeoffs are investigated.

---

# 14. Verification Strategy

Verification will occur at several independent levels.

## 14.1 Algebraic Tests

Computed roots must satisfy the Peng–Robinson cubic.

For calculated \(Z\),

\[
r(Z)=
Z^3-(1-B)Z^2+
(A-3B^2-2B)Z-
(AB-B^2-B^3).
\]

Require

\[
|r(Z)|
\]

to remain below a documented numerical tolerance.

---

## 14.2 Limiting Cases

Tests will include physically meaningful limits such as

\[
P\rightarrow0
\]

for which

\[
Z\rightarrow1.
\]

Pure-component cases will be tested separately from mixtures.

---

## 14.3 Composition Tests

For every input state,

\[
\sum_i z_i=1
\]

within a documented tolerance.

Invalid compositions must be rejected or explicitly normalized according to a documented API policy.

Silent normalization is prohibited unless explicitly requested.

---

## 14.4 Independent Reference Comparison

Selected states will be compared against an independent Peng–Robinson implementation.

Differences between implementations must be investigated before accepting the test case.

Comparison against a more sophisticated reference EOS may additionally be performed to characterize **model error**, but such disagreement must not automatically be interpreted as implementation error.

---

## 14.5 CPU/GPU Differential Testing

Large randomized state populations will be evaluated by all three backends.

For each property, calculate statistics including:

\[
\max |\Delta|,
\]

\[
\operatorname{RMS}(\Delta),
\]

and relative error where meaningful.

The states producing the largest discrepancies will be retained as permanent regression cases.

---

# 15. Pathological Test Cases

The verification suite must deliberately include difficult thermodynamic states rather than relying exclusively on random sampling.

These should include:

- low-pressure states,
- high-pressure states,
- near-critical states,
- nearly pure mixtures,
- trace components,
- strongly asymmetric compositions,
- states producing multiple real cubic roots.

States for which V1.0's single-phase interpretation becomes ambiguous should be detected and classified.

---

# 16. Benchmarking

Performance testing is a primary deliverable.

Benchmark across batch sizes such as

\[
N=
1,\,
10,\,
10^2,\,
10^3,\,
10^4,\,
10^5,\,
10^6
\]

and larger where hardware permits.

Test multiple component counts where practical.

At minimum report:

### Wall-clock execution time

\[
t(N)
\]

### Throughput

\[
Q(N)=\frac{N}{t(N)}
\]

in states per second.

### GPU speedup over scalar CPU

\[
S_\mathrm{scalar}(N)=
\frac{t_\mathrm{scalar}(N)}
{t_\mathrm{GPU}(N)}.
\]

### GPU speedup over multicore CPU

\[
S_\mathrm{OMP}(N)=
\frac{t_\mathrm{OMP}(N)}
{t_\mathrm{GPU}(N)}.
\]

---

# 17. Data-Transfer Accounting

GPU benchmarks must distinguish between:

### Kernel-only execution time

and

### End-to-end execution time

including host/device data transfer.

This distinction is essential.

A GPU implementation that accelerates the EOS kernel but loses the advantage through transfer overhead must not be reported simply as a GPU speedup.

---

# 18. Crossover Analysis

One of the principal V1.0 results will be determining the approximate batch size

\[
N^*
\]

for which GPU execution becomes faster than the relevant CPU implementation:

\[
t_\mathrm{GPU}(N^*)
<
t_\mathrm{CPU}(N^*).
\]

The project should characterize how this crossover depends upon:

- number of states,
- number of components,
- CPU thread count,
- GPU hardware,
- whether transfer costs are included.

---

# 19. Profiling

GPU profiling should investigate at minimum:

- kernel execution time,
- memory-transfer time,
- achieved occupancy,
- memory bandwidth,
- warp divergence,
- register usage.

CPU profiling should identify:

- EOS arithmetic cost,
- mixture-rule cost,
- cubic-solution cost,
- fugacity calculation cost,
- parallel scaling.

Optimization decisions must be documented with supporting measurements.

---

# 20. Implementation Languages

Primary implementation:

**C++20**

GPU implementation:

**CUDA C++**

Parallel CPU implementation:

**OpenMP**

Build system:

**CMake**

Analysis and plotting:

**Python**

Python may use packages such as NumPy, pandas, and Matplotlib for analysis, but the production EOS implementations will remain C++/CUDA.

---

# 21. Proposed Repository Structure

```text
thermogpu/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── include/
│   └── thermogpu/
│       ├── component.hpp
│       ├── mixture.hpp
│       ├── state.hpp
│       ├── result.hpp
│       └── peng_robinson.hpp
│
├── src/
│   ├── common/
│   ├── cpu/
│   │   ├── pr_scalar.cpp
│   │   └── pr_openmp.cpp
│   │
│   └── cuda/
│       └── pr_cuda.cu
│
├── tests/
│   ├── unit/
│   ├── regression/
│   └── reference/
│
├── benchmarks/
│   ├── benchmark_cpu.cpp
│   ├── benchmark_cuda.cu
│   └── datasets/
│
├── python/
│   ├── analyze_accuracy.py
│   ├── analyze_performance.py
│   └── generate_plots.py
│
├── data/
│   ├── components.csv
│   └── binary_interactions.csv
│
├── docs/
│   ├── theory.md
│   ├── verification.md
│   ├── gpu_design.md
│   └── benchmarking.md
│
└── results/
    ├── accuracy/
    ├── performance/
    └── figures/
```

---

# 22. Required V1.0 Figures

The repository should contain reproducibly generated figures showing at least:

### Throughput versus batch size

\[
Q(N)
\]

for scalar CPU, OpenMP CPU, and CUDA.

### Runtime versus batch size

\[
t(N).
\]

### GPU speedup versus batch size

\[
S(N).
\]

### CPU/GPU numerical disagreement

Property-error distributions for

\[
Z,\quad\rho,\quad\ln\phi_i.
\]

### CPU/GPU crossover

A figure clearly showing the workload regime where GPU execution becomes advantageous.

---

# 23. Reproducibility

A user with compatible hardware should be able to:

```bash
git clone ...
cmake -S . -B build
cmake --build build
ctest --test-dir build
./build/benchmarks/thermogpu_benchmark
python python/generate_plots.py
```

and reproduce the major verification and performance results.

Exact commands may evolve during implementation, but V1.0 must provide an equivalent documented workflow.

---

# 24. Continuous Integration

CPU builds and verification tests should run automatically in CI.

CI should test at least:

- compilation,
- unit tests,
- reference cases,
- regression cases.

CUDA CI is desirable where suitable infrastructure is available but is not required for declaring V1.0 complete.

---

# 25. Documentation Requirements

The repository must contain enough documentation for another numerical-software engineer to understand:

1. what problem is being solved,
2. the governing equations,
3. thermodynamic conventions,
4. numerical algorithms,
5. CPU/GPU architecture,
6. verification methodology,
7. benchmark methodology,
8. known limitations,
9. reproducibility procedure.

The README should summarize the project rather than duplicate the detailed technical documentation.

---

# 26. Required Engineering Conclusions

V1.0 is not complete merely when the CUDA implementation works.

The project must answer:

1. How accurately does the GPU implementation reproduce the CPU reference?
2. What portions of PR evaluation dominate execution time?
3. How effectively does OpenMP scale?
4. How effectively does CUDA scale with batch size?
5. At approximately what workload does GPU execution become advantageous?
6. How significant is host/device transfer overhead?
7. How does component count affect CPU and GPU performance?
8. What numerical or architectural limitations were discovered?

Negative results are valid results.

If GPU acceleration is ineffective for some workload regimes, those regimes must be reported rather than hidden.

---

# 27. Definition of Done

ThermoGPU V1.0 is complete when:

- Peng–Robinson pure-component calculations work.
- Peng–Robinson mixture calculations work.
- Binary interaction coefficients are supported.
- \(Z\), density, and component fugacity coefficients are calculated.
- Scalar CPU implementation passes verification.
- OpenMP implementation agrees with the scalar reference.
- CUDA implementation agrees with the scalar reference.
- Pathological states are included in testing.
- Large randomized differential tests pass documented tolerances.
- CPU and GPU benchmarks are reproducible.
- Host/device transfer costs are separately measured.
- Performance crossover behavior is characterized.
- Profiling results are documented.
- Major benchmark figures are reproducibly generated.
- Theory and verification methodology are documented.
- Known limitations are explicitly stated.
- CI verifies the CPU implementation.
- A clean checkout can reproduce the CPU results using documented commands.

At that point the project is tagged:

**`v1.0.0`**

and further thermodynamic or application development occurs only after the V1.0 baseline has been preserved.

---

# 28. Post-V1.0 Possibilities

Possible future work includes:

```text
V1.x
 ├── additional components
 ├── additional validation datasets
 ├── CPU SIMD optimization
 └── CUDA kernel optimization

V2
 └── PT flash / phase equilibrium

V3
 └── GPU-accelerated flash calculations

V4
 ├── transient pipeline integration
 └── reservoir-simulation integration

V5
 └── ensemble simulation / optimization / uncertainty quantification
```

These are intentionally excluded from the V1.0 completion criteria.

---

# 29. V1.0 Success Criterion

The ultimate V1.0 deliverable is not simply software that evaluates Peng–Robinson.

It is a reproducible answer to:

> **For batched engineering EOS calculations, when does GPU computation become advantageous over conventional CPU execution, by how much, and at what numerical cost?**

The software, verification suite, benchmarks, profiling data, and documentation together constitute the result.
