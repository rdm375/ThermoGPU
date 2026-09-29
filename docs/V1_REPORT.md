# ThermoGPU V1.0 --- Final Report

## Executive summary

ThermoGPU V1.0 investigated a narrow scientific-computing question:
**can GPU execution materially accelerate large batches of single-phase
multicomponent Peng--Robinson (PR) equation-of-state calculations
relative to scalar and multicore CPU implementations while maintaining
engineering-quality numerical agreement?**

The project implemented the same PR workload in scalar C++, OpenMP, and
CUDA, then treated correctness and performance as separate experimental
questions. The scalar implementation was checked through unit and
invariant tests, a separately coded Python cross-implementation
reference, and controlled comparison with ThermoPack 2.2.3. The CUDA
backend was then differentially validated against the externally
anchored scalar implementation over broad and structured state
campaigns.

The principal results are:

-   Controlled ThermoPack cases agreed with ThermoGPU at approximately
    machine precision after aligning pseudo-component setup and the
    exact PR critical constants.
-   A 20,000-state broad CPU/CUDA differential campaign, supplemented by
    a structured critical/root stress grid, produced no
    root-classification or ambiguity mismatches. The campaign exposed a
    numerically unstable one-real-root Cardano formulation: a one-ULP
    CPU/GPU coefficient difference could be amplified by cancellation.
    Replacing that evaluation with a stable formulation reduced the
    problematic cubic residual from roughly `1.7e-12` to `7e-18`,
    without disabling FMA or relaxing validation tolerances.
-   On the Dell Precision 7710, the five-component, one-million-state
    workload measured approximately `533.6 ns/state` scalar,
    `173.8 ns/state` OpenMP-8, `29.33 ns/state` CUDA resident, and
    `96.83 ns/state` CUDA end-to-end. Resident CUDA was therefore about
    `18.2x` faster than scalar and `5.93x` faster than the best measured
    CPU backend; end-to-end CUDA was about `5.51x` faster than scalar
    and `1.79x` faster than the best CPU backend.
-   Focused Precision 7710 CUDA-resident crossover estimates were
    approximately 191, 171, 328, and 339 states for 1-, 2-, 3-, and
    5-component mixtures. End-to-end crossover estimates were
    approximately 5,004, 4,414, 3,989, and 3,402 states. These are
    machine- and workload-specific measurements, not universal dispatch
    thresholds.
-   The frozen `v0.7.0` implementation and complete configured test
    suite reproduced unchanged on a substantially different Dell
    Precision 7680 / RTX 4090 Laptop GPU system under WSL2 and CUDA
    13.0.
-   Controlled profiling showed that increasing the CUDA workload from
    one to five components increased kernel duration by about `3.40x` on
    Maxwell and `3.50x` on Ada. Ada hardware counters showed essentially
    unchanged register count, occupancy, launch geometry, and compute
    utilization, with very low DRAM utilization. The evidence therefore
    supports increased component-dependent PR arithmetic as the primary
    explanation rather than DRAM saturation, occupancy collapse, or
    increasing register pressure.

The central engineering conclusion is not merely that a GPU can be
faster. ThermoGPU demonstrates a workflow in which numerical equivalence
is established first, discrepancies are investigated causally, GPU
overhead is separated from resident computation, crossover is measured
rather than assumed, and performance conclusions remain tied to the
hardware and execution environment that produced them.

## 1. Research question and scope

ThermoGPU V1.0 evaluates batches of thermodynamic states

\[ (P_s,T_s,`\mathbf `{=tex}z_s), `\qquad `{=tex}s=1,`\ldots`{=tex},N \]

for a fixed component set and binary-interaction matrix. For each state
it computes:

-   compressibility factor `Z`;
-   mixture density;
-   component fugacity coefficients `ln(phi_i)`;
-   cubic residual;
-   real-root classification; and
-   root-selection ambiguity metadata.

The V1 calculation is explicitly **single phase**. The largest real
Peng--Robinson root is selected. If the cubic has three real roots, the
result is flagged as root-selection ambiguous rather than being
presented as a phase-stability determination.

V1.0 deliberately excludes vapor-liquid flash calculations,
phase-stability analysis, saturation calculations, multiphase
equilibrium, automatic phase selection, transport properties, EOS
fitting, distributed execution, and multi-GPU execution. The purpose was
to keep the mathematical workload narrow enough that numerical and
performance behavior could be studied rigorously.

The initial component set is natural-gas oriented: methane, ethane,
propane, nitrogen, and carbon dioxide. Binary interaction coefficients
are represented explicitly through a dense symmetric `kij` matrix.

## 2. Thermodynamic model

ThermoGPU implements the Peng--Robinson equation of state

\[ P = `\frac{RT}{v-b}`{=tex} - `\frac{a\alpha}{v(v+b)+b(v-b)}`{=tex}.
\]

For each pure component,

\[ a_i = `\Omega`{=tex}\_A`\frac{R^2T_{c,i}^2}{P_{c,i}}`{=tex},
`\qquad`{=tex} b_i = `\Omega`{=tex}\_B`\frac{RT_{c,i}}{P_{c,i}}`{=tex},
\]

with

\[ `\Omega`{=tex}\_A = 0.4572355289213822, `\qquad`{=tex}
`\Omega`{=tex}\_B = 0.0777960739038884, \]

and

\[ R = 8.31446261815324 `\mathrm{J\,mol^{-1}\,K^{-1}}`{=tex}. \]

The attractive mixture parameter uses the conventional quadratic mixing
rule,

\[ a_m = `\sum`{=tex}\_i`\sum`{=tex}*j z_i z_j
`\sqrt{a_i\alpha_i\,a_j\alpha_j}`{=tex}(1-k*{ij}), \]

and

\[ b_m = `\sum`{=tex}\_i z_i b_i. \]

The dimensionless PR parameters are

\[ A=`\frac{a_mP}{R^2T^2}`{=tex}, `\qquad`{=tex}
B=`\frac{b_mP}{RT}`{=tex}, \]

leading to the cubic

\[ Z^3-(1-B)Z^2+(A-3B^2-2B)Z-(AB-B^2-B\^3)=0. \]

All production backends evaluate the same declared thermodynamic
problem.

## 3. Software architecture

### 3.1 Scalar C++

The scalar implementation is the readable production reference. It
supplies both single-state and batch evaluation and is the reference
used for backend differential testing.

The scalar implementation is not treated as correct merely because it is
the CPU version. It is anchored through invariant tests and external
comparison before being used as the reference for OpenMP and CUDA.

### 3.2 OpenMP

The OpenMP backend parallelizes the outer state loop with static
scheduling. Thermodynamic states are independent, so no inter-state
reduction or locking is required.

OpenMP measurements include 1-, 2-, 4-, and 8-thread configurations
where appropriate. OpenMP-1 is retained as an overhead diagnostic rather
than assumed to be a useful performance configuration.

### 3.3 CUDA

The CUDA backend maps one thermodynamic state to one CUDA thread.
Component properties and the binary-interaction matrix are invariant
across a batch; pressure, temperature, and composition vary by state.

Two CUDA timing modes are deliberately distinguished:

-   **resident** --- reusable device storage is retained and repeated
    kernel execution is timed;
-   **end-to-end** --- the public host API is exercised, including
    allocation and host/device data movement.

This distinction is essential because fixed allocation, transfer, and
launch costs can move the CPU/GPU crossover by orders of magnitude.

## 4. Verification strategy

ThermoGPU uses layered verification rather than relying on a single
oracle.

### 4.1 Unit, invariant, and limiting-case checks

The scalar test suite covers, among other cases:

-   `alpha(Tc) = 1`;
-   the low-pressure `Z -> 1` limit;
-   cubic residual checks;
-   pure-component versus pure-as-mixture equivalence;
-   finite binary and five-component calculations;
-   nonzero `kij`;
-   composition validation; and
-   `kij` symmetry.

### 4.2 Python cross-implementation reference

`tools/pr_reference.py` is separately coded and produces committed
fixtures and intermediate quantities. It shares no C++ implementation
code, but it was written from the same mathematical specification. It is
therefore a **cross-implementation regression reference**, not an
independent mathematical oracle.

### 4.3 Independent ThermoPack comparison

The strongest independent third-party check uses ThermoPack 2.2.3 with
PR, vdW mixing, Classic alpha, genuine `PSEUDO` component slots, and the
same component constants.

For the controlled methane/ethane case at 300 K and 5 MPa with `kij=0`,
the documented differences were approximately:

-   `dZ = 1.11e-16`;
-   `dlnphi_CH4 = 2.22e-16`;
-   `dlnphi_C2H6 = 3.33e-16`.

The investigation also explained an earlier few-ppm disagreement.
ThermoPack pseudo-component properties must be installed into slots
declared `PSEUDO`, and ThermoPack derives the PR critical constants
rather than using rounded textbook values. ThermoGPU consequently
retains the derived PR constants shown above.

ThermoPack is an optional audit dependency; it is not a runtime
dependency of the C++ library.

### 4.4 CPU/CUDA differential campaign

M6 asks whether CUDA reproduces the externally anchored scalar
thermodynamics throughout the intended V1 domain.

The deterministic broad campaign samples 20,000 CH4/C2H6 states over:

-   pressure: `0.01–20 MPa`;
-   temperature: `180–500 K`;
-   methane mole fraction: `0.0001–0.9999`.

A structured grid additionally exercises critical temperatures and
pressures, low-temperature/high-pressure combinations, and near-pure
through mixed compositions.

For every state the validator compares:

-   `Z`;
-   density;
-   cubic residual;
-   every `ln(phi_i)`;
-   root classification; and
-   root-selection ambiguity.

The final broad comparison recorded zero root/ambiguity mismatches.
Maximum documented discrepancies were approximately `3.05e-15` absolute
in `Z` and `9.13e-15` absolute in `ln(phi_i)`. A separate structured
campaign accepted 1,727 states, rejected one state according to the
campaign's admissibility logic, and produced zero root/ambiguity
mismatches among accepted states.

## 5. Numerical-stability finding

The differential campaign produced one of the most important results of
the project.

A one-ULP CPU/CUDA difference in a cubic coefficient was amplified by
cancellation in the original one-real-root Cardano evaluation. The
discrepancy was not handled by disabling fused multiply-add operations
or widening tolerances. Instead, the algebraic formulation was changed.

For the depressed cubic, the stable implementation computes the
larger-magnitude cube-root term directly and recovers the smaller term
using the identity

\[ uv=-p/3. \]

For the problematic state, the cubic residual improved from
approximately

\[ 1.7`\times10`{=tex}\^{-12} \]

to approximately

\[ 7`\times10`{=tex}\^{-18}. \]

This episode illustrates the project's numerical policy: unexplained
backend disagreement is treated as diagnostic evidence. Tolerances are
not relaxed until the cause is understood.

## 6. Performance methodology

Production performance measurements use deterministic workloads,
warm-up, calibrated repeated calls targeting roughly 200 ms per sample,
nine samples, the median as the primary statistic, and median absolute
deviation (MAD) as a stability diagnostic. Correctness checks are kept
outside the timed region.

The component-scaling study evaluates 1-, 2-, 3-, and 5-component
mixtures over broad batch-size ranges. Focused sweeps are then used to
resolve crossover regions.

A measured crossover is defined by adjacent sampled points around
speedup 1. Local interpolation is descriptive; the measured bracket
remains authoritative. No crossover measured on one machine is treated
as a universal dispatch threshold.

## 7. Precision 7710 performance results

The primary controlled M7 campaign used:

-   Dell Precision 7710;
-   Intel Core i7-6920HQ;
-   4 physical cores / 8 hardware threads;
-   NVIDIA Quadro M3000M;
-   Maxwell, compute capability 5.2;
-   native Linux;
-   GCC 13;
-   CUDA 12.4.

### 7.1 CPU behavior

A simple affine timing model over the focused CPU crossover region
produced the following approximate marginal costs:

  Backend      Fixed term    Marginal term
  ---------- ------------ ----------------
  Scalar         0.546 us   326.3 ns/state
  OpenMP-1       1.685 us   381.2 ns/state
  OpenMP-2       3.343 us   198.9 ns/state
  OpenMP-4       3.805 us   106.0 ns/state
  OpenMP-8       4.361 us   100.3 ns/state

The corresponding fitted scalar/OpenMP crossover estimates were
approximately 22 states for OpenMP-2, 15 for OpenMP-4, and 17 for
OpenMP-8. These compact fits describe the measured crossover region;
they are not universal execution models.

At large batch sizes, four physical cores supplied nearly all useful CPU
parallelism, while eight SMT threads added comparatively little.

### 7.2 Focused CUDA crossover

The authoritative focused M7 component-scaling results were:

    Components                Resident CUDA crossover   End-to-end CUDA crossover
  ------------ -------------------------------------- ---------------------------
             1   175--200 states (\~191 interpolated)         5000--5500 (\~5004)
             2                       150--175 (\~171)         4000--4500 (\~4414)
             3                       300--350 (\~328)         3500--4000 (\~3989)
             5                       300--350 (\~339)         3000--3500 (\~3402)

The resident crossover is non-monotonic with component count. The
project records that behavior without inventing a causal explanation.

The end-to-end crossover moves downward as component count increases,
consistent with increased per-state computation amortizing fixed
allocation and transfer costs.

### 7.3 Large-batch throughput

At five components and one million states:

  ------------------------------------------------------------------------
  Backend          Time/state        Approx.     Speedup vs     Speedup vs
                                  throughput         scalar       best CPU
  ------------ -------------- -------------- -------------- --------------
  Scalar             533.6 ns         1.87 M          1.00x            ---
                                    states/s                

  OpenMP-8           173.8 ns         5.75 M          3.07x          1.00x
                                    states/s                

  CUDA               29.33 ns         34.1 M          18.2x          5.93x
  resident                          states/s                

  CUDA               96.83 ns        10.33 M          5.51x          1.79x
  end-to-end                        states/s                
  ------------------------------------------------------------------------

These are measurements of this machine and workload. They are evidence
of a substantial large-batch GPU advantage, not portable constants.

## 8. Component-count scaling and profiling

The broad M7 measurements showed that resident CUDA speedup over the
best CPU backend decreased as component count increased. Profiling was
used to investigate why.

### 8.1 CPU profiling

On the Precision 7710, increasing from one to five components
approximately doubled retired scalar instruction count while increasing
measured cycles by only about 1.7x. Scalar IPC increased from roughly
1.98 to 2.58. OpenMP-8 showed a similar pattern: wall time increased
about 1.61x while retired instructions increased about 2.01x.

The CPU therefore handled the additional regular arithmetic more
efficiently than raw instruction-count growth alone would suggest.

### 8.2 Maxwell CUDA profiling

Nsight Systems measured the controlled 100,000-state resident PR kernel
at approximately:

-   one component: `1.11 ms`;
-   five components: `3.76 ms`.

The increase was about `3.40x` and was stable across repeated launches.

The Maxwell production kernel used 58 registers per thread, an 832-byte
stack frame, and no spills according to `ptxas`. Nsight Compute hardware
counters were unavailable on this GPU, so the Precision 7710 experiment
did not claim an unmeasured microarchitectural cause.

The direct observation was sufficient to explain the performance trend:
GPU kernel cost grew proportionally more than CPU execution time as
component count increased.

## 9. Cross-machine reproduction on Precision 7680

The frozen `v0.7.0` implementation was reproduced without changing the
EOS implementation, CUDA kernel, numerical algorithms, or validation
tolerances on:

-   Dell Precision 7680;
-   Intel Core i9-13950HX;
-   NVIDIA RTX 4090 Laptop GPU;
-   Ada, compute capability 8.9;
-   Windows 11 host;
-   WSL2 / Ubuntu 24.04;
-   GCC 13;
-   CUDA 13.0.

The complete configured test suite passed unchanged:

> **8 / 8 tests passed**

This is a numerical portability result independent of the performance
comparison: the same validated implementation reproduced across
Maxwell/CUDA 12.4/native Linux and Ada/CUDA 13.0/WSL2 without
architecture-specific numerical workarounds.

### 9.1 Resident crossover

Focused resident crossover measurements were:

    Components   Measured bracket   Descriptive interpolation
  ------------ ------------------ ---------------------------
             1           400--600                \~545 states
             2           500--600                \~513 states
             3          500--1200                \~758 states
             5           500--900                \~850 states

These later crossover locations coexist with much larger large-batch
resident speedups on the newer GPU. The experiment therefore
demonstrates that **crossover location and asymptotic throughput are
different characteristics**.

### 9.2 CPU/end-to-end limitation

The i9-13950HX is physically hybrid, but WSL2 exposed 32 logical
processors as 16 homogeneous SMT core pairs. Linux affinity could
constrain WSL-visible logical CPUs but could not establish physical
P-core/E-core placement on the Windows host.

The selected stable CPU baseline also changed non-monotonically with
batch size. Consequently, the raw Precision 7680 CPU and end-to-end
measurements are retained, but precise end-to-end crossover
interpolations are not promoted as robust dispatch thresholds.

The large-batch CUDA-resident advantage is more defensible than a
precise CPU/E2E crossover location on this environment.

## 10. Ada hardware-counter experiment

The Precision 7680 made Nsight Compute hardware counters available for
the component-scaling question.

At a fixed 100,000-state batch size, changing only mixture component
count from one to five produced:

  Metric                      1 component    5 components
  ------------------------- ------------- ---------------
  Kernel duration                \~269 us        \~942 us
  Relative duration                 1.00x         \~3.50x
  Compute (SM) throughput         \~80.6%   \~80.2--80.7%
  DRAM throughput             \~1.9--2.0%          \~1.3%
  Registers/thread                     56              56
  Theoretical occupancy               75%             75%
  Achieved occupancy              \~55.8%         \~56.1%
  Waves/SM                           1.14            1.14

The 1c-to-5c increase therefore occurs without:

-   increasing static register pressure;
-   occupancy collapse;
-   changing launch geometry;
-   DRAM bandwidth saturation; or
-   loss of overall compute utilization.

The evidence supports the simpler explanation that each CUDA thread
performs substantially more component-dependent PR work, including the
quadratic mixture interaction calculation and component fugacity
calculations.

The Ada duration ratio of about `3.50x` is also close to the Maxwell
ratio of about `3.40x`. Because the systems differ in CPU, GPU,
operating environment, CUDA toolkit, and profiler, this is not a
controlled architecture comparison. It nevertheless provides
cross-platform evidence that the scaling behavior is associated with the
workload/kernel structure rather than being peculiar to Maxwell.

The Ada build used 56 registers per thread, with 75% theoretical and
about 56% achieved occupancy. This may identify an absolute-performance
optimization opportunity, but it does **not** explain the
component-count scaling because those quantities remain essentially
unchanged between the controlled one- and five-component cases.

Likewise, both cases used the same 782-block by 128-thread launch
geometry and approximately 1.14 waves per SM. A partial final wave may
affect absolute performance, but cannot explain the difference between
the two component counts.

## 11. Cross-machine interpretation

The two-machine study supports several conclusions.

First, the V1 numerical implementation is portable across the tested
Maxwell/native-Linux/CUDA-12.4 and Ada/WSL2/CUDA-13 environments. The
validation suite passed unchanged.

Second, CUDA provides a clear large-batch resident-compute advantage on
both tested systems.

Third, CPU/GPU crossover is not a property of the algorithm alone. It
depends on CPU performance, GPU performance, launch overhead, data
movement, runtime behavior, and the benchmark workload. Crossover
measurements therefore belong to a particular machine/environment and
should not be hard-coded as universal thresholds.

Fourth, a faster GPU does not imply an earlier crossover. The Precision
7680 combined substantially greater asymptotic CUDA throughput with
later measured resident crossover than the older Precision 7710.

Fifth, component-count scaling is a workload-level effect visible on
both GPU generations. The controlled kernel duration grows by roughly
3.4--3.5x from one to five components. Ada counters exclude several
obvious alternative explanations and support increased per-thread
arithmetic work as the primary cause.

## 12. Engineering conclusions

ThermoGPU V1.0 answers its original research question with a qualified
**yes** for the measured workloads: GPU execution can materially
accelerate sufficiently large batches of single-phase multicomponent
Peng--Robinson calculations while maintaining close numerical agreement
with the externally anchored scalar implementation.

The qualification matters. GPU acceleration is not automatically
advantageous for small batches. Resident and end-to-end crossover differ
substantially because transfer, allocation, and launch costs matter. The
crossover also changes with component count and with the host/GPU
environment.

The numerical work produced an equally important conclusion: backend
differential testing is useful not only as a correctness gate but as a
numerical-analysis tool. The CPU/CUDA discrepancy that exposed the
Cardano cancellation problem was small in its initiating coefficient but
large enough to reveal a formulation weakness. Fixing the mathematics
was preferable to suppressing the symptom through looser tolerances or
altered compiler behavior.

The project therefore supports a broader scientific-software workflow:

1.  define a narrow mathematical workload;
2.  establish a readable scalar implementation;
3.  anchor it through invariants and an external implementation;
4.  implement optimized backends against the same mathematical contract;
5.  investigate discrepancies rather than normalize them away;
6.  separate resident compute performance from end-to-end application
    cost;
7.  measure crossover on the target machine;
8.  profile observed scaling before assigning a microarchitectural
    cause; and
9.  retain raw evidence and environment information alongside
    conclusions.

## 13. Limitations

The V1 results should not be generalized beyond the tested scope.

ThermoGPU V1.0 does not implement flash calculations, phase stability,
multiphase equilibrium, automatic phase selection, or a complete
reservoir/pipeline simulator. Its performance conclusions apply to
batched single-phase PR calculations with the tested component counts
and implementation.

Only two machines were used for the cross-platform study, and they
differ in many variables simultaneously. The comparison is therefore a
reproducibility and portability study, not a controlled GPU-generation
benchmark.

The Precision 7680 CPU/end-to-end results are additionally limited by
WSL2's presentation of the hybrid i9-13950HX topology.

Profiler observations are diagnostic measurements, not promises of
attainable optimization speedup. In particular, register-limited
theoretical occupancy and the partial final launch wave are possible
optimization targets, but neither explains the measured
one-to-five-component scaling.

## 14. Reproducibility and evidence

The repository retains the methodology, raw measurements, profiler
outputs, and analysis needed to inspect the V1 conclusions.

Key documents are:

-   `docs/V1_SPEC.md` --- original V1 mathematical and engineering
    specification;
-   `docs/theory.md` --- PR equations, conventions, and units;
-   `docs/verification.md` --- verification hierarchy and external
    validation;
-   `docs/m6_validation.md` --- CPU/CUDA differential-validation
    campaign;
-   `docs/benchmarking.md` --- benchmark methodology and earlier
    crossover work;
-   `docs/m7_performance.md` --- primary M7 heterogeneous performance
    analysis;
-   `results/m7/CROSS_MACHINE_ANALYSIS.md` --- 7710/7680 synthesis and
    Ada counter analysis;
-   `docs/reproducibility.md` --- build, test, analysis-environment, and
    new-machine reproduction instructions.

Raw M7 evidence is retained under:

-   `results/m7/precision-7710/`;
-   `results/m7/precision-7680/`.

The analysis environment is separate from the production C++ runtime.
`requirements-analysis.txt` records the Python
analysis/external-validation dependencies.

## 15. Final V1.0 result

ThermoGPU V1.0 produced a validated scalar/OpenMP/CUDA implementation of
the single-phase multicomponent Peng--Robinson EOS, demonstrated close
CPU/GPU numerical agreement across a broad deterministic campaign,
identified and corrected a real cubic-solver stability problem, measured
CPU/GPU crossover rather than assuming it, demonstrated substantial
large-batch CUDA throughput advantage on the primary system, reproduced
the validated implementation on a second substantially different
GPU/software environment, and used hardware counters to explain the
dominant component-scaling behavior without overclaiming unsupported
microarchitectural causes.

The resulting answer is deliberately narrower than "GPUs are faster for
EOS." For this implementation and these measured systems, GPU
acceleration becomes valuable once enough independent EOS states are
available to amortize fixed GPU costs. The exact crossover is machine-
and workload-specific, while the numerical and profiling evidence is
reproducible enough to support the engineering conclusions above.
