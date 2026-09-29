# M7 Cross-Machine Performance Analysis

## Purpose

ThermoGPU M7 characterized the performance of the scalar, OpenMP, and CUDA
Peng-Robinson EOS implementations and investigated the causes of the observed
CPU/GPU scaling behavior.

The original M7 campaign was performed on a Dell Precision 7710. The frozen
ThermoGPU `v0.7.0` implementation was subsequently reproduced on a Dell
Precision 7680 without changing the EOS implementation, CUDA kernel, numerical
algorithms, or validation tolerances.

The second system was not intended to be a hardware-only controlled comparison.
It provides a portability and reproducibility experiment across substantially
different CPU, GPU, CUDA, and operating environments.

## Systems

### Precision 7710

- Dell Precision 7710
- Intel Core i7-6920HQ
- 4 physical cores / 8 hardware threads
- NVIDIA Quadro M3000M
- Maxwell GPU, compute capability 5.2
- native Linux
- GCC 13
- CUDA 12.4
- OpenMP 4.5

### Precision 7680

- Dell Precision 7680
- Intel Core i9-13950HX
- NVIDIA RTX 4090 Laptop GPU
- Ada GPU, compute capability 8.9
- 76 SMs
- Windows 11 host
- WSL2 / Ubuntu 24.04
- GCC 13
- CUDA 13.0
- OpenMP 4.5

The 13950HX is physically a hybrid CPU, but WSL2 exposes 32 logical processors
as 16 homogeneous SMT core pairs. The Linux environment therefore does not
provide a reliable mapping from WSL CPU IDs to the processor's physical
performance-core and efficiency-core topology.

For this reason, CPU performance results from the Precision 7680 require
additional qualification.

## Reproducibility

The Precision 7680 used the same frozen ThermoGPU `v0.7.0` source tree used for
the completed M7 implementation.

The project configured and built successfully using GCC 13 and CUDA 13.0 with
CUDA architecture 8.9 enabled.

The complete test suite passed unchanged:

    8 / 8 tests passed

This included scalar, scalar-batch, OpenMP, CUDA-batch, CUDA differential,
external-reference, controlled ThermoPack, and frozen-reference tests.

This is important independently of the performance measurements. The same
numerically validated implementation reproduced successfully across Maxwell
and Ada GPUs and across CUDA 12.4 and CUDA 13.0 without changing numerical
tolerances or introducing architecture-specific numerical workarounds.

## Precision 7710 performance results

The original Precision 7710 M7 campaign established scalar, OpenMP, CUDA
resident, and CUDA end-to-end performance over component counts of 1, 2, 3,
and 5.

Focused CUDA-resident crossover estimates were approximately:

| Components | Resident crossover |
|---:|---:|
| 1 | 191 states |
| 2 | 171 states |
| 3 | 328 states |
| 5 | 339 states |

Focused CUDA end-to-end crossover estimates were approximately:

| Components | End-to-end crossover |
|---:|---:|
| 1 | 5,004 states |
| 2 | 4,414 states |
| 3 | 3,989 states |
| 5 | 3,402 states |

These crossover values are properties of the measured machine and benchmark
environment, not universal dispatch thresholds.

At five components and one million states, the Precision 7710 measured:

- scalar: approximately 533.6 ns/state;
- OpenMP-8: approximately 173.8 ns/state;
- CUDA resident: approximately 29.3 ns/state;
- CUDA end-to-end: approximately 96.8 ns/state.

The resident CUDA implementation therefore achieved a large asymptotic
throughput advantage on the older Maxwell system.

## Precision 7680 performance results

The Precision 7680 also showed a strong large-batch CUDA-resident throughput
advantage, but its crossover behavior differed substantially from the older
system.

Focused resident crossover brackets and descriptive local interpolations were:

| Components | Measured bracket | Local interpolation |
|---:|---:|---:|
| 1 | 400-600 | ~545 states |
| 2 | 500-600 | ~513 states |
| 3 | 500-1200 | ~758 states |
| 5 | 500-900 | ~850 states |

The later resident crossover does not contradict the much greater asymptotic
throughput of the RTX 4090 Laptop GPU. Crossover location and asymptotic
throughput measure different properties. A much faster GPU can still require
a larger workload before its fixed launch and execution overheads are
amortized relative to a much faster host CPU.

The focused campaign measured maximum CUDA-resident speedups relative to the
selected stable CPU baseline in the approximate range 20x to 61x. These
figures must be interpreted with the CPU-baseline qualification discussed
below and should not be treated as universal speedup claims.

## Precision 7680 CPU baseline limitation

The Precision 7680 CPU measurements displayed substantial non-monotonic
changes in the fastest stable OpenMP baseline as batch size changed.

Individual benchmark points could have low median absolute deviation while
the selected CPU baseline still changed substantially between neighboring
batch sizes. Low within-point variation therefore did not establish that all
points were measured under equivalent physical CPU scheduling and operating
conditions.

This matters because the i9-13950HX has a hybrid physical topology while WSL2
presents a flattened homogeneous SMT topology.

Linux `taskset` successfully constrained the WSL-visible logical CPUs and the
EOS computation remained deterministic, but this could not establish whether
the Windows host placed those virtual CPUs on particular physical P-cores,
E-cores, or SMT contexts.

Consequently, the Precision 7680 CPU and CUDA end-to-end measurements provide
useful performance observations but do not support a precise universal
CPU/GPU dispatch crossover.

In particular, focused end-to-end crossover interpolations should not be
presented as robust dispatch thresholds.

The large-batch CUDA-resident advantage is substantially clearer than the
precise CPU/E2E crossover location.

## Component-count scaling

A central M7 observation on the Precision 7710 was that increasing mixture
complexity affected CPU and GPU execution differently.

For the scalar CPU implementation, increasing from one to five components
approximately doubled retired instruction count while increasing measured
cycle count by only about 1.7x. CPU execution efficiency therefore improved
as additional regular arithmetic work was introduced.

For CUDA, Nsight Systems measurements on the M3000M showed approximately:

    1 component: ~1.11 ms
    5 components: ~3.76 ms

or approximately a 3.4x increase in kernel duration.

Hardware performance counters were unavailable on the M3000M, so the original
7710 experiment could measure the scaling but could not directly determine
its GPU microarchitectural cause.

## Ada hardware-counter experiment

The Precision 7680 allowed the same question to be investigated using Nsight
Compute hardware performance counters.

A controlled experiment profiled the same `pr_batch_kernel` at the same
100,000-state batch size while changing only mixture component count from one
to five.

Representative results were:

| Metric | 1 component | 5 components |
|---|---:|---:|
| Kernel duration | ~269 us | ~942 us |
| Relative duration | 1.00x | ~3.50x |
| Compute (SM) throughput | ~80.6% | ~80.2-80.7% |
| DRAM throughput | ~1.9-2.0% | ~1.3% |
| Registers/thread | 56 | 56 |
| Theoretical occupancy | 75% | 75% |
| Achieved occupancy | ~55.8% | ~56.1% |
| Waves/SM | 1.14 | 1.14 |

The measurements were highly repeatable across the profiled launches.

Increasing component count therefore increased kernel duration by
approximately 3.5x without increasing register count, reducing theoretical
occupancy, materially reducing achieved occupancy, changing launch geometry,
or approaching DRAM bandwidth saturation.

Compute utilization remained near 80% in both cases.

## Interpretation of GPU component scaling

The Ada measurements rule out several candidate explanations for the
one-component to five-component scaling behavior.

The increase is not explained by:

- increasing static register pressure with component count;
- occupancy collapse;
- changing launch geometry;
- DRAM bandwidth saturation; or
- loss of overall compute utilization.

Instead, the evidence supports a simpler explanation: each CUDA thread has
substantially more computational work to perform as mixture component count
increases.

Peng-Robinson mixture evaluation contains component-dependent work including
the double sum used to construct the mixture attraction parameter and
additional component-dependent fugacity calculations.

The approximately 3.5x Ada kernel-duration increase is also close to the
approximately 3.4x increase measured previously on Maxwell.

Because the machines and toolchains differ, this agreement is not a
controlled architectural comparison. It nevertheless provides evidence that
the component-count scaling is associated with the workload and kernel
structure rather than being peculiar to the older Maxwell GPU.

## Register and occupancy observation

Nsight Compute reports 56 registers per thread for the Ada build. This limits
theoretical occupancy to 75%, while achieved occupancy is approximately 56%.

This identifies a possible absolute-performance optimization opportunity, but
it does not explain the component-count scaling: register count and occupancy
are essentially unchanged between the one-component and five-component
experiments.

Accordingly, reducing register usage should not be presented as the
established solution to the observed 1c-to-5c scaling behavior.

## Launch-tail observation

At 100,000 states, both controlled Ada experiments launch 782 blocks of
128 threads and produce approximately 1.14 waves per SM.

Nsight Compute identifies the resulting partial wave as a possible
optimization opportunity.

This may affect absolute performance at this batch size, but it cannot explain
the difference between the one-component and five-component cases because
their launch geometry is identical.

Profiler optimization estimates are diagnostic guidance and are not treated
as measured achievable speedups.

## Cross-machine conclusions

The cross-machine experiment supports the following conclusions.

1. ThermoGPU `v0.7.0` is numerically portable across the tested Maxwell/CUDA
   12.4/native-Linux and Ada/CUDA-13/WSL2 environments. The complete validation
   suite passed unchanged.

2. CUDA provides a clear large-batch resident-compute advantage on both tested
   GPU generations.

3. CPU/GPU crossover values are machine- and environment-specific. They should
   not be hard-coded as universal dispatch thresholds.

4. The Precision 7680 demonstrates particularly clearly that high asymptotic
   GPU throughput and an early GPU crossover are not the same property.

5. Precise CPU and CUDA end-to-end crossover characterization on the Precision
   7680 is complicated by the hybrid host CPU and WSL2's flattened CPU topology.
   The measured data are retained, but precise end-to-end crossover
   interpolations are not promoted as robust dispatch recommendations.

6. Increasing mixture component count from one to five increases CUDA kernel
   duration by approximately 3.4-3.5x on both tested GPU generations.

7. Ada hardware counters show that this component-count penalty occurs while
   register count, occupancy, launch geometry, and compute utilization remain
   essentially unchanged and while DRAM utilization remains very low.

8. The observed GPU component scaling is therefore best explained by increased
   per-state computational work rather than memory-bandwidth saturation,
   occupancy collapse, increasing register pressure, or failure to utilize the
   GPU.

## Scope and limitations

These results characterize the ThermoGPU V1 Peng-Robinson workload on two
specific systems.

They do not establish universal CPU/GPU crossover thresholds, universal
speedups, or general performance characteristics for all EOS workloads.

The two machines differ in CPU architecture, GPU architecture, operating
environment, CUDA toolkit, profiler version, and other system characteristics.
The cross-machine comparison should therefore be interpreted as a
reproducibility and portability study rather than a controlled hardware-only
benchmark.
