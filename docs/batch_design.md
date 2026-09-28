# M3 batch design

ThermoGPU M3 adds workload batching without changing Peng-Robinson thermodynamics.

## Input layout

For `S` states and `N` mixture components, `MixtureBatch` uses a structure-of-arrays layout:

- `pressure_Pa[S]`
- `temperature_K[S]`
- `mole_fractions[S*N]`

Composition is flattened state-major. Component `i` of state `s` is stored at `s*N + i`. The component property table and dense `kij` matrix remain invariant in `Mixture`; composition is state data and may vary across the batch.

This layout makes the state dimension explicit and avoids an array of heap-owning `Mixture` objects. It is also the contract intended for M4 OpenMP and M5 CUDA backends.

## Output layout

`MixtureBatchResult` stores:

- `Z[S]`
- `density_kg_per_m3[S]`
- `cubic_residual[S]`
- `root_classification[S]`
- `root_selection_ambiguous[S]`
- `ln_phi[S*N]`, using the same state-major indexing as composition

M3 intentionally exposes only outputs needed by the V1 workload and differential validation; diagnostic per-component intermediates remain available through the single-state diagnostic API.

## Numerical contract

The scalar batch backend is an orchestration layer over the validated single-state kernel. It introduces no new EOS equations. Every state is required to reproduce direct `evaluate_mixture()` results to floating-point roundoff. Batch inputs receive the same physical validation policy as scalar inputs: finite positive pressure/temperature, finite mole fractions in `[0,1]`, and independently normalized composition for every state. No silent normalization is performed.

An empty batch is valid and returns empty result arrays. Malformed array dimensions are rejected.

## M4 OpenMP backend

`evaluate_mixture_batch_openmp()` preserves the M3 batch contract and parallelizes only the independent state dimension:

```cpp
#pragma omp parallel for schedule(static)
for (std::size_t s = 0; s < nstate; ++s) {
    // complete PR evaluation for state s
}
```

Each iteration owns a local `Mixture` copy and writes only the output slots for its state. There is no thermodynamic reduction or synchronization between states. `schedule(static)` is the baseline because states have approximately uniform computational cost. Runtime thread count is controlled with standard OpenMP settings such as `OMP_NUM_THREADS`.

The M4 differential test uses 1,000 deterministic binary-mixture states spanning 0.1--12 MPa, 240--420 K, and methane mole fractions from 0.05--0.95. It compares OpenMP and scalar-batch Z, density, cubic residual, root classification, root ambiguity, and every component `ln_phi`.
