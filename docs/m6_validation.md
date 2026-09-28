# M6 CPU/CUDA differential validation

M6 asks whether the CUDA Peng--Robinson backend reproduces the already externally validated scalar thermodynamics throughout the intended V1 domain, including difficult cubic-root regimes.

## Campaigns

The validator uses deterministic inputs. The broad campaign samples 20,000 CH4/C2H6 states over 0.01--20 MPa, 180--500 K, and mole fractions 0.0001--0.9999. A structured grid explicitly includes both component critical temperatures and pressures, low-temperature/high-pressure combinations, and near-pure through mixed compositions.

The report counts one-real and three-real-root scalar states. A zero three-real count means the stress campaign did not exercise the intended branch and should be improved rather than silently accepted.

## Quantities and diagnostics

Every state compares compressibility `Z`, density, cubic residual, both `ln(phi_i)` values, root classification, and root-selection ambiguity. Floating outputs report maximum absolute, relative, and ULP differences plus the complete state associated with the maximum relative discrepancy.

## Numerical policy

An unexplained CPU/GPU difference is investigated, not normalized away by relaxing tolerances. ULP distance is diagnostic only, especially near zero. `Z`, density, and `ln(phi_i)` use combined absolute/relative acceptance tests; cubic residual uses an absolute test; root metadata must agree exactly.
