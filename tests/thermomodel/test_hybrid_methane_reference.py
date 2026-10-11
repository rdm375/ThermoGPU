#!/usr/bin/env python3
"""Compare analytic ThermoModel outputs against frozen external data."""
import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/reference/data/methane_pr_hybrid.csv"
EXE = Path(sys.argv[1]).resolve()

FIELDS = [
    "cp_J_kgK",
    "cv_J_kgK",
    "sound_speed_m_s",
    "drho_dp_T",
    "drho_dT_p",
    "dh_dp_T",
    "ds_dp_T",
]

# Initial numerical criteria. Do not loosen after a failure
# without investigating and documenting its cause.
TOL = {
    "cp_J_kgK": (5e-10, 1e-7),
    "cv_J_kgK": (5e-10, 1e-7),
    "sound_speed_m_s": (5e-10, 1e-7),
    "drho_dp_T": (5e-9, 1e-12),
    "drho_dT_p": (5e-9, 1e-10),
    "dh_dp_T": (5e-9, 1e-10),
    "ds_dp_T": (5e-9, 1e-12),
}

maximum = {field: (0.0, None) for field in FIELDS}
failures = []
count = 0

with REFERENCE.open(newline="") as f:
    for row in csv.DictReader(f):
        T, p = row["T_K"], row["p_Pa"]
        output = subprocess.check_output(
            [str(EXE), T, p], text=True
        ).strip()
        actual = [float(x) for x in output.split(",")]

        if len(actual) != len(FIELDS):
            raise RuntimeError(
                f"Unexpected probe output at T={T}, p={p}"
            )

        for field, value in zip(FIELDS, actual):
            expected = float(row[field])
            error = abs(value - expected)
            rtol, atol = TOL[field]

            if error > maximum[field][0]:
                maximum[field] = (error, (T, p))

            if not math.isfinite(value) or not math.isclose(
                value, expected, rel_tol=rtol, abs_tol=atol
            ):
                failures.append(
                    (T, p, field, value, expected, error)
                )
            count += 1

print(f"Hybrid external reference: {count} comparisons")
print("Maximum absolute discrepancies:")

for field in FIELDS:
    error, location = maximum[field]
    print(f"  {field:18s} {error:.12e} at {location}")

for T, p, field, actual, expected, error in failures[:20]:
    print(
        f"FAIL T={T} p={p} {field}: "
        f"actual={actual:.17g}, "
        f"reference={expected:.17g}, "
        f"abs_error={error:.12e}"
    )

if failures:
    raise SystemExit(
        f"FAIL: {len(failures)} of {count} comparisons"
    )

print(f"PASS: {count} comparisons")
