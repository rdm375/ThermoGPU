#!/usr/bin/env python3
"""Complete CPWA hardware sweep for the frozen distinct-mixture TDAR study.

This is a hardware-characterization experiment only.  It does not rerun TDAR
or select artifacts according to an accuracy threshold.

For every requested (component count, TDAR budget), it:

  1. locates the already-computed TDAR artifact,
  2. constructs the exact native CPWA DAG,
  3. performs bottom-up synthesis,
  4. performs CP-SAT min-circuit optimization,
  5. benchmarks the min-live fused-CUDA schedule at every requested batch size.

The resulting table is suitable for constructing batch-conditioned
accuracy/performance Pareto frontiers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_COMPONENTS = [1, 2, 3, 4, 5]
DEFAULT_BUDGETS = [16, 24, 32, 48, 64, 96, 128]
DEFAULT_BATCHES = [1, 10, 100, 1000, 10000, 100000, 1000000]


def run(cmd, *, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
    )


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def fv(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def iv(row, key):
    value = row.get(key)
    return None if value in (None, "") else int(float(value))


def locate_source(tdar, results, components, budget):
    rows = read_csv(results / f"c{components}" / "summary.csv")

    hit = next(
        (r for r in rows if int(r["budget"]) == budget),
        None,
    )
    if hit is None:
        raise RuntimeError(
            f"C{components} B{budget}: budget absent from TDAR summary"
        )

    raw = hit.get("artifact")
    if not raw:
        raise RuntimeError(
            f"C{components} B{budget}: summary has no artifact column"
        )

    path = Path(raw)
    if not path.is_absolute():
        path = (tdar / path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"C{components} B{budget}: TDAR artifact not found: {path}"
        )

    return path, hit


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tdar", type=Path, required=True)
    ap.add_argument("--cpwa-relu", type=Path, required=True)

    ap.add_argument(
        "--tdar-results",
        type=Path,
        default=Path("benchmarks/results-distinct-mixture-tdar"),
    )

    ap.add_argument(
        "--components",
        nargs="+",
        type=int,
        default=DEFAULT_COMPONENTS,
    )
    ap.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    ap.add_argument(
        "--batches",
        nargs="+",
        type=int,
        default=DEFAULT_BATCHES,
    )

    ap.add_argument("--cpsat-time-limit", type=float, default=60.0)
    ap.add_argument("--cpsat-workers", type=int, default=8)
    ap.add_argument("--arch")

    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/results-distinct-mixture-hardware-sweep"
        ),
    )

    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    tdar = a.tdar.resolve()
    cpwa = a.cpwa_relu.resolve()

    results = (
        (root / a.tdar_results).resolve()
        if not a.tdar_results.is_absolute()
        else a.tdar_results.resolve()
    )

    for path, label in [
        (tdar, "TDAR"),
        (cpwa, "cpwa-relu"),
        (results, "TDAR results"),
    ]:
        if not path.exists():
            ap.error(f"{label} not found: {path}")

    invalid_components = sorted(
        set(a.components) - set(DEFAULT_COMPONENTS)
    )
    if invalid_components:
        ap.error(
            f"components must be in 1..5: {invalid_components}"
        )

    if any(b <= 0 for b in a.budgets):
        ap.error("budgets must be positive")

    if any(b <= 0 for b in a.batches):
        ap.error("batches must be positive")

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    cenv = os.environ.copy()
    cenv["PYTHONPATH"] = (
        str(cpwa / "src")
        + os.pathsep
        + cenv.get("PYTHONPATH", "")
    )

    rows = []

    for nc in a.components:
        for budget in a.budgets:
            source, tdar_row = locate_source(
                tdar, results, nc, budget
            )

            case = out / f"c{nc}-b{budget}"
            case.mkdir(parents=True, exist_ok=True)

            native = case / f"c{nc}-b{budget}-native.cpwa"
            bottom = case / f"c{nc}-b{budget}-bottom-up.cpwa"
            final = case / f"c{nc}-b{budget}-cpsat.cpwa"
            cuda = case / "cuda"

            print(
                f"\n===== C{nc} B{budget}: "
                "exact CPWA synthesis =====",
                flush=True,
            )

            run(
                [
                    sys.executable,
                    "benchmarks/build_native_dag_artifact.py",
                    "--source",
                    source,
                    "--output",
                    native,
                ],
                cwd=cpwa,
                env=cenv,
            )

            run(
                [
                    sys.executable,
                    "benchmarks/synthesize_bottom_up.py",
                    "--artifact",
                    native,
                    "--check-points",
                    "0",
                    "--output",
                    bottom,
                ],
                cwd=cpwa,
                env=cenv,
            )

            run(
                [
                    sys.executable,
                    "benchmarks/optimize_min_circuit_cpsat.py",
                    "--artifact",
                    bottom,
                    "--intersections",
                    "0",
                    "--time-limit",
                    a.cpsat_time_limit,
                    "--workers",
                    a.cpsat_workers,
                    "--output",
                    final,
                ],
                cwd=cpwa,
                env=cenv,
            )

            cmd = [
                sys.executable,
                "benchmarks/benchmark_register_schedules.py",
                final,
                "--schedules",
                "min-live",
                "--batches",
                *map(str, a.batches),
                "--output",
                cuda,
            ]

            if a.arch:
                cmd.extend(["--arch", a.arch])

            run(cmd, cwd=cpwa, env=cenv)

            bench_rows = read_csv(cuda / "summary.csv")

            observed_batches = {
                int(r["batch"]) for r in bench_rows
            }
            expected_batches = set(a.batches)

            if observed_batches != expected_batches:
                raise RuntimeError(
                    f"C{nc} B{budget}: batch mismatch: "
                    f"expected {sorted(expected_batches)}, "
                    f"got {sorted(observed_batches)}"
                )

            for r in bench_rows:
                rows.append(
                    {
                        "components": nc,
                        "budget": budget,
                        "rmse": fv(tdar_row, "rmse"),
                        "p99_abs": fv(tdar_row, "p99_abs"),
                        "max_abs": fv(tdar_row, "max_abs"),
                        "points": iv(tdar_row, "points"),
                        "simplices": iv(tdar_row, "simplices"),
                        "tdar_seconds": fv(
                            tdar_row, "tdar_seconds"
                        ),
                        "batch": iv(r, "batch"),
                        "ns_per_eval": fv(r, "ns_per_eval"),
                        "dag_nodes": iv(r, "nodes"),
                        "affine_nodes": iv(r, "affine_nodes"),
                        "min_nodes": iv(r, "min_nodes"),
                        "max_nodes": iv(r, "max_nodes"),
                        "executed_ops": iv(r, "executed_ops"),
                        "live_slots": iv(r, "live_slots"),
                        "registers": iv(r, "registers"),
                        "spill_store_bytes": iv(
                            r, "spill_store_bytes"
                        ),
                        "spill_load_bytes": iv(
                            r, "spill_load_bytes"
                        ),
                        "stack_frame_bytes": iv(
                            r, "stack_frame_bytes"
                        ),
                        "max_delta_cpu": fv(
                            r, "max_delta_cpu"
                        ),
                        "tdar_artifact": str(source),
                        "cpwa_artifact": str(final),
                    }
                )

    rows.sort(
        key=lambda r: (
            r["components"],
            r["budget"],
            r["batch"],
        )
    )

    expected_count = (
        len(a.components)
        * len(a.budgets)
        * len(a.batches)
    )

    if len(rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} rows, got {len(rows)}"
        )

    fields = [
        "components",
        "budget",
        "rmse",
        "p99_abs",
        "max_abs",
        "points",
        "simplices",
        "tdar_seconds",
        "batch",
        "ns_per_eval",
        "dag_nodes",
        "affine_nodes",
        "min_nodes",
        "max_nodes",
        "executed_ops",
        "live_slots",
        "registers",
        "spill_store_bytes",
        "spill_load_bytes",
        "stack_frame_bytes",
        "max_delta_cpu",
        "tdar_artifact",
        "cpwa_artifact",
    ]

    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    (out / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    metadata = {
        "experiment":
            "distinct-mixture complete CPWA hardware sweep",
        "components": a.components,
        "budgets": a.budgets,
        "batches": a.batches,
        "cpsat_time_limit": a.cpsat_time_limit,
        "cpsat_workers": a.cpsat_workers,
        "arch": a.arch,
        "tdar_results": str(results),
        "tdar_rerun": False,
        "selection":
            "all requested measured TDAR budgets; "
            "no accuracy-threshold selection",
        "expected_rows": expected_count,
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print()
    print(f"wrote: {out / 'summary.csv'}")
    print(f"wrote: {out / 'summary.json'}")
    print(f"wrote: {out / 'metadata.json'}")
    print(f"rows:  {len(rows)}")


if __name__ == "__main__":
    main()
