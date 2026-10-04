#!/usr/bin/env python3
"""End-to-end fixed-composition PR -> TDAR -> CPWA/CUDA viability sweep.

This driver deliberately keeps the surrogate input dimension fixed at (T,P) while
varying the number of components in the direct Peng-Robinson source calculation.
It delegates TDAR sampling and cpwa-relu synthesis/codegen to sibling checkouts so
that each project remains the authority for its own artifacts.
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

VALID_COMPONENTS = (1, 2, 4, 8, 16)

def run(cmd, *, cwd=None, env=None, capture=False):
    print('+', ' '.join(map(str, cmd)), flush=True)
    return subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True,
                          text=True, capture_output=capture)

def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def fget(row, *names, default=None):
    for n in names:
        if n in row and row[n] not in ('', None): return float(row[n])
    return default

def iget(row, *names, default=None):
    v=fget(row,*names,default=default); return None if v is None else int(v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tdar', type=Path, required=True, help='TDAR checkout')
    ap.add_argument('--cpwa-relu', type=Path, required=True, help='cpwa-relu checkout')
    ap.add_argument('--oracle', type=Path, default=Path('build/thermogpu_tdar_oracle'))
    ap.add_argument('--direct-benchmark', type=Path, default=Path('build/thermogpu_fixed_composition_benchmark'))
    ap.add_argument('--components', nargs='+', type=int, default=list(VALID_COMPONENTS))
    ap.add_argument('--budgets', nargs='+', type=int, default=[16,24,32,48,64,96,128])
    ap.add_argument('--validation-points', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=100000)
    ap.add_argument('--repeats', type=int, default=9)
    ap.add_argument('--cpsat-time-limit', type=float, default=60.0)
    ap.add_argument('--cpsat-workers', type=int, default=8)
    ap.add_argument('--output', type=Path, default=Path('benchmarks/results-pr-mixture-viability'))
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    tdar=a.tdar.resolve(); cpwa=a.cpwa_relu.resolve()
    oracle=(root/a.oracle).resolve() if not a.oracle.is_absolute() else a.oracle.resolve()
    direct=(root/a.direct_benchmark).resolve() if not a.direct_benchmark.is_absolute() else a.direct_benchmark.resolve()
    for p,label in [(tdar,'TDAR'),(cpwa,'cpwa-relu'),(oracle,'oracle'),(direct,'direct benchmark')]:
        if not p.exists(): ap.error(f'{label} not found: {p}')
    bad=sorted(set(a.components)-set(VALID_COMPONENTS))
    if bad: ap.error(f'unsupported component counts: {bad}; choose from {VALID_COMPONENTS}')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)

    # Direct generic/fixed CUDA baselines under one common harness.
    q=run([direct,a.batch,a.repeats],cwd=root,capture=True)
    (out/'direct-benchmark.csv').write_text(q.stdout)
    direct_rows=list(csv.DictReader(q.stdout.splitlines()))
    direct_by_n={int(r['components']):r for r in direct_rows}

    rows=[]
    for nc in a.components:
        case=out/f'c{nc}'; case.mkdir(exist_ok=True)
        env=os.environ.copy(); env['THERMOGPU_TDAR_COMPONENTS']=str(nc)
        print(f'\n===== C{nc}: TDAR accuracy sweep =====',flush=True)
        run([sys.executable,'benchmarks/run_thermogpu_tradeoff.py',
             '--oracle',oracle,'--budgets',*map(str,a.budgets),
             '--validation-points',a.validation_points,'--output',case/'tdar'],cwd=tdar,env=env)
        trows=read_csv(case/'tdar'/'summary.csv')
        for tr in trows:
            budget=int(tr['budget']); src=Path(tr['artifact'])
            if not src.is_absolute(): src=(tdar/src).resolve()
            synth=case/'artifacts'; synth.mkdir(exist_ok=True)
            native=synth/f'c{nc}-b{budget}-native.cpwa'
            bottom=synth/f'c{nc}-b{budget}-bottom-up.cpwa'
            final=synth/f'c{nc}-b{budget}-cpsat.cpwa'
            print(f'\n===== C{nc} B{budget}: exact synthesis =====',flush=True)
            cenv=os.environ.copy(); cenv['PYTHONPATH']=str(cpwa/'src')+os.pathsep+cenv.get('PYTHONPATH','')
            run([sys.executable,'benchmarks/build_native_dag_artifact.py','--source',src,'--output',native],cwd=cpwa,env=cenv)
            run([sys.executable,'benchmarks/synthesize_bottom_up.py','--artifact',native,'--check-points','0','--output',bottom],cwd=cpwa,env=cenv)
            run([sys.executable,'benchmarks/optimize_min_circuit_cpsat.py','--artifact',bottom,
                 '--intersections','0','--time-limit',a.cpsat_time_limit,'--workers',a.cpsat_workers,'--output',final],cwd=cpwa,env=cenv)
            timing=case/f'timing-b{budget}'
            run([sys.executable,'benchmarks/benchmark_register_schedules.py',final,
                 '--schedules','min-live','--batches',a.batch,'--output',timing],cwd=cpwa,env=cenv)
            rr=read_csv(timing/'summary.csv')[0]
            dr=direct_by_n[nc]
            cpwa_ns=fget(rr,'ns_per_eval','median_ns_per_eval')
            if cpwa_ns is None:
                # Some versions emit one row per batch with a more explicit field.
                cpwa_ns=fget(rr,'cuda_ns_per_eval','time_ns_per_eval')
            fixed=float(dr['fixed_ns_per_state']); generic=float(dr['generic_ns_per_state'])
            row={
                'components':nc,'budget':budget,
                'points':iget(tr,'points'),'simplices':iget(tr,'simplices'),
                'rmse':fget(tr,'rmse'),'p99_abs':fget(tr,'p99_abs'),'max_abs':fget(tr,'max_abs'),
                'direct_generic_ns_per_eval':generic,'direct_fixed_ns_per_eval':fixed,
                'cpwa_ns_per_eval':cpwa_ns,
                'speedup_vs_generic':generic/cpwa_ns if cpwa_ns else None,
                'speedup_vs_fixed':fixed/cpwa_ns if cpwa_ns else None,
                'cpwa_registers':iget(rr,'registers','regs'),
                'cpwa_spill_stores':iget(rr,'spill_stores','spill_store_bytes'),
                'cpwa_spill_loads':iget(rr,'spill_loads','spill_load_bytes'),
                'cpwa_stack':iget(rr,'stack','stack_bytes'),
                'cpwa_live_slots':iget(rr,'live_slots'),
                'cpwa_executed_nodes':iget(rr,'executed_nodes','nodes'),
                'artifact':str(final),
            }
            rows.append(row)

    fields=list(rows[0])
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Compact markdown report; CSV remains the canonical machine-readable result.
    lines=['# PR mixture surrogate viability','',
           f'Batch: {a.batch}; validation points: {a.validation_points}.','',
           '| C | B | RMSE | P99 | Max | direct fixed ns | CPWA ns | speedup |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def fmt(x): return '' if x is None else f'{x:.6g}'
        lines.append(f"| {r['components']} | {r['budget']} | {fmt(r['rmse'])} | {fmt(r['p99_abs'])} | {fmt(r['max_abs'])} | {fmt(r['direct_fixed_ns_per_eval'])} | {fmt(r['cpwa_ns_per_eval'])} | {fmt(r['speedup_vs_fixed'])}x |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'\nwrote: {out/"summary.csv"}')
    print(f'wrote: {out/"report.md"}')

if __name__=='__main__': main()
