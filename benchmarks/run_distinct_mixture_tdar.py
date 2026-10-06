#!/usr/bin/env python3
"""TDAR accuracy/complexity sweep for genuinely distinct C1..C5 PR mixtures.

All cases use the same TDAR seed, budgets, and frozen Sobol validation design.
This stage intentionally stops before CPWA-ReLU synthesis so approximation
geometry is measured independently of lowering/code-generation effects.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys
from pathlib import Path

VALID_COMPONENTS=(1,2,3,4,5)

def run(cmd, *, cwd=None, env=None):
    print('+', ' '.join(map(str,cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)

def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def num(row,name):
    v=row.get(name); return None if v in (None,'') else float(v)

def integer(row,name):
    v=row.get(name); return None if v in (None,'') else int(float(v))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tdar',type=Path,required=True,help='TDAR checkout')
    ap.add_argument('--oracle',type=Path,default=Path('build/thermogpu_tdar_oracle'))
    ap.add_argument('--components',nargs='+',type=int,default=list(VALID_COMPONENTS))
    ap.add_argument('--budgets',nargs='+',type=int,default=[16,24,32,48,64,96,128])
    ap.add_argument('--validation-points',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20261002)
    ap.add_argument('--output',type=Path,default=Path('benchmarks/results-distinct-mixture-tdar'))
    a=ap.parse_args()
    bad=sorted(set(a.components)-set(VALID_COMPONENTS))
    if bad: ap.error(f'unsupported distinct component counts: {bad}; choose from {VALID_COMPONENTS}')
    root=Path(__file__).resolve().parents[1]
    tdar=a.tdar.resolve()
    oracle=(root/a.oracle).resolve() if not a.oracle.is_absolute() else a.oracle.resolve()
    if not tdar.exists(): ap.error(f'TDAR checkout not found: {tdar}')
    if not oracle.exists(): ap.error(f'oracle not found: {oracle}')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)

    rows=[]; validation_hash=None
    for nc in a.components:
        case=out/f'c{nc}'
        env=os.environ.copy()
        env['THERMOGPU_TDAR_MIXTURE']='distinct'
        env['THERMOGPU_TDAR_COMPONENTS']=str(nc)
        print(f'\n===== distinct C{nc}: TDAR accuracy sweep =====',flush=True)
        run([sys.executable,'benchmarks/run_thermogpu_tradeoff.py',
             '--oracle',oracle,'--budgets',*map(str,a.budgets),
             '--seed',a.seed,'--validation-points',a.validation_points,
             '--output',case],cwd=tdar,env=env)
        h=digest(case/'validation-points.npy')
        if validation_hash is None: validation_hash=h
        elif h != validation_hash:
            raise RuntimeError(f'validation design mismatch for C{nc}: {h} != {validation_hash}')
        for r in read_csv(case/'summary.csv'):
            rows.append({
                'family':'distinct','components':nc,'budget':int(r['budget']),
                'points':integer(r,'points'),'simplices':integer(r,'simplices'),
                'tdar_seconds':num(r,'tdar_seconds'),'rmse':num(r,'rmse'),
                'p99_abs':num(r,'p99_abs'),'max_abs':num(r,'max_abs'),
                'validation_sha256':h,
            })

    fields=list(rows[0])
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    metadata={'family':'distinct','components':a.components,'budgets':a.budgets,
              'validation_points':a.validation_points,'seed':a.seed,
              'validation_sha256':validation_hash,
              'domain':{'temperature_K':[250.0,350.0],'pressure_Pa':[1.0e6,1.0e7]}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Distinct-mixture TDAR accuracy/complexity','',
           'C1..C5 are nested prefixes of (0.80, 0.08, 0.04, 0.04, 0.04), renormalized at each prefix.',
           f'All cases share validation design SHA-256 `{validation_hash}`.','',
           '| C | budget | points | simplices | RMSE | p99 abs | max abs | TDAR s |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return '' if x is None else f'{x:.6g}'
    for r in rows:
        lines.append(f"| {r['components']} | {r['budget']} | {r['points']} | {r['simplices']} | {fmt(r['rmse'])} | {fmt(r['p99_abs'])} | {fmt(r['max_abs'])} | {fmt(r['tdar_seconds'])} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'\nwrote: {out/"summary.csv"}')
    print(f'wrote: {out/"report.md"}')

if __name__=='__main__': main()
