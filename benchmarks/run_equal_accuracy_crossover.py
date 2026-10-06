#!/usr/bin/env python3
"""Equal-accuracy TDAR -> exact CPWA -> min-live fused-CUDA crossover.

Consumes a completed distinct-mixture TDAR sweep. For each C1..C5 and RMSE
threshold, selects the smallest *measured* TDAR budget satisfying the threshold,
synthesizes each selected artifact once, benchmarks its fused CUDA DAG, and
compares it with direct fixed-composition CUDA PR for the same distinct mixture.
"""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path

def run(cmd,*,cwd=None,env=None,capture=False):
    print('+',' '.join(map(str,cmd)),flush=True)
    return subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,check=True,text=True,capture_output=capture)

def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def fv(r,k):
    v=r.get(k);return None if v in (None,'') else float(v)
def iv(r,k):
    v=r.get(k);return None if v in (None,'') else int(float(v))
def locate_source(tdar,results,nc,budget):
    rows=read_csv(results/f'c{nc}'/'summary.csv')
    row=next(r for r in rows if int(r['budget'])==budget)
    raw=row.get('artifact')
    if not raw: raise RuntimeError(f'C{nc} B{budget}: individual summary has no artifact column')
    p=Path(raw)
    if not p.is_absolute(): p=(tdar/p).resolve()
    if not p.exists(): raise FileNotFoundError(f'C{nc} B{budget}: TDAR artifact not found: {p}')
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tdar',type=Path,required=True)
    ap.add_argument('--cpwa-relu',type=Path,required=True)
    ap.add_argument('--tdar-results',type=Path,default=Path('benchmarks/results-distinct-mixture-tdar'))
    ap.add_argument('--direct-benchmark',type=Path,default=Path('build/thermogpu_distinct_composition_benchmark'))
    ap.add_argument('--components',nargs='+',type=int,default=[1,2,3,4,5])
    ap.add_argument('--rmse-targets',nargs='+',type=float,default=[1e-3,7.5e-4,5e-4,4e-4])
    ap.add_argument('--batches',nargs='+',type=int,default=[1,1000,100000])
    ap.add_argument('--repeats',type=int,default=9)
    ap.add_argument('--cpsat-time-limit',type=float,default=60.0)
    ap.add_argument('--cpsat-workers',type=int,default=8)
    ap.add_argument('--arch')
    ap.add_argument('--output',type=Path,default=Path('benchmarks/results-equal-accuracy-crossover'))
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];tdar=a.tdar.resolve();cpwa=a.cpwa_relu.resolve()
    results=(root/a.tdar_results).resolve() if not a.tdar_results.is_absolute() else a.tdar_results.resolve()
    direct=(root/a.direct_benchmark).resolve() if not a.direct_benchmark.is_absolute() else a.direct_benchmark.resolve()
    for p,label in [(tdar,'TDAR'),(cpwa,'cpwa-relu'),(results,'TDAR results'),(direct,'distinct direct benchmark')]:
        if not p.exists():ap.error(f'{label} not found: {p}')
    if sorted(set(a.components)-set(range(1,6))):ap.error('components must be in 1..5')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source_rows=read_csv(results/'summary.csv')
    # Select smallest actually measured budget meeting each threshold.
    selections=[]
    for nc in a.components:
        cr=sorted((r for r in source_rows if int(r['components'])==nc),key=lambda r:int(r['budget']))
        for target in a.rmse_targets:
            hit=next((r for r in cr if float(r['rmse'])<=target),None)
            selections.append(dict(components=nc,rmse_target=target,status='SELECTED' if hit else 'UNATTAINED',tdar=hit))
    # Direct PR baseline at each requested batch. The executable emits all C1..C5 per invocation.
    direct_by={}
    for batch in a.batches:
        q=run([direct,batch,a.repeats],cwd=root,capture=True)
        (out/f'direct-benchmark-b{batch}.csv').write_text(q.stdout)
        for r in csv.DictReader(q.stdout.splitlines()): direct_by[(int(r['components']),batch)]=r
    # Synthesize/benchmark each unique selected C,budget once.
    unique=sorted({(s['components'],int(s['tdar']['budget'])) for s in selections if s['tdar']})
    bench={}
    cenv=os.environ.copy();cenv['PYTHONPATH']=str(cpwa/'src')+os.pathsep+cenv.get('PYTHONPATH','')
    for nc,budget in unique:
        src=locate_source(tdar,results,nc,budget); case=out/f'c{nc}-b{budget}';case.mkdir(exist_ok=True)
        native=case/f'c{nc}-b{budget}-native.cpwa';bottom=case/f'c{nc}-b{budget}-bottom-up.cpwa';final=case/f'c{nc}-b{budget}-cpsat.cpwa'
        print(f'\n===== C{nc} B{budget}: exact CPWA synthesis =====',flush=True)
        run([sys.executable,'benchmarks/build_native_dag_artifact.py','--source',src,'--output',native],cwd=cpwa,env=cenv)
        run([sys.executable,'benchmarks/synthesize_bottom_up.py','--artifact',native,'--check-points','0','--output',bottom],cwd=cpwa,env=cenv)
        run([sys.executable,'benchmarks/optimize_min_circuit_cpsat.py','--artifact',bottom,'--intersections','0','--time-limit',a.cpsat_time_limit,'--workers',a.cpsat_workers,'--output',final],cwd=cpwa,env=cenv)
        cmd=[sys.executable,'benchmarks/benchmark_register_schedules.py',final,'--schedules','min-live','--batches',*map(str,a.batches),'--output',case/'cuda']
        if a.arch:cmd.extend(['--arch',a.arch])
        run(cmd,cwd=cpwa,env=cenv)
        for r in read_csv(case/'cuda'/'summary.csv'):bench[(nc,budget,int(r['batch']))]=(r,final)
    rows=[]
    for s in selections:
        nc=s['components'];target=s['rmse_target'];tr=s['tdar']
        if tr is None:
            rows.append({'components':nc,'rmse_target':target,'status':'UNATTAINED'})
            continue
        budget=int(tr['budget'])
        for batch in a.batches:
            rr,final=bench[(nc,budget,batch)];dr=direct_by[(nc,batch)]
            pr=float(dr['fixed_ns_per_state']);cp=float(rr['ns_per_eval'])
            rows.append({'components':nc,'rmse_target':target,'status':'PASS','tdar_budget':budget,
                'actual_rmse':fv(tr,'rmse'),'p99_abs':fv(tr,'p99_abs'),'max_abs':fv(tr,'max_abs'),
                'points':iv(tr,'points'),'simplices':iv(tr,'simplices'),'tdar_seconds':fv(tr,'tdar_seconds'),
                'dag_nodes':iv(rr,'nodes'),'affine_nodes':iv(rr,'affine_nodes'),'min_nodes':iv(rr,'min_nodes'),'max_nodes':iv(rr,'max_nodes'),
                'executed_ops':iv(rr,'executed_ops'),'live_slots':iv(rr,'live_slots'),'registers':iv(rr,'registers'),
                'spill_store_bytes':iv(rr,'spill_store_bytes'),'spill_load_bytes':iv(rr,'spill_load_bytes'),'stack_frame_bytes':iv(rr,'stack_frame_bytes'),
                'batch':batch,'pr_ns_per_state':pr,'cpwa_ns_per_state':cp,'speedup':pr/cp,
                'max_delta_cpu':fv(rr,'max_delta_cpu'),'artifact':str(final)})
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Equal-accuracy distinct-mixture GPU crossover','',
           'Each target selects the smallest measured TDAR budget whose frozen-validation RMSE meets the target. No budget interpolation is used.','',
           '| C | RMSE target | B | actual RMSE | batch | PR ns | CPWA ns | speedup | regs | spills S/L |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in selections:
        if not s['tdar']:
            lines.append(f"| {s['components']} | {s['rmse_target']:.3g} | NA | NA | NA | NA | NA | NA | NA | NA |")
            continue
        match=[r for r in rows if r.get('components')==s['components'] and r.get('rmse_target')==s['rmse_target']]
        for r in match:lines.append(f"| {r['components']} | {r['rmse_target']:.3g} | {r['tdar_budget']} | {r['actual_rmse']:.6g} | {r['batch']} | {r['pr_ns_per_state']:.6g} | {r['cpwa_ns_per_state']:.6g} | {r['speedup']:.4g}x | {r['registers']} | {r['spill_store_bytes']}/{r['spill_load_bytes']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    metadata={'components':a.components,'rmse_targets':a.rmse_targets,'batches':a.batches,'selection':'smallest measured budget satisfying RMSE target; no interpolation','tdar_results':str(results)}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'\nwrote: {out/"summary.csv"}');print(f'wrote: {out/"report.md"}')
if __name__=='__main__':main()
