#!/usr/bin/env python3
"""Analyze ThermoGPU M7 scaling CSV and generate reproducible figures/tables."""
import argparse, csv, math
from pathlib import Path
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--outdir',default='results/m7');a=p.parse_args()
out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
with open(a.csv,newline='') as f: rows=list(csv.DictReader(f))
for r in rows:
    for k in ('components','states','threads','calls_per_sample'):r[k]=int(r[k])
    for k in ('median_seconds','ns_per_state','states_per_second','mad_percent','speedup_vs_scalar','speedup_vs_best_cpu'):r[k]=float(r[k])

# Throughput by component count, one figure per component count.
for nc in sorted({r['components'] for r in rows}):
    fig,ax=plt.subplots(figsize=(7,4.5))
    rr=[r for r in rows if r['components']==nc]
    keys=[]
    for r in rr:
        label=r['backend']+(f"-{r['threads']}" if r['backend']=='openmp' else '')
        if label not in keys:keys.append(label)
    for label in keys:
        q=[r for r in rr if r['backend']+(f"-{r['threads']}" if r['backend']=='openmp' else '')==label]
        q.sort(key=lambda x:x['states']);ax.plot([x['states'] for x in q],[x['states_per_second'] for x in q],marker='o',label=label)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Batch states');ax.set_ylabel('States / second');ax.set_title(f'ThermoGPU throughput — {nc} component(s)');ax.grid(True,which='both',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/f'throughput_{nc}c.png',dpi=160);plt.close(fig)

# Measured crossover brackets against the fastest stable CPU row at each N.
# The interpolation is local and descriptive: linearly interpolate the measured
# time difference (GPU - best CPU) between the two points that bracket zero.
summary=[]
for nc in sorted({r['components'] for r in rows}):
    rr=[r for r in rows if r['components']==nc]
    for gpu in ('cuda_resident','cuda_e2e'):
        points=[]
        for n in sorted({r['states'] for r in rr}):
            cpu=[r for r in rr if r['states']==n and r['backend'] in ('scalar','openmp') and r['mad_percent']<=10.0]
            gr=[r for r in rr if r['states']==n and r['backend']==gpu and r['mad_percent']<=10.0]
            if cpu and gr:
                tcpu=min(x['median_seconds'] for x in cpu)
                tgpu=gr[0]['median_seconds']
                points.append((n,tcpu/tgpu,tgpu-tcpu))
        bracket='not bracketed'; estimate=math.nan
        for (n0,s0,d0),(n1,s1,d1) in zip(points,points[1:]):
            if (d0>0>=d1) or (d1>0>=d0):
                bracket=f'{n0}..{n1}'
                if d1 != d0:
                    estimate=n0+(n1-n0)*(-d0)/(d1-d0)
                elif d0 == 0:
                    estimate=float(n0)
                break
        summary.append((nc,gpu,bracket,estimate,max((s for _,s,_ in points),default=math.nan)))
with open(out/'crossover_summary.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['components','gpu_backend','measured_crossover_bracket_states','local_interpolated_crossover_states','max_measured_speedup_vs_best_cpu']);w.writerows(summary)
print(f'wrote {out}')
