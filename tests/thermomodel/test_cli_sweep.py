import csv,io,math,subprocess,sys
r=subprocess.run([sys.argv[1],"sweep","--model","polynomial","--p","200000","--T-start","250","--T-stop","800","--steps","101"],capture_output=True,text=True,check=True)
rows=list(csv.DictReader(io.StringIO(r.stdout)))
assert len(rows)==101,len(rows)
for i,row in enumerate(rows):
    v={k:float(x) for k,x in row.items()}
    assert all(math.isfinite(x) for x in v.values())
    assert abs(v['T_K']-(250+5.5*i))<1e-10
    assert abs(v['cp_J_kgK']-v['cv_J_kgK']-287.05)<1e-9
print('CLI sweep PASS')
