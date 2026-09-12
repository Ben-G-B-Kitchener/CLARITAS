#!/usr/bin/env python3
"""V24.52 pre-authorization cross-case production integrity matrix.

Engineering execution-integrity only. It uses the real production trace kernel but
never finalizes scientific H170 outputs. The exact 1,000,000-ray loess 0.5 g/L
qualification is performed separately before this matrix.
"""
from __future__ import annotations
import sys; sys.dont_write_bytecode=True
import json,time
from pathlib import Path
import pandas as pd
from v24_52_release import HERE,RELEASE
from run_v24_52_case import make_model,write_production_failure_diagnostics

INTEGRITY_RAYS=100_000
CASES=(
    ('loess',2.0,2_441_100),('loess',4.0,2_441_200),
    ('kaolin',0.5,2_451_000),('kaolin',2.0,2_451_100),('kaolin',4.0,2_451_200),
)

def run_integrity_matrix(root:Path,config_path:Path|None=None,n_rays:int=INTEGRITY_RAYS)->dict:
    root=Path(root);out=root/'_integrity_matrix';out.mkdir(parents=True,exist_ok=True)
    config_path=HERE/'claritas_v24_52_config.json' if config_path is None else Path(config_path)
    model,_=make_model(config_path); rows=[];all_pass=True
    for mat,conc,seed in CASES:
        t=time.perf_counter();res=model.simulate(mat,conc,n_rays=int(n_rays),seed=int(seed));dt=time.perf_counter()-t
        case_dir=out/f'{mat}_{conc:g}gL';case_dir.mkdir(exist_ok=True)
        diag=write_production_failure_diagnostics(case_dir,res,{'material':mat,'concentration_g_per_L':conc,'seed':seed,'qualification_kind':'pre_authorization_integrity_matrix'})
        failed=int(diag['production_failure_count']);watch=int(diag['watchdog_count']);nonfinite=int(diag['nonfinite_count'])
        ok=bool(int(res.n_rays)==int(n_rays) and failed==0 and watch==0 and nonfinite==0 and bool(diag['failure_code_count_closure']))
        all_pass &= ok
        rows.append({'release':RELEASE,'material':mat,'concentration_g_per_L':conc,'seed':seed,'n_rays':int(res.n_rays),'production_failure_count':failed,'watchdog_count':watch,'nonfinite_count':nonfinite,'failure_code_count_closure':bool(diag['failure_code_count_closure']),'runtime_seconds':dt,'rays_per_second':float(res.n_rays/max(dt,1e-30)),'passed':ok})
    df=pd.DataFrame(rows);df.to_csv(out/'v24_52_integrity_matrix_cases.csv',index=False)
    d={'release':RELEASE,'cross_case_integrity_matrix_run':True,'cross_case_integrity_matrix_pass':bool(all_pass and len(rows)==len(CASES)),'integrity_matrix_rays_per_case':int(n_rays),'integrity_matrix_cases_required':len(CASES),'integrity_matrix_cases_run':len(rows),'total_failures':int(df.production_failure_count.sum()) if len(df) else -1,'total_watchdogs':int(df.watchdog_count.sum()) if len(df) else -1,'total_nonfinite':int(df.nonfinite_count.sum()) if len(df) else -1,'scientific_result_finalized':False,'h170_used_for_design_or_acceptance':False}
    (out/'v24_52_integrity_matrix_decision.json').write_text(json.dumps(d,indent=2)+'\n');pd.DataFrame([d]).to_csv(out/'v24_52_integrity_matrix_decision.csv',index=False)
    print(('PASS' if d['cross_case_integrity_matrix_pass'] else 'FAIL')+f" GPU integrity matrix: {len(rows)}/{len(CASES)} cases run at {n_rays} rays/case; failures={d['total_failures']}")
    return d

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results');ap.add_argument('--config');ap.add_argument('--n-rays',type=int,default=INTEGRITY_RAYS);a=ap.parse_args()
    d=run_integrity_matrix(Path(a.output_dir),Path(a.config) if a.config else None,a.n_rays);raise SystemExit(0 if d['cross_case_integrity_matrix_pass'] else 2)
