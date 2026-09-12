#!/usr/bin/env python3
"""V24.52 fail-closed qualification of the real CLARITAS production trace path.

This is an engineering integrity gate, not an H170 scientific result.  It executes the
canonical production model for the exact V24.42 failure-regression configuration and
writes only integrity/failure diagnostics required for authorization.
"""
from __future__ import annotations
import sys; sys.dont_write_bytecode=True
import argparse,json,time
from pathlib import Path
import pandas as pd
from v24_52_release import HERE,RELEASE
from run_v24_52_case import make_model,write_production_failure_diagnostics

QUALIFICATION_MATERIAL='loess'
QUALIFICATION_CONCENTRATION_G_L=0.5
QUALIFICATION_RAYS=1_000_000
QUALIFICATION_SEED=2_441_000


def _gpu_metadata():
    try:
        import cupy as cp
        dev=int(cp.cuda.runtime.getDevice())
        props=cp.cuda.runtime.getDeviceProperties(dev)
        name=props.get('name',b'unknown')
        if isinstance(name,bytes): name=name.decode(errors='replace')
        return {'gpu_name':str(name),'cuda_runtime_version':int(cp.cuda.runtime.runtimeGetVersion()),'cupy_version':str(cp.__version__),'device_index':dev}
    except Exception as exc:
        return {'gpu_name':'unknown','cuda_runtime_version':None,'cupy_version':None,'device_index':None,'metadata_error':repr(exc)}


def run_qualification(root:Path,config_path:Path|None=None,n_rays:int=QUALIFICATION_RAYS,seed:int=QUALIFICATION_SEED)->dict:
    root=Path(root);out=root/'_production_qualification';out.mkdir(parents=True,exist_ok=True)
    config_path=HERE/'claritas_v24_52_config.json' if config_path is None else Path(config_path)
    t0=time.perf_counter();model,cfg=make_model(config_path);compile_s=time.perf_counter()-t0
    t1=time.perf_counter();res=model.simulate(QUALIFICATION_MATERIAL,QUALIFICATION_CONCENTRATION_G_L,n_rays=int(n_rays),seed=int(seed));transport_s=time.perf_counter()-t1
    base={'material':QUALIFICATION_MATERIAL,'concentration_g_per_L':QUALIFICATION_CONCENTRATION_G_L,'seed':int(seed),'qualification_kind':'engineering_integrity_not_scientific_result'}
    diag=write_production_failure_diagnostics(out,res,base,prefix='v24_52')
    failed=int(diag['production_failure_count']);watchdogs=int(diag['watchdog_count']);geom=int(diag['geometry_failure_count']);nonfinite=int(diag['nonfinite_count']);closure=bool(diag['failure_code_count_closure'])
    code_counts=dict(getattr(res,'production_failure_code_counts',{}) or {})
    sub_counts=dict(getattr(res,'production_failure_subcode_counts',{}) or {})
    micro_numerical=int(code_counts.get('PARTICLE_MICROSURFACE_NUMERICAL',0))
    water_no_boundary_sub=int(sub_counts.get('3001',sub_counts.get(3001,0)))
    all_codes_zero=not any(int(v) for v in code_counts.values())
    water_no_boundary=int(code_counts.get('WATER_BOUNDARY_NO_INTERSECTION',0))
    headspace_fail=int(code_counts.get('HEADSPACE_BOUNDARY_NO_INTERSECTION',0))
    annulus_fail=int(code_counts.get('ACRYLIC_ANNULUS_BOUNCE_LIMIT',0))+int(code_counts.get('ACRYLIC_ANNULUS_NO_INTERSECTION',0))
    invalid_medium=int(code_counts.get('INVALID_MEDIUM_STATE',0))
    water_outside=int(sub_counts.get('3005',sub_counts.get(3005,0)))
    passed=bool(int(res.n_rays)==int(n_rays) and failed==0 and watchdogs==0 and geom==0 and nonfinite==0 and closure and all_codes_zero and water_no_boundary==0 and water_no_boundary_sub==0 and annulus_fail==0 and invalid_medium==0 and water_outside==0 and headspace_fail==0)
    meta=_gpu_metadata()
    old_v2450_failure_count=7734
    old_v2450_path=HERE/'references'/'v24_50_gpu_results'/'v24_50_production_qualification_summary.csv'
    if old_v2450_path.is_file():
        try: old_v2450_failure_count=int(pd.read_csv(old_v2450_path).iloc[0]['production_failure_count'])
        except Exception: pass
    decision={
        'release':RELEASE,'production_transport_qualification_run':True,'production_transport_qualification_pass':passed,
        'production_qualification_material':QUALIFICATION_MATERIAL,'production_qualification_concentration_g_per_L':QUALIFICATION_CONCENTRATION_G_L,
        'production_qualification_rays':int(res.n_rays),'production_qualification_requested_rays':int(n_rays),'production_qualification_seed':int(seed),
        'production_qualification_failure_count':failed,'production_qualification_numerical_zero':failed==0,
        'production_qualification_watchdogs_zero':watchdogs==0,'production_qualification_watchdog_count':watchdogs,
        'production_qualification_nonfinite_zero':nonfinite==0,'production_qualification_nonfinite_count':nonfinite,
        'production_qualification_geometry_failures_zero':geom==0,'production_qualification_geometry_failure_count':geom,'production_geometry_failures_zero':geom==0,
        'production_microsurface_failures_zero':micro_numerical==0,'production_microsurface_failure_count':micro_numerical,
        'production_watchdogs_zero':watchdogs==0,'production_nonfinite_zero':nonfinite==0,
        'production_water_no_boundary_zero':water_no_boundary==0 and water_no_boundary_sub==0,'production_water_no_boundary_count':water_no_boundary,'production_water_no_boundary_subcode_count':water_no_boundary_sub,'production_headspace_failures_zero':headspace_fail==0,'production_headspace_failure_count':headspace_fail,
        'production_failure_codes_all_zero':all_codes_zero,'production_failure_diagnostics_written':True,'production_failure_code_count_closure':closure,'production_acrylic_annulus_failures_zero':annulus_fail==0,'production_acrylic_annulus_failure_count':annulus_fail,'production_invalid_medium_zero':invalid_medium==0,'production_invalid_medium_count':invalid_medium,'production_water_outside_radius_zero':water_outside==0,'production_water_outside_radius_count':water_outside,
        'v24_50_production_failure_count_baseline':int(old_v2450_failure_count),'v24_52_failure_reduction_vs_v24_50':int(old_v2450_failure_count-failed),'v24_50_failure_mechanism_eliminated':bool(annulus_fail==0),'scientific_result_finalized':False,'h170_used_for_design_or_acceptance':False,
        'cuda_compile_model_init_seconds':float(compile_s),'transport_runtime_seconds':float(transport_s),'rays_per_second':float(res.n_rays/max(transport_s,1e-30)),
        **meta,
    }
    (out/'v24_52_production_qualification_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=True)+'\n')
    pd.DataFrame([decision]).to_csv(out/'v24_52_production_qualification_decision.csv',index=False)
    pd.DataFrame([{'release':RELEASE,'compile_model_init_seconds':compile_s,'transport_runtime_seconds':transport_s,'rays_per_second':decision['rays_per_second'],**meta,'seed':int(seed),'n_rays':int(res.n_rays)}]).to_csv(out/'v24_52_production_qualification_performance.csv',index=False)
    if passed:
        print(f"PASS GPU production qualification: {res.n_rays} rays, 0 production failures, 0 watchdogs")
    else:
        print('FAIL GPU production qualification')
        print(f"case: {QUALIFICATION_MATERIAL} {QUALIFICATION_CONCENTRATION_G_L:g} g/L")
        print(f"rays: {res.n_rays}")
        print(f"failed rays: {failed}")
        print(f"watchdogs: {watchdogs}")
        print(f"diagnostics: {out}")
        print('sediment campaign NOT authorized')
    return decision


def main():
    ap=argparse.ArgumentParser(description='CLARITAS V24.52 full-production engineering qualification')
    ap.add_argument('--output-dir',default='results');ap.add_argument('--config');ap.add_argument('--n-rays',type=int,default=QUALIFICATION_RAYS);ap.add_argument('--seed',type=int,default=QUALIFICATION_SEED)
    a=ap.parse_args();d=run_qualification(Path(a.output_dir),Path(a.config) if a.config else None,a.n_rays,a.seed)
    raise SystemExit(0 if d['production_transport_qualification_pass'] else 2)

if __name__=='__main__':main()
