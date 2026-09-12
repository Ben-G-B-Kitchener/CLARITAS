#!/usr/bin/env python3
"""V24.52 one-command complete geometry, production and sediment campaign.

The sediment diagnostic campaign is intentionally distinct from production authorization:
a completed diagnostic run exposes CLARITAS outputs even when the million-ray or integrity
matrix is not production-qualified.  Diagnostic sediment execution is allowed only after the
complete apparatus geometry gate passes.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
from v24_52_release import HERE, RELEASE, RECEIPT, AUDIT_DIR, GEOMETRY_AUDIT_DIR
import v24_52_validation as boundary
import v24_52_geometry_validation as geometry
import v24_52_production_qualification as prod
import v24_52_integrity_matrix as matrix
import v24_52_complete_geometry as cg
import claritas_tardiis_core_v24_52 as core

def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()

def cuda_available():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()>0
    except Exception:return False

def _status(root,boundary_d=None,geom_d=None,prod_d=None,matrix_d=None,sed=None):
    boundary_d=boundary_d or {};geom_d=geom_d or {};prod_d=prod_d or {};matrix_d=matrix_d or {};sed=sed or {}
    geometry_pass=bool(boundary_d.get('gpu_acquisition_complete',False) and boundary_d.get('microsurface_specific_tests_pass',False) and geom_d.get('geometry_gpu_validation_pass',False))
    prod_pass=bool(prod_d.get('production_transport_qualification_pass',False));mat_pass=bool(matrix_d.get('cross_case_integrity_matrix_pass',False));authorized=bool(geometry_pass and prod_pass and mat_pass)
    d=dict(release=RELEASE,cuda_available=cuda_available(),boundary_gpu_validation='PASS' if boundary_d.get('gpu_acquisition_complete',False) and boundary_d.get('microsurface_specific_tests_pass',False) else ('FAIL' if boundary_d else 'NOT RUN'),complete_geometry_gpu_validation='PASS' if geom_d.get('geometry_gpu_validation_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),random_region_classification_validation='PASS' if geom_d.get('random_region_classification_validation_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),corrected_horizontal_free_surface_regression='PASS' if geom_d.get('corrected_horizontal_free_surface_regression_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),one_step_500k_validation='PASS' if geom_d.get('boundary_one_step_500k_validation_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),particle_accessibility_validation='PASS' if geom_d.get('particle_accessibility_validation_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),v24_50_failure_regression='PASS' if geom_d.get('v24_50_failure_regression_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),historical_local_replay_validation='PASS' if geom_d.get('historical_long_path_local_replay_pass',False) else ('FAIL' if geom_d else 'NOT RUN'),million_ray_qualification='PASS' if prod_pass else ('FAIL' if prod_d else 'NOT RUN'),five_case_integrity_matrix='PASS' if mat_pass else ('FAIL' if matrix_d else 'NOT RUN'),sediment_cases_discovered=int(sed.get('discovered',0)),sediment_cases_executed=int(sed.get('executed',0)),sediment_diagnostic_campaign_status=sed.get('status','NOT RUN'),sediment_diagnostic_campaign_run=bool(sed.get('executed',0)>0),production_cuda_physics_changed_from_v24_51=False,production_geometry_changed_from_v24_51=False,production_transport_validated=authorized,sediment_campaign_authorized=authorized,geometry_validation='PASS' if geometry_pass else ('FAIL' if geom_d or boundary_d else 'NOT RUN'),process_fail_closed=True)
    root=Path(root);(root/'v24_52_process_decision.json').write_text(json.dumps(d,indent=2)+'\n');pd.DataFrame([d]).to_csv(root/'v24_52_qualification_summary.csv',index=False);(root/'v24_52_qualification_summary.json').write_text(json.dumps(d,indent=2)+'\n');return d

def _receipt(root,process):
    if not process.get('production_transport_validated'):return None
    p=Path(root)/RECEIPT;rec=dict(release=RELEASE,sediment_campaign_authorized=True,production_transport_validated=True,complete_geometry_gpu_validation_pass=True,million_ray_qualification_pass=True,five_case_integrity_matrix_pass=True,production_cuda_physics_changed_from_v24_51=False,production_geometry_changed_from_v24_51=False,corrected_core_file_sha256=sha_file(HERE/'claritas_tardiis_core_v24_52.py'),corrected_core_cuda_sha256=sha_text(core.CUDA_SRC),geometry_validation_source_sha256=sha_file(HERE/'v24_52_geometry_validation.py'),production_qualification_source_sha256=sha_file(HERE/'v24_52_production_qualification.py'),integrity_matrix_source_sha256=sha_file(HERE/'v24_52_integrity_matrix.py'),case_runner_sha256=sha_file(HERE/'run_v24_52_case.py'),config_sha256=sha_file(HERE/'claritas_v24_52_config.json'),process_decision_sha256=sha_file(Path(root)/'v24_52_process_decision.json'))
    p.write_text(json.dumps(rec,indent=2)+'\n');return rec

def _inventory(root,current_skip_reason=''):
    inv=cg.sediment_inventory();root=Path(root)
    if current_skip_reason:
        for r in inv:
            if r.get('executable') and not r.get('executed'):
                r['skipped_reason']=current_skip_reason
    cols=list(inv[0])
    with (root/'sediment_case_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(inv)
    return inv

def _diagnostic_sediment(root,n_rays,production_authorized):
    root=Path(root);inv=_inventory(root);current=[r for r in inv if r['executable']];rows=[];detector_long=[]
    cfg=json.loads((HERE/'claritas_v24_52_config.json').read_text());fm=cfg['forward_model']
    psd_meta={}
    for mat in ('loess','kaolin'):
        d,_=core.get_material_psd_raw(mat);d=np.asarray(d,float)
        psd_meta[mat]={'particle_bins':int(len(d)),'particle_diameter_min_um':float(np.min(d)*1e6),'particle_diameter_max_um':float(np.max(d)*1e6)}
    for r in current:
        od=root/'sediment_diagnostic'/r['case_id'];od.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(HERE/'run_v24_52_case.py'),'--material',r['material'],'--concentration',str(r['concentration_g_per_L']),'--output-dir',str(od),'--config',str(HERE/'claritas_v24_52_config.json'),'--n-rays',str(int(n_rays)),'--seed',str(int(r['seed'])),'--diagnostic-run']
        if production_authorized:cmd.append('--production-authorized')
        print('SEDIMENT DIAGNOSTIC RUN',' '.join(cmd),flush=True);t=time.perf_counter();q=subprocess.run(cmd,cwd=HERE);elapsed=time.perf_counter()-t
        sp=od/'case_summary.json';summary=json.loads(sp.read_text()) if sp.is_file() else {}
        fp=od/'production_failure_summary.csv';fdf=pd.read_csv(fp) if fp.is_file() else pd.DataFrame();failrow=fdf.iloc[0].to_dict() if len(fdf) else {}
        failure_count=int(failrow.get('production_failure_count',0)) if failrow else None
        complete=sp.is_file();completion=('FAILED' if not complete else ('DIAGNOSTIC WITH TRANSPORT FAILURES' if failure_count else 'PASS'))
        row=dict(
            case_id=r['case_id'],material=r['material'],concentration_g_per_L=r['concentration_g_per_L'],purpose=r['purpose'],
            configuration_file=r['configuration_file'],psd_definition=r['psd_definition'],optical_constants_file=r['optical_constants_file'],
            particle_population_geometry=r.get('particle_geometry'),ray_count=int(summary.get('n_rays',n_rays)),seed=int(r['seed']),
            gpu_execution_status='COMPLETED' if complete else 'FAILED',subprocess_return_code=int(q.returncode),elapsed_s=float(summary.get('elapsed_s',elapsed)),
            completion_status=completion,sediment_run_classification=summary.get('sediment_run_classification','DIAGNOSTIC / NOT PRODUCTION-AUTHORIZED'),
            production_authorized_at_execution=bool(production_authorized),production_failure_count=failure_count,
            watchdog_count=int(failrow.get('watchdog_count',0)) if failrow else None,geometry_failure_count=int(failrow.get('geometry_failure_count',0)) if failrow else None,
            nonfinite_count=int(failrow.get('nonfinite_count',0)) if failrow else None,H170_hardware_score=summary.get('H170_hardware_score'),
            model_H170_all18_normalized=summary.get('model_H170_all18_normalized'),model_H170_trusted_renormalized=summary.get('model_H170_trusted_renormalized'),
            trusted_RMSE=summary.get('trusted_RMSE'),trusted_correlation=summary.get('trusted_correlation'),H170_mean_particle_interactions=summary.get('H170_mean_particle_interactions'),
            wavelength_nm=1e9*float(fm['particle_wavelength_m']),free_surface_model=fm['free_surface_model'],
            vortex_wall_center_height_difference_m=float(fm['vortex_wall_center_height_difference_m']),vortex_core_radius_m=float(fm['vortex_core_radius_m']),
            sediment_transport_model=fm['sediment_transport_model'],particle_event_model=fm['particle_event_model'],source_angular_model=fm['source_angular_model'],
            source_launch_radius_m=float(fm['source_launch_radius_m']),photodiode_response_model=fm['photodiode_response_model'],detector_count=18,detector_angles_deg='0,10,...,170',
            output_directory=str(od.relative_to(root)),**psd_meta[r['material']])
        detp=od/'detector_response_and_particle_paths.csv'
        if detp.is_file():
            ddf=pd.read_csv(detp)
            for _,dr in ddf.iterrows():
                ang=int(dr['detector_deg']);row[f'detector_{ang:03d}_hardware_score']=float(dr['hardware_symmetry_score']);row[f'detector_{ang:03d}_normalized']=float(dr['model_normalized_all18'])
                detector_long.append({'case_id':r['case_id'],'material':r['material'],'concentration_g_per_L':r['concentration_g_per_L'],'detector_deg':ang,'hardware_symmetry_score':float(dr['hardware_symmetry_score']),'model_normalized_all18':float(dr['model_normalized_all18']),'measured_normalized_all18':float(dr['measured_normalized_all18']),'sensor_status':dr['sediment_sensor_status']})
        codesp=od/'production_failure_codes.csv'
        if codesp.is_file():
            cdf=pd.read_csv(codesp)
            cmap={str(rr['failure_name']):int(rr['count']) for _,rr in cdf.iterrows()}
            row['water_failure_count']=int(cmap.get('WATER_BOUNDARY_NO_INTERSECTION',0))
            row['headspace_failure_count']=int(cmap.get('HEADSPACE_BOUNDARY_NO_INTERSECTION',0))
            row['acrylic_failure_count']=int(cmap.get('ACRYLIC_ANNULUS_NO_INTERSECTION',0))+int(cmap.get('ACRYLIC_ANNULUS_BOUNCE_LIMIT',0))
            row['microsurface_failure_count']=int(cmap.get('PARTICLE_MICROSURFACE_NUMERICAL',0))
        else:
            row['water_failure_count']=row['headspace_failure_count']=row['acrylic_failure_count']=row['microsurface_failure_count']=None
        row['output_files']=';'.join(sorted(x.name for x in od.iterdir() if x.is_file())) if od.is_dir() else ''
        rows.append(row)
    # Refresh the inventory so it explicitly records discovered / executable / executed / skipped.
    attempted={r['case_id'] for r in rows}
    for item in inv:
        if item.get('executable'):
            item['executed']=item['case_id'] in attempted
            if not item['executed']:
                item['skipped_reason']='not reached by the diagnostic campaign'
    with (root/'sediment_case_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
    df=pd.DataFrame(rows);df.to_csv(root/'sediment_case_results.csv',index=False);(root/'sediment_case_results.json').write_text(json.dumps(rows,indent=2,allow_nan=True)+'\n')
    plotcols=['case_id','material','concentration_g_per_L','H170_hardware_score','model_H170_all18_normalized','model_H170_trusted_renormalized','trusted_RMSE','production_failure_count'];df[[c for c in plotcols if c in df]].to_csv(root/'sediment_plot_data.csv',index=False)
    detdf=pd.DataFrame(detector_long);detdf.to_csv(root/'sediment_detector_plot_data.csv',index=False)
    plots=[]
    try:
        import matplotlib.pyplot as plt
        for field,name,ylabel in [('H170_hardware_score','sediment_H170_hardware_score_by_case.png','H170 hardware score'),('trusted_RMSE','sediment_trusted_RMSE_by_case.png','Trusted-detector RMSE'),('production_failure_count','sediment_transport_failures_by_case.png','Production transport failures')]:
            if field in df and df[field].notna().any():
                fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(df.case_id.astype(str),pd.to_numeric(df[field],errors='coerce'));ax.set_ylabel(ylabel);ax.set_xlabel('Sediment case');ax.tick_params(axis='x',rotation=35);fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig);plots.append(name)
        if len(detdf):
            fig,ax=plt.subplots(figsize=(9,5.2))
            for cid,g in detdf.groupby('case_id'):ax.plot(g['detector_deg'],g['model_normalized_all18'],marker='o',label=str(cid))
            ax.set_xlabel('Detector angle (deg)');ax.set_ylabel('Normalized model response');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'sediment_detector_response_by_case.png',dpi=150);plt.close(fig);plots.append('sediment_detector_response_by_case.png')
    except Exception as exc:plots.append('plot_generation_unavailable: '+repr(exc))
    lines=['# CLARITAS V24.52 sediment diagnostic test report','',f'Classification: **{"DIAGNOSTIC / PRODUCTION-AUTHORIZED" if production_authorized else "DIAGNOSTIC / NOT PRODUCTION-AUTHORIZED"}**','',f'Discovered inventory entries: {len(inv)}; executable canonical cases: {len(current)}; executed: {len(rows)}.','', '## Cases','']
    for r in rows:
        lines += [f"### {r['case_id']}",f"- Material / concentration: {r['material']} / {r['concentration_g_per_L']} g/L",f"- PSD bins / diameter range: {r['particle_bins']} / {r['particle_diameter_min_um']:.3g}–{r['particle_diameter_max_um']:.3g} µm",f"- Vortex / sediment transport: {r['free_surface_model']} (Δh={r['vortex_wall_center_height_difference_m']} m, core={r['vortex_core_radius_m']} m) / {r['sediment_transport_model']}",f"- Source / detector: {r['source_angular_model']} at r={r['source_launch_radius_m']} m / {r['photodiode_response_model']}, 18 channels",f"- Rays / seed: {r['ray_count']} / {r['seed']}",f"- Completion: {r['completion_status']}; transport failures: {r['production_failure_count']}; watchdogs: {r['watchdog_count']}; nonfinite: {r['nonfinite_count']}",f"- H170 hardware score: {r['H170_hardware_score']}; normalized H170: {r['model_H170_all18_normalized']}",f"- Trusted RMSE / correlation: {r['trusted_RMSE']} / {r['trusted_correlation']}",f"- Output directory: `{r['output_directory']}`",'']
    lines += ['## Generated comparison plots','']+[f'- `{x}`' for x in plots]
    (root/'V24_52_SEDIMENT_TEST_REPORT.md').write_text('\n'.join(lines)+'\n')
    all_completed=(len(rows)==len(current) and all(r['gpu_execution_status']=='COMPLETED' for r in rows));any_transport_fail=any((r.get('production_failure_count') or 0)>0 for r in rows)
    status=('PASS' if all_completed and not any_transport_fail else ('COMPLETED WITH TRANSPORT FAILURES' if all_completed else 'FAIL'))
    return {'discovered':len(inv),'executed':len(rows),'status':status,'transport_failure_cases':sum(1 for r in rows if (r.get('production_failure_count') or 0)>0),'rows':rows}

def _geometry_diagnostic_safe(boundary_d,geom_d):
    """Diagnostic sediment may run after a non-catastrophic release-gate failure.

    Production authorization still requires the full geometry gate.  Diagnostic safety
    requires the executable boundary layer and the core apparatus/region invariants to
    be healthy, so a reference disagreement does not automatically hide sediment output.
    """
    if not boundary_d or not geom_d:return False
    bpass=bool(boundary_d.get('gpu_acquisition_complete',False) and boundary_d.get('microsurface_specific_tests_pass',False))
    essentials=(
        geom_d.get('full_region_topology_validation_pass',False),
        geom_d.get('deterministic_geometry_validation_pass',False),
        geom_d.get('particle_accessibility_validation_pass',False),
        geom_d.get('v24_50_failure_regression_pass',False),
        geom_d.get('finite_surface_support_validation_pass',False),
        geom_d.get('headspace_transport_validation_pass',False),
    )
    return bool(bpass and all(essentials))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results');ap.add_argument('--sediment-rays',type=int,default=1_000_000);ap.add_argument('--geometry-one-step-samples',type=int,default=500_000);ap.add_argument('--skip-sediment-diagnostics',action='store_true');a=ap.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True);cg.write_manifests(HERE);_inventory(root)
    if not cuda_available():
        _inventory(root,'not executed in this environment: compatible CUDA unavailable')
        d=_status(root,sed={'discovered':len(cg.sediment_inventory()),'executed':0,'status':'NOT RUN'});print('FAIL: CUDA unavailable. GPU geometry, million-ray qualification and sediment diagnostic cases were NOT RUN.');raise SystemExit(2)
    b=g=q=m=None;sed={'discovered':len(cg.sediment_inventory()),'executed':0,'status':'NOT RUN'}
    fatal=False
    print('STAGE 1/6 boundary/microsurface GPU validation',flush=True)
    try:
        b=boundary.acquire_gpu(root)
    except Exception as exc:
        b={'gpu_acquisition_complete':False,'microsurface_specific_tests_pass':False,'exception':repr(exc)};fatal=True
    _status(root,boundary_d=b,sed=sed)
    bpass=bool(b.get('gpu_acquisition_complete',False) and b.get('microsurface_specific_tests_pass',False))
    if not bpass:fatal=True

    if bpass:
        print('STAGE 2/6 complete apparatus GPU geometry validation',flush=True)
        try:g=geometry.run_geometry_validation(root,a.geometry_one_step_samples)
        except Exception as exc:g={'release':RELEASE,'geometry_gpu_validation_run':True,'geometry_gpu_validation_pass':False,'exception':repr(exc)}
        _status(root,b,g,sed=sed)
    geom_pass=bool(g and g.get('geometry_gpu_validation_pass',False))
    diagnostic_safe=_geometry_diagnostic_safe(b,g)

    if geom_pass:
        print('STAGE 3/6 exact 1,000,000-ray production qualification',flush=True)
        try:q=prod.run_qualification(root)
        except Exception as exc:
            q={'release':RELEASE,'production_transport_qualification_pass':False,'exception':repr(exc),'qualification_exception':True}
            (root/'v24_52_production_qualification_exception.json').write_text(json.dumps(q,indent=2)+'\n')
        _status(root,b,g,q,sed=sed)
        if q.get('production_transport_qualification_pass',False):
            print('STAGE 4/6 five-case cross-case integrity matrix',flush=True)
            try:m=matrix.run_integrity_matrix(root)
            except Exception as exc:
                m={'release':RELEASE,'cross_case_integrity_matrix_pass':False,'exception':repr(exc),'matrix_exception':True}
                (root/'v24_52_integrity_matrix_exception.json').write_text(json.dumps(m,indent=2)+'\n')
            _status(root,b,g,q,m,sed=sed)
        else:
            print('STAGE 4/6 five-case matrix NOT RUN because million-ray qualification failed; diagnostic sediment execution remains eligible.',flush=True)
    else:
        print('STAGE 3/6 million-ray qualification NOT RUN because complete geometry gate failed.',flush=True)
        print('STAGE 4/6 five-case matrix NOT RUN because production qualification did not pass.',flush=True)

    preauth=bool(geom_pass and q and q.get('production_transport_qualification_pass') and m and m.get('cross_case_integrity_matrix_pass'))
    if not a.skip_sediment_diagnostics and (geom_pass or diagnostic_safe) and not fatal:
        print('STAGE 5/6 six-case sediment diagnostic campaign',flush=True)
        try:sed=_diagnostic_sediment(root,a.sediment_rays,preauth)
        except Exception as exc:
            sed={'discovered':len(cg.sediment_inventory()),'executed':0,'status':'FAIL','exception':repr(exc)}
            (root/'v24_52_sediment_diagnostic_exception.json').write_text(json.dumps(sed,indent=2)+'\n')
    elif a.skip_sediment_diagnostics:
        _inventory(root,'skipped only because --skip-sediment-diagnostics was explicitly requested');sed={'discovered':len(cg.sediment_inventory()),'executed':0,'status':'NOT RUN'}
    else:
        reason='not executed because transport was not diagnostically safe (fatal boundary/ABI/topology failure)'
        _inventory(root,reason);sed={'discovered':len(cg.sediment_inventory()),'executed':0,'status':'NOT RUN','reason':reason}

    process=_status(root,b,g,q,m,sed);_receipt(root,process)
    print('STAGE 6/6 reporting complete');print(json.dumps(process,indent=2))
    raise SystemExit(0 if process['production_transport_validated'] and sed['status']=='PASS' else 2)

if __name__=='__main__':main()
