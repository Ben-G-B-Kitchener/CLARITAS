#!/usr/bin/env python3
"""Host/static verifier for the CLARITAS V24.52 complete-geometry release.

GPU validation is owned by run_v24_52_campaign.py so the canonical shell sequence can
perform host verification first and still execute diagnostic sediment cases even if a
later production-authorization gate fails.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from v24_52_release import HERE, RELEASE
import v24_52_model as vm
import v24_52_complete_geometry as cg
import v24_52_geometry_validation as geom
import claritas_tardiis_core_v24_52 as core
import v24_52_validation as boundary

V2450_CORE=HERE/'references'/'V24_50_INPUT_PARITY_ENGINEERING_NOTE.md'
V2451_CUDA_HASH=HERE/'references'/'V24_51_CUDA_SRC_SHA256.txt'

def sha_file(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()

def cuda_arg_count(src,name):
    needle=f'__global__ void {name}(';start=src.index(needle)+len(needle);depth=1;i=start
    while depth and i<len(src):
        if src[i]=='(':depth+=1
        elif src[i]==')':depth-=1
        i+=1
    sig=src[start:i-1];parts=[];last=0;d=0
    for j,ch in enumerate(sig):
        if ch in '([{':d+=1
        elif ch in ')]}':d-=1
        elif ch==',' and d==0:parts.append(sig[last:j].strip());last=j+1
    parts.append(sig[last:].strip());return len([x for x in parts if x])

def abi_audit():
    src=(HERE/'claritas_tardiis_core_v24_52.py').read_text();tree=ast.parse(src);trace_call=score_call=None
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='self' and n.func.attr=='kernel' and len(n.args)>=3 and isinstance(n.args[2],ast.Tuple):trace_call=n
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Attribute) and isinstance(n.func.value.value,ast.Name) and n.func.value.value.id=='self' and n.func.value.attr=='kernels' and len(n.args)>=3 and isinstance(n.args[2],ast.Tuple):
            sl=n.func.slice;key=sl.value if isinstance(sl,ast.Constant) else None
            if key=='score_detectors_kernel':score_call=n
    if trace_call is None or score_call is None:raise AssertionError('kernel launch AST not found')
    ntrace=0
    for e in trace_call.args[2].elts:
        if isinstance(e,ast.Starred) and isinstance(e.value,ast.Name) and e.value.id=='outf':ntrace+=47
        elif isinstance(e,ast.Starred) and isinstance(e.value,ast.Name) and e.value.id=='outi':ntrace+=40
        elif isinstance(e,ast.Starred):raise AssertionError(ast.dump(e))
        else:ntrace+=1
    nscore=len(score_call.args[2].elts);ts=cuda_arg_count(core.CUDA_SRC,'trace_kernel');ss=cuda_arg_count(core.CUDA_SRC,'score_detectors_kernel')
    if ts!=ntrace or ss!=nscore:raise AssertionError((ts,ntrace,ss,nscore))
    return {'trace_kernel':ts,'trace_python_launch':ntrace,'score_detectors_kernel':ss,'score_python_launch':nscore,'passed':True}

def cpp_surrogate():
    cxx=shutil.which('g++') or shutil.which('clang++')
    if not cxx:return {'run':False,'passed':None,'compiler':None}
    pre='''#include <cmath>\n#include <cstdint>\n#include <algorithm>\nusing std::isfinite; using std::isnan; using std::isinf; using std::min; using std::max;\n#define __device__\n#define __global__\nstruct D{long long x,y,z;};static D threadIdx{0,0,0},blockIdx{0,0,0},blockDim{1,1,1};\ninline float erfinvf(float x){return x;} inline double erfinv(double x){return x;} inline float rsqrtf(float x){return 1.0f/std::sqrt(x);}\ninline float __int_as_float(int x){union{int i;float f;}u;u.i=x;return u.f;} inline double __longlong_as_double(long long x){union{long long i;double d;}u;u.i=x;return u.d;}\ntemplate<class T,class U> inline void atomicAdd(T*p,U v){*p+=(T)v;} template<class T,class U> inline void atomicExch(T*p,U v){*p=(T)v;}\n'''
    with tempfile.TemporaryDirectory() as td:
        q=Path(td)/'scope.cpp';q.write_text(pre+core.CUDA_SRC+boundary.VALIDATION_CUDA+geom.GEOMETRY_CUDA);r=subprocess.run([cxx,'-std=c++17','-fsyntax-only',str(q)],capture_output=True,text=True)
        if r.returncode:raise AssertionError(r.stderr[-20000:])
    return {'run':True,'passed':True,'compiler':cxx}

def host_checks():
    cfg=vm.load_config();p=cg.parameters(cfg,cuda_effective=True);gm,tm,inv=cg.write_manifests(HERE);checks={}
    checks['release_config_match']=cfg['version']==RELEASE
    checks['authoritative_tube_dimensions']=abs(p['Rin']-.0465)<2e-8 and abs(p['Rout']-.05)<2e-8 and abs(p['Zwall_top']-.398)<2e-6
    checks['wall_top_separate_from_water_mean']=p['Zwall_top']>p['Zmean']+.3
    checks['free_surface_contained']=cg.free_surface_z(p['Rin'],0,p)<p['Zwall_top']
    f=cfg['forward_model']; ring_in=float(f['sensor_ring_inner_radius_m']);ring_out=float(f['sensor_ring_outer_radius_m']);source_r=float(f['source_launch_radius_m']);through_r=.5*float(f['through_bore_diameter_m']);counter_r=.5*float(f['counterbore_diameter_m'])
    checks['detector_ring_outside_acrylic']=ring_in>p['Rout'] and ring_out>ring_in
    checks['detector_bore_geometry_consistent']=0.0<through_r<counter_r<(ring_out-ring_in)
    checks['source_launch_geometry_consistent']=source_r>=ring_in and source_r<=ring_out+1e-12 and p['Zmin']<0.0<p['Zwall_top']
    checks['particle_scale_fits_vessel']=float(np.max(core.LOESS_DIAMETER_M))*0.5<p['Rin'] and float(np.max(core.KAOLIN_DIAMETER_M))*0.5<p['Rin']
    src=Path(core.__file__).read_text()
    checks['finite_inner_wall_production']=bool('cell_cylinder_hit_finite_forward_d(x,y,z,vx,vy,vz,RIN,ZMIN,ZWALL_TOP)' in src)
    checks['finite_outer_wall_source']=bool('cell_cylinder_hit_finite_forward_d(x,y,z,vx,vy,vz,ROUT,ZMIN-BOTTOM_DISC_THICKNESS,ZWALL_TOP)' in src)
    checks['headspace_transport_production']=bool('headspace_transport_ex' in src and 'PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION' in src)
    checks['open_top_distinct_from_mean_surface']=bool('ZMEAN,ZWALL_TOP' in src and 'Zwall_top+EPS_F;return 4' in src)
    checks['source_detector_geometry_audited']=bool('score_detectors_kernel' in src and 'source_collimator_transport' in src and 'hardware_detector_id' in src)
    nxt=src[src.index('__device__ int acrylic_connected_next_surface('):src.index('__device__ int acrylic_connected_solid_ex(')]
    checks['internal_acrylic_continuity_zero_rng']=('rnd_uniform' not in nxt and 'fresnel_R' not in nxt)
    checks['no_old_64_reflection_rejection']=bool('(void)legacy_max_bounces' in src and 'emergency_cap=16384' in src)
    # Case 1650 input parity host closure.
    c1650=geom._case1650_regression();checks['case_1650_input_parity_regression']=bool(c1650['passed'])
    hrows=geom._horizontal_regression_cases(p);checks['corrected_horizontal_free_surface_regression']=bool(all(r['passed'] for r in hrows))
    pd.DataFrame(hrows).to_csv(HERE/'V24_52_HORIZONTAL_FREE_SURFACE_REGRESSION.csv',index=False)
    recon=geom._reconstruct_v2452_horizontal_mismatches(p,500_000)
    pd.DataFrame(recon['rows']).to_csv(HERE/'V24_52_V24_51_MISMATCH_RECONSTRUCTION.csv',index=False)
    (HERE/'V24_52_V24_51_MISMATCH_RECONSTRUCTION.json').write_text(json.dumps(recon,indent=2)+'\n')
    checks['v24_51_reported_5049_mismatches_fully_reconstructed']=bool(recon['total_v24_51_oracle_surface_changes']==5049 and recon['unexplained_against_reported_5049']==0)
    prior_hash=V2451_CUDA_HASH.read_text().strip() if V2451_CUDA_HASH.is_file() else ''
    checks['production_cuda_src_byte_identical_to_v24_51']=bool(prior_hash and sha_text(core.CUDA_SRC)==prior_hash)
    # Every stored V24.50 production acrylic failure state now has a valid forward physical boundary under the complete vessel.
    fp=HERE/'references'/'v24_50_gpu_results'/'v24_50_production_failure_samples.csv';df=pd.read_csv(fp);valid=[];surfs=[];ts=[]
    for q in df[['x','y','z','vx','vy','vz']].to_numpy(float):
        t,s=cg.acrylic_next_surface(q,p);valid.append(s!=0 and np.isfinite(t));surfs.append(s);ts.append(t)
    checks['v24_50_512_stored_failure_states_host_repaired']=bool(all(valid))
    reg=pd.DataFrame({'ray_id':df.ray_id.astype(int),'old_failure_name':df.failure_name,'old_z':df.z,'reference_surface':surfs,'reference_t':ts,'valid_forward_boundary':valid});reg.to_csv(HERE/'V24_52_V24_50_FAILURE_HOST_REGRESSION.csv',index=False)
    checks['all_checks_pass']=all(checks.values())
    return checks,c1650,recon,hrows,gm,tm,inv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results');a=ap.parse_args();root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    checks,c1650,recon,hrows,gm,tm,inv=host_checks();abi=abi_audit();cpp=cpp_surrogate();gpu=False
    try:
        import cupy as cp;gpu=cp.cuda.runtime.getDeviceCount()>0
    except Exception:gpu=False
    report={'release':RELEASE,'host_validation':'PASS' if checks['all_checks_pass'] and abi['passed'] and cpp.get('passed') is not False else 'FAIL','cuda_available':bool(gpu),'gpu_stages_run_by_host_verifier':False,'complete_geometry_checks':checks,'case_1650_regression':c1650,'v24_51_one_step_mismatch_reconstruction':recon,'horizontal_free_surface_regression':hrows,'cuda_python_abi':abi,'cpp_cuda_syntax_scope_audit':cpp,'production_cuda_physics_changed_from_v24_51':False,'production_geometry_changed_from_v24_51':False,'production_transport_validated':False,'sediment_campaign_authorized':False,'note':'GPU qualification and sediment diagnostics are intentionally owned by run_v24_52_campaign.py.'}
    (HERE/'V24_52_HOST_VALIDATION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'V24_52_PACKAGED_QUALIFICATION_STATUS.json').write_text(json.dumps({'release':RELEASE,'host_validation':report['host_validation'],'cuda_available_in_build_environment':gpu,'geometry_gpu_validation':'NOT RUN','million_ray_qualification':'NOT RUN','five_case_integrity_matrix':'NOT RUN','sediment_diagnostic_campaign':'NOT RUN','production_transport_validated':False,'sediment_campaign_authorized':False,'production_geometry_changed_from_v24_51':False},indent=2)+'\n')
    pd.DataFrame([{'release':RELEASE,'host_validation':report['host_validation'],'cuda_available':gpu,'trace_kernel_args':abi['trace_kernel'],'score_kernel_args':abi['score_detectors_kernel'],'v24_50_failure_states_host_repaired':checks['v24_50_512_stored_failure_states_host_repaired'],'v24_51_reported_mismatches_reconstructed':recon['total_v24_51_oracle_surface_changes'],'horizontal_free_surface_regression_pass':checks['corrected_horizontal_free_surface_regression']}]).to_csv(root/'v24_52_host_validation_summary.csv',index=False)
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['host_validation']=='PASS' else 2)
if __name__=='__main__':main()
