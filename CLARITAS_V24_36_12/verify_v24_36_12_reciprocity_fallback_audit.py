#!/usr/bin/env python3
from __future__ import annotations
import sys as _v243611_sys
_v243611_sys.dont_write_bytecode=True
import argparse,ast,hashlib,importlib.util,json,math,re,shutil,subprocess,sys,time,tempfile,os
from pathlib import Path
import numpy as np
import pandas as pd
from v24_36_12_release import HERE,RELEASE,PREFIX,AUDIT_SUBDIR,RECEIPT_NAME,OUTPUT_FILENAMES
from v24_36_12_io import SCHEMAS,finalize_metadata
import v24_36_12_model as vm
import v24_36_12_process_reference as pref
import v24_36_12_complete_kernel as ck
import claritas_tardiis_core_v24_36 as core

DRIVER=HERE/'CLARITAS_24_36_12_09-09-2026_cuda_reciprocity_fallback_audit.py'
SUMMARY=HERE/'summarize_v24_36_12_reciprocity_fallback_audit.py'
PLOTTER=HERE/'plot_v24_36_12_reciprocity_fallback_audit.py'
CAMPAIGN=HERE/'run_v24_36_12_campaign.py'
PREFLIGHT=HERE/'preflight_v24_36_12_package.py'
RECEIPT=HERE/RECEIPT_NAME
CORE=HERE/'claritas_tardiis_core_v24_36.py'
CORE_FILE_SHA='38ce70b515adba207484ed0ce66065cd12dfd5bc96067b61719c6999384a4057'
CORE_CUDA_SHA='a64339e0167644cae3abe86294aaae1001a9d398cd60ebc36e54eedadef98671'
_spec=importlib.util.spec_from_file_location('v243610_driver',DRIVER);drv=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(drv)

def sha_text(s:str)->str:return hashlib.sha256(s.encode()).hexdigest()
def sha_file(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def _kernel_params(src:str,name:str)->list[str]:
    m=re.search(r'__global__\s+void\s+'+re.escape(name)+r'\s*\((.*?)\)\s*\{',src,re.S)
    if not m:raise AssertionError(f'kernel not found: {name}')
    return [x.strip() for x in m.group(1).split(',') if x.strip()]

def _launch_tuple_arities(path:Path)->list[int]:
    tree=ast.parse(path.read_text());out=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id=='args' and isinstance(n.value,ast.Tuple):out.append(len(n.value.elts))
    return out

def _release_runtime():
    cfg=vm.load_config();assert RELEASE=='V24.36.12';assert cfg['version']==RELEASE
    try:finalize_metadata({'release':'V24.36.9'});raise AssertionError('stale metadata accepted')
    except RuntimeError:pass
    text='\n'.join(p.read_text() for p in (DRIVER,SUMMARY,PLOTTER,CAMPAIGN,HERE/'v24_36_12_io.py'))
    assert 'bash' not in text.lower() and 'cygpath' not in text.lower()
    assert 'sys.executable' in CAMPAIGN.read_text()
    print('PASS host: release identity is single-sourced; stale metadata rejected; canonical orchestration is Python-only and uses sys.executable')

def _frozen_physics():
    cfg=vm.load_config();b=cfg['reciprocity_fallback_audit_v24_36_12'];old=json.loads((HERE/'references'/'claritas_v24_36_9_config.json').read_text())
    assert cfg['forward_model']==old['forward_model'],'forward_model changed relative to V24.36.9 diagnostic baseline'
    assert sha_file(CORE)==CORE_FILE_SHA and sha_text(core.CUDA_SRC)==CORE_CUDA_SHA
    assert int(cfg['forward_model']['max_internal_bounces'])==64 and int(b['production_reflection_limit'])==65 and int(b['diagnostic_reflection_limit'])==512
    assert float(cfg['forward_model']['particle_surface_rms_slope_deg'])==25.0
    optics=vm.particle_optics_override();assert all(np.allclose(v['k_imag'],0.0) for v in optics.values())
    src=core.CUDA_SRC
    for needle in ('for(int bounce=0;bounce<=max_bounces;++bounce)','if(bounce==max_bounces) return 0;','if(ps!=1){st=4;break;}'):assert needle in src
    runtime='\n'.join(p.read_text() for p in (DRIVER,HERE/'v24_36_12_model.py',SUMMARY,CAMPAIGN))
    forbidden=('cdf_rank','angular_map','moment matching','H170 fit','empirical loss factor','roughness tuning')
    for x in forbidden:assert x.lower() not in runtime.lower(),x
    assert '.simulate(' not in DRIVER.read_text()
    print('PASS host: frozen production core/config preserved; 64 inclusive loop means 65 reflections; k=0 and 25-degree roughness remain unchanged; no sediment/closure physics added')

def _cap_conservation_and_fallback_static():
    s=core.CUDA_SRC
    assert 'if(bounce==max_bounces) return 0;' in s
    # Production fallback explicitly replaces an incompatible rough transmission with geometric normal.
    for needle in ('if(!refr_out || dot3(ta,tb,tc,nox,noy,noz)<=1e-7f)','rnx=nox;rny=noy;rnz=noz;cii=cii_geom;','if(dot3(ra,rb,rc,nox,noy,noz)>=-1e-7f)'):
        assert needle in s
    # Sampler is Gaussian slope NDF with facing rejection, not a Smith VNDF implementation.
    block=s[s.index('__device__ void beckmann_microfacet_normal'):s.index('__device__ int sample_cdf_index')]
    for needle in ('sigma=0.7071067811865475f*alpha','for(int attempt=0;attempt<8;++attempt)','sqrtf(-2.0f*logf(u1))','desired_dot_sign'):assert needle in block
    assert 'smith' not in block.lower() and 'visible' not in block.lower() and 'mask' not in block.lower()
    print('PASS host: source audit confirms numerical cap sink, rough-to-geometric fallback semantics, and raw Beckmann slope sampling with facing rejection rather than VNDF/masking logic')

def _abi_memory_static():
    assert len(_kernel_params(drv.AUDIT_CUDA,'reciprocity_fallback_kernel'))==36
    assert len(_kernel_params(drv.AUDIT_CUDA,'visibility_kernel'))==5
    assert len(_kernel_params(drv.AUDIT_CUDA,'fixed_microfacet_kernel'))==11
    assert len(_kernel_params(ck.COMPLETE_KERNEL_CUDA,'complete_kernel_kernel'))==19
    ar=_launch_tuple_arities(DRIVER);assert 36 in ar,ar
    cfg=vm.load_config();d=int(cfg['reciprocity_fallback_audit_v24_36_12']['diagnostic_reflection_limit']);assert d==512 and d<np.iinfo(np.uint16).max
    assert 'diag+1' in DRIVER.read_text() and 'diag*5' in DRIVER.read_text() and 'diag*2' in DRIVER.read_text()
    assert not re.search(r'(?<![A-Za-z0-9_])(NAN|INFINITY)(?![A-Za-z0-9_])',drv.AUDIT_CUDA)
    print('PASS host: CUDA/Python ABI closes (main=36, visibility=5, fixed=11, complete-kernel=19); 512-order arrays and 513 reached-state counters are explicitly allocated; no bare NAN/INFINITY')

def _actual_run_bin_regression():
    class _Null:
        @staticmethod
        def synchronize():pass
    class _Stream:null=_Null()
    class _Cuda:Stream=_Stream()
    class CP:
        uint8=np.uint8;uint16=np.uint16;uint32=np.uint32;int32=np.int32;float32=np.float32;cuda=_Cuda()
        @staticmethod
        def zeros(n,dtype=None):return np.zeros(int(n),dtype=dtype)
        @staticmethod
        def asnumpy(x):return np.asarray(x)
    def kernel(grid,block,args):
        assert len(args)==36
        n=int(args[0]);args[10][:n]=1;args[11][:n]=1;args[12][:n]=1;args[14][:n]=1;args[15][:n]=1;args[19][0]=n
    props={'n_medium':1.333,'n_particle':1.59,'k_particle':0.0,'wavelength':850e-9,'radius':0.5e-6}
    out=drv._run_bin(CP,kernel,props,8,123,math.tan(math.radians(25)),65,512)
    assert out['reached'].shape==(513,) and out['macro'].shape==(512,) and out['fbclass'].shape==(512,5) and out['cand'].shape==(512,2)
    assert out['pair_f'].shape==(5,21) and out['pair_i'].shape==(5,8) and len(out['prod_status'])==8
    print('PASS host: actual _run_bin allocation/conversion path executes with real 512-order shapes, pair buffers, and zero-population outputs')

def _cpu_microfacet_and_snell_regressions():
    rows=pref.fixed_microfacet_rows();assert len(rows)>=4 and all(r['passed'] for r in rows)
    # Permanent V24.36.7 conditioning regression reconstructed independently here.
    th=4.977023564332113e-4;n1=1.33;n2=1.58
    i=np.array([math.sin(th),0.0,math.cos(th)]);n=np.array([0.0,0.0,-1.0]);o=pref.refract(i,n,n1,n2);assert o is not None
    ci32=np.float32(abs(np.dot(i,n)));co32=np.float32(abs(np.dot(o,-n)))
    legacy=abs(n1*math.sqrt(max(0.0,1.0-float(ci32*ci32)))-n2*math.sqrt(max(0.0,1.0-float(co32*co32))))
    stable=abs(n1*np.linalg.norm(np.cross(n,i))-n2*np.linalg.norm(np.cross(n,o)))
    assert legacy>2e-5 and stable<1e-12,(legacy,stable)
    print(f'PASS host: fixed-microfacet reflection/refraction reversibility and Walter-style BSDF reciprocity close; V24.36.7 legacy Snell false-positive remains reproduced (legacy={legacy:.3g}, vector={stable:.3g})')

def _visible_normal_reference_regression():
    # Independent deterministic Monte Carlo normalization check for the Smith
    # visible-normal importance ratio used by the GPU diagnostic.
    rng=np.random.default_rng(243610);alpha=math.tan(math.radians(25.0));N=80000
    sx=rng.normal(0.0,alpha/math.sqrt(2.0),N);sy=rng.normal(0.0,alpha/math.sqrt(2.0),N);mz=1.0/np.sqrt(1.0+sx*sx+sy*sy);mx=sx*mz
    worst=0.0
    for c in (0.02,0.1,0.4,0.8,1.0):
        ss=math.sqrt(max(0.0,1.0-c*c));vm=ss*mx+c*mz;face=vm>0.0;v=np.array([ss,0.0,c]);G=pref.smith_G1_beckmann(v,np.array([0.0,0.0,1.0]),np.array([0.0,0.0,1.0]),alpha);w=np.where(face,G*np.abs(vm)/(c*mz),0.0);err=abs(float(w.mean())-1.0);worst=max(worst,err)
    assert worst<0.02,worst
    print(f'PASS host: independent Smith visible-normal reference importance ratio normalizes under raw Beckmann slope sampling (worst MC error {worst:.4g} <0.02)')

def _qsca_event_mu_regressions():
    cfg=vm.load_config();assert not vm.qsca_in_audit(np.nextafter(1.0,0.0),cfg);assert vm.qsca_in_audit(1.0,cfg);assert vm.qsca_in_audit(np.nextafter(1.0,2.0),cfg)
    scale=32767.0
    for mu in (-1.0,-0.99999,0.0,0.99999,1.0):
        q=np.int16(np.clip(np.rint(mu*scale),-32767,32767));assert abs(float(q)/scale-mu)<=1/scale+1e-12
    assert 'PRODUCTION_TYPE[tid]=(unsigned char)((pper>0)?5:((ppir>0)?7:6));' in core.CUDA_SRC
    assert (HERE/'references'/'v24_29_particle_event_type_reference.csv').is_file()
    print('PASS host: Qsca threshold is exact at 1; native-mu quantization edge cases and maximum particle event code 7 remain covered')

def _v243611_bug_regressions():
    ds=DRIVER.read_text();rs=(HERE/'v24_36_12_process_reference.py').read_text()
    assert 'resolved=escaped+absorbed+num+cens' in ds and 'resolved=entry+escaped' not in ds
    fk=drv.AUDIT_CUDA[drv.AUDIT_CUDA.index('__global__ void fixed_microfacet_kernel'):]
    assert '(MODE[j]==0)?fresnel_R_complex(co,N1[j],0.0f,N2[j],0.0f):fresnel_R_complex(co,N2[j],0.0f,N1[j],0.0f)' in fk
    assert 'exact deterministic direction used for invariants' in rs
    # Archived V24.36.11 failure signature: old closure error exactly tracked entry reflection count.
    ref=pd.read_csv(HERE/'references'/'v24_36_11_probability_conservation.csv')
    assert np.all(ref.count_closure_error.to_numpy(int)==ref.entry_reflection_count.to_numpy(int))
    print('PASS host: all three V24.36.11 audit defects have exact permanent regressions (closure, reflection n1/n2, exact-direction invariant)')

def _path_budget():
    root=r'F:\Google Drive Sync\CLARITAS_code\CLARITAS\CLARITAS_V24_36_12'
    names=[]
    for x in OUTPUT_FILENAMES:
        if x in (f'{PREFIX}_process_decision.csv',f'{PREFIX}_process_decision.json'):names.append(f'results\\{x}')
        else:names.append(f'results\\{AUDIT_SUBDIR}\\{x}')
    names += ['PACKAGE_MANIFEST.json','MICROFACET_RECIPROCITY_REFERENCE.md','V24_36_12_ROUGHNESS_PROVENANCE.md']
    longest=max(len(root)+1+len(x) for x in names);assert longest<240,longest
    print(f'PASS Windows path budget: longest default V24.36.12 output/package path is {longest} chars (<240)')

def _surrogate():
    cxx=shutil.which('g++') or shutil.which('clang++')
    if not cxx:print('INFO host: no C++ compiler; CUDA syntax/scope surrogate skipped');return
    pre='''#include <cmath>\n#include <cstdint>\n#include <algorithm>\nusing std::isfinite; using std::llrint; using std::min; using std::max;\n#ifdef NAN\n#undef NAN\n#endif\n#ifdef INFINITY\n#undef INFINITY\n#endif\n#define __device__\n#define __global__\nstruct _dim3{long long x;long long y;long long z;}; static _dim3 threadIdx{0,0,0},blockIdx{0,0,0},blockDim{1,1,1};\ninline float rsqrtf(float x){return 1.0f/std::sqrt(x);}\ntemplate<class T,class U> inline void atomicAdd(T*p,U v){*p+=(T)v;} template<class T,class U> inline void atomicExch(T*p,U v){*p=(T)v;}\ninline unsigned int atomicCAS(unsigned int* p,unsigned int c,unsigned int v){unsigned int o=*p;if(o==c)*p=v;return o;}\n'''
    with tempfile.TemporaryDirectory(prefix='v243612_cuda_') as td:
        p=Path(td)/'audit.cpp';p.write_text(pre+core.CUDA_SRC+drv.AUDIT_CUDA+ck.COMPLETE_KERNEL_CUDA)
        q=subprocess.run([cxx,'-std=c++17','-fsyntax-only',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if q.returncode:raise AssertionError(q.stderr[-20000:])
    print('PASS host: complete frozen production CUDA + appended reciprocity/fallback audit CUDA pass C++ syntax/scope surrogate with host NAN/INFINITY disabled')

def _host_smoke_chain():
    d=Path(tempfile.mkdtemp(prefix='v243612_host_smoke_'))
    try:
        q=subprocess.run([sys.executable,str(DRIVER),'--host-smoke','--output-dir',str(d)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if q.returncode:raise RuntimeError(q.stdout+'\n'+q.stderr)
        for key,cols in SCHEMAS.items():
            p=d/AUDIT_SUBDIR/f'{PREFIX}_{key}.csv';assert p.is_file(),p;assert list(pd.read_csv(p).columns)==cols,p.name
        for script in (SUMMARY,PLOTTER):
            q=subprocess.run([sys.executable,str(script),str(d)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            if q.returncode:raise RuntimeError(q.stdout+'\n'+q.stderr)
        for img in (x for x in OUTPUT_FILENAMES if x.endswith('.png')):assert (d/AUDIT_SUBDIR/img).is_file(),img
        dec=json.loads((d/f'{PREFIX}_process_decision.json').read_text());assert dec['decision']=='host_smoke_structural_only' and not dec['process_correction_authorized']
    finally:shutil.rmtree(d,ignore_errors=True)
    print('PASS host: actual driver -> real writers -> summarizer -> decision writer -> plotter executes end-to-end with empty pair tables and zero/edge populations')

def _campaign_gate():
    import run_v24_36_12_campaign as camp
    d=Path(tempfile.mkdtemp(prefix='v243612_campaign_smoke_'));RECEIPT.unlink(missing_ok=True)
    try:
        q=subprocess.run([sys.executable,str(CAMPAIGN),'--output-dir',str(d)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert q.returncode!=0
        q=subprocess.run([sys.executable,str(DRIVER),'--host-smoke','--output-dir',str(d)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert q.returncode==0,(q.stdout,q.stderr)
        dec=camp.finalize(d,allow_host_smoke=True,require_receipt=False);assert dec['decision']=='host_smoke_structural_only'
    finally:RECEIPT.unlink(missing_ok=True);shutil.rmtree(d,ignore_errors=True)
    print('PASS host: canonical campaign rejects missing GPU receipt; internal host-smoke finalization uses the real summarizer/plotter and never authorizes sediment transport')

def _adversarial_source_review():
    runtime_files=[DRIVER,SUMMARY,PLOTTER,CAMPAIGN,HERE/'v24_36_12_io.py',HERE/'v24_36_12_model.py',HERE/'v24_36_12_process_reference.py',HERE/'v24_36_12_release.py']
    text='\n'.join(p.read_text() for p in runtime_files)
    for bad in ('subprocess.Popen([\'bash','cygpath','run_v24_36_9_campaign','verify_v24_36_9_','CLARITAS_24_36_9_09-09-2026_cuda_trapped_ray_audit'):
        assert bad not in text,bad
    assert json.loads((HERE/'claritas_v24_36_12_config.json').read_text())['forward_model']['max_internal_bounces']==64
    stale=[p.name for p in HERE.glob('*.py') if 'v24_36_9' in p.name.lower() or '24_36_9' in p.name]
    assert not stale,stale
    for p in ('V24_36_10_HANDOFF_REQUIREMENTS.txt','claritas_v24_36_9_config.json','v24_36_9_process_decision.json','v24_36_9_fallback_by_order.csv'):
        assert (HERE/'references'/p).is_file(),p
    print('PASS host: adversarial source review finds no stale V24.36.9 executable modules, hidden cap/roughness change, nested shell handoff, or missing prior evidence')

def _locked_pycache_regression():
    with tempfile.TemporaryDirectory(prefix='v243612_locked_pycache_') as td:
        t=Path(td); pc=t/'__pycache__';pc.mkdir();dummy=pc/'locked.pyc';dummy.write_bytes(b'locked')
        try:pc.chmod(0o555);dummy.chmod(0o444)
        except OSError:pass
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        q=subprocess.run([sys.executable,'-c',f"compile(open(r'{HERE/'v24_36_12_process_reference.py'}','rb').read(),r'{HERE/'v24_36_12_process_reference.py'}','exec');print('ok')"],cwd=t,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if q.returncode:raise AssertionError(q.stdout+'\\n'+q.stderr)
        if list(pc.glob('*.pyc'))!=[dummy]:raise AssertionError('locked __pycache__ contents changed')
    print('PASS host: non-writable/locked __pycache__ regression; syntax validation requires no package-tree bytecode writes')

def host():
    RECEIPT.unlink(missing_ok=True)
    if PREFLIGHT.is_file():subprocess.run([sys.executable,str(PREFLIGHT)],cwd=HERE,check=True)
    for p in HERE.glob('*.py'): compile(p.read_bytes(), str(p), 'exec', dont_inherit=True)
    assert ('py_'+'compile') not in Path(__file__).read_text(); print('PASS host: all runtime Python files compile in memory; verifier does not write __pycache__/pyc into the package tree')
    _release_runtime();_frozen_physics();_cap_conservation_and_fallback_static();_abi_memory_static();_actual_run_bin_regression();_cpu_microfacet_and_snell_regressions();_visible_normal_reference_regression();_qsca_event_mu_regressions();_v243611_bug_regressions();_path_budget();_surrogate();_host_smoke_chain();_campaign_gate();_locked_pycache_regression();_adversarial_source_review()

def gpu(samples_per_bin:int|None=None):
    cfg=vm.load_config();b=cfg['reciprocity_fallback_audit_v24_36_12'];out=HERE/'results';out.mkdir(exist_ok=True)
    print('INFO GPU: compiling frozen production CUDA plus appended reciprocity/fallback audit; no sediment transport will run.')
    t=time.perf_counter();r=drv.acquire_gpu(out,samples_per_bin,False);elapsed=time.perf_counter()-t
    q=subprocess.run([sys.executable,str(SUMMARY),str(out)],cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if q.returncode:raise RuntimeError('GPU acquisition completed but diagnostic integrity failed:\n'+q.stdout+'\n'+q.stderr)
    by=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_macro_tir_flux_by_bin.csv');pc=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_probability_conservation.csv');fix=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_fixed_microfacet_reversibility.csv');vis=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_microfacet_visibility_audit.csv');pairs=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_reciprocity_pairs.csv');res=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_reciprocity_residuals.csv');kp=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_complete_kernel_reciprocity_pairs.csv');mf=pd.read_csv(out/AUDIT_SUBDIR/f'{PREFIX}_macro_tir_reciprocal_flux.csv')
    assert set(by.material)=={'loess','kaolin'}
    for mat,nexp in {'loess':38,'kaolin':28}.items():
        assert by[(by.material==mat)&(by.qsca>=1)].bin_index.nunique()==nexp,(mat,len(by[by.material==mat]));assert kp[(kp.material==mat)&(kp.qsca>=1)].bin_index.nunique()==nexp,(mat,'complete kernel coverage')
    assert float(by.max_prefix_direction_l2.max())<=b['invariant_tolerances']['production_prefix_direction_l2'];assert float(by.prefix_rng_mismatch_fraction.max())==0.0;assert float(by.prefix_counter_mismatch_fraction.max())==0.0
    assert int(by.other_production_failure_count.sum())==0 and int(by.production_cap_hits.sum())>0
    assert float(np.abs(pc.count_closure_error).max())==0.0 and int(pc.numerical_geometry_failure_count.sum())==0
    assert bool(fix.passed.astype(str).str.lower().isin(['true','1']).all()) and len(vis)==len(b['visibility_incident_cosines'])
    assert {'accepted_rough_reflection','accepted_rough_escape'}.issubset(set(pairs.category.astype(str)))
    assert len(res)>0 and bool(res.passed_local_optics.astype(str).str.lower().isin(['true','1']).all());assert len(kp)>0 and len(mf)==2 and set(mf.material)=={'loess','kaolin'}
    meta=json.loads((out/AUDIT_SUBDIR/f'{PREFIX}_acquisition_summary.json').read_text());assert meta['particle_k_forced_zero'] is True and meta['audited_bins']=={'loess':38,'kaolin':28}
    receipt=finalize_metadata({'gpu_acquisition_complete':True,'production_core_cuda_sha256':sha_text(core.CUDA_SRC),'audit_cuda_sha256':sha_text(drv.AUDIT_CUDA+ck.COMPLETE_KERNEL_CUDA),'output_root':'results','materials':['loess','kaolin'],'samples_per_bin':int(samples_per_bin or b['samples_per_bin']),'process_correction_authorized':False});RECEIPT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(f'PASS GPU: real production+audit CUDA acquisition completed in {elapsed:.1f} s (audit compile {r["compile_seconds"]:.1f} s); both materials/all Qsca>=1 bins, frozen-prefix parity, conservation accounting, fixed-facet reciprocity, visibility reference and representative pair reconstruction and complete mixed-kernel reciprocity acquisition passed integrity; receipt written')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host-only',action='store_true');ap.add_argument('--samples-per-bin',type=int);a=ap.parse_args();host()
    if not a.host_only:gpu(a.samples_per_bin)
if __name__=='__main__':main()
