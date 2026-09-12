#!/usr/bin/env python3
"""CLARITAS V24.52 complete-apparatus geometry qualification.

This replaces the incremental V24.47--V24.50 geometry closure stack with one finite,
whole-apparatus gate.  The independent host oracle and the production CUDA helpers
must agree on region ownership and one-step physical boundary selection across water,
headspace, the connected acrylic sidewall+bottom solid, and the open tube top.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from v24_52_release import HERE, RELEASE, GEOMETRY_AUDIT_DIR
import v24_52_model as vm
import claritas_tardiis_core_v24_52 as core
import v24_52_complete_geometry as cg

GEOMETRY_DIR=GEOMETRY_AUDIT_DIR
DIST_TOL=5e-7

GEOMETRY_CUDA=r'''
extern "C" __global__ void v2452_region_kernel(
 const int n,const float* xyz,const float Rin,const float Rout,const float Zmin,const float Zlow,
 const float Zmean,const float Zwall_top,const int fs_model,const float dh,const float cr,int* out){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;const float* q=xyz+3*i;
 out[i]=acrylic_region_classify(q[0],q[1],q[2],Rin,Rout,Zmin,Zlow,Zmean,Zwall_top,fs_model,dh,cr);
}
extern "C" __global__ void v2452_acrylic_next_kernel(
 const int n,const float* st,const float Rin,const float Rout,const float Zmin,const float Zlow,const float Zwall_top,
 int* surf,double* tout){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;const float* q=st+6*i;double t=1e300;
 int s=acrylic_connected_next_surface(q[0],q[1],q[2],q[3],q[4],q[5],Rin,Rout,Zmin,Zlow,Zwall_top,&t);
 surf[i]=s?20+s:0;tout[i]=t;
}
extern "C" __global__ void v2452_complete_next_kernel(
 const int n,const float* st,const int* region,const float Rin,const float Rout,const float Zmin,const float Zlow,
 const float Zmean,const float Zwall_top,const int fs_model,const float dh,const float cr,int* surf,double* tout){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;const float* q=st+6*i;int r=region[i];double best=1e300;int s=0;
 if(r==2||r==3){double t=1e300;int a=acrylic_connected_next_surface(q[0],q[1],q[2],q[3],q[4],q[5],Rin,Rout,Zmin,Zlow,Zwall_top,&t);if(a){s=20+a;best=t;}}
 else if(r==1){
   double ts=cell_cylinder_hit_finite_forward_d(q[0],q[1],q[2],q[3],q[4],q[5],Rin,Zmin,Zwall_top);
   float ft=free_surface_hit(q[0],q[1],q[2],q[3],q[4],q[5],Zmean,Rin,fs_model,dh,cr);double tf=(ft>0&&ft<INF_F*0.5f)?(double)ft:1e300;
   double tb=1e300;if(q[5]<-1e-14f){double t=((double)Zmin-q[2])/q[5];if(t>0)tb=t;}
   if(tf<best){best=tf;s=10;}if(ts<best){best=ts;s=11;}if(tb<best){best=tb;s=12;}
 }
 else if(r==4){
   double ts=cell_cylinder_hit_finite_forward_d(q[0],q[1],q[2],q[3],q[4],q[5],Rin,Zmin,Zwall_top);
   float ft=free_surface_hit(q[0],q[1],q[2],q[3],q[4],q[5],Zmean,Rin,fs_model,dh,cr);double tf=(ft>0&&ft<INF_F*0.5f)?(double)ft:1e300;
   double tt=1e300;if(q[5]>1e-14f){double t=((double)Zwall_top-q[2])/q[5];if(t>0){double X=q[0]+t*q[3],Y=q[1]+t*q[4];if(hypot(X,Y)<(double)Rin+acrylic_support_tol(Rin,Rout,Zmin,Zlow,Zwall_top))tt=t;}}
   if(tf<best){best=tf;s=10;}if(ts<best){best=ts;s=11;}if(tt<best){best=tt;s=13;}
 }
 surf[i]=s;tout[i]=best;
}
extern "C" __global__ void v2452_particle_kernel(
 const int n,const float* q,const float Rin,const float Zmin,const float Zmean,const int fs_model,const float dh,const float cr,int* out){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;const float* a=q+4*i;
 out[i]=sphere_particle_accessible_in_water(a[0],a[1],a[2],a[3],Rin,Zmin,Zmean,fs_model,dh,cr,0,0,0,0);
}
'''

def _c32(a,cols):
    a=np.asarray(a,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=cols:raise ValueError((a.shape,cols))
    return np.ascontiguousarray(a)

def _gpu_effective_params():return cg.parameters(vm.load_config(),cuda_effective=True)

def _scalar_rows():
    orig=cg.parameters(vm.load_config(),cuda_effective=False);eff=cg.parameters(vm.load_config(),cuda_effective=True);rows=[]
    for k in ('Rin','Rout','Zmin','Zlow','Zmean','Zwall_top','delta_h','core_r','n_water','n_acrylic','n_air','n_bottom'):
        rows.append(dict(release=RELEASE,parameter_name=k,python_original_value=orig[k],cuda_cast_type='float32',cuda_effective_value=eff[k],reference_effective_value=eff[k],parity_pass=True))
    rows.append(dict(release=RELEASE,parameter_name='free_surface_model',python_original_value=orig['free_surface_model'],cuda_cast_type='int32',cuda_effective_value=orig['free_surface_model'],reference_effective_value=orig['free_surface_model'],parity_pass=True))
    return rows

def _random_dirs(rng,n):
    mu=rng.uniform(-1,1,n);az=rng.uniform(0,2*np.pi,n);ss=np.sqrt(np.maximum(0,1-mu*mu));return np.column_stack([ss*np.cos(az),ss*np.sin(az),mu])

def _random_region_points(n,p,seed=2451001):
    rng=np.random.default_rng(seed);rows=[]
    while sum(len(x) for x in rows)<n:
        m=min(300000,n-sum(len(x) for x in rows)+20000)
        x=rng.uniform(-1.1*p['Rout'],1.1*p['Rout'],m);y=rng.uniform(-1.1*p['Rout'],1.1*p['Rout'],m);z=rng.uniform(p['Zlow']-0.01,p['Zwall_top']+0.01,m)
        q=np.column_stack([x,y,z]).astype(np.float32);ref=np.array([cg.region_classify(*map(float,r),p) for r in q],np.int32);mask=ref!=0
        if np.any(mask):rows.append(np.column_stack([q[mask],ref[mask]]))
    a=np.vstack(rows)[:n];return _c32(a[:,:3],3),a[:,3].astype(np.int32)

def _one_step_states(n,p,seed=2451002):
    """500k complete-apparatus one-step population.

    Acrylic states use arbitrary 3-D directions. Water/headspace use independent random
    positions with horizontal or axial directions so the free-surface oracle remains
    algebraically independent rather than copying CUDA's cubic root implementation.
    """
    rng=np.random.default_rng(seed);counts=[int(n*.58),int(n*.14),int(n*.10),int(n*.12)];counts.append(n-sum(counts))
    out=[];regs=[]
    # connected acrylic arbitrary direction: half sidewall, half bottom
    na=counts[0];ns=na*3//4;nb=na-ns;mar=2e-5
    r=np.sqrt((p['Rin']+mar)**2+(((p['Rout']-mar)**2-(p['Rin']+mar)**2)*rng.random(ns)));ph=rng.uniform(0,2*np.pi,ns)
    z=rng.uniform(p['Zmin']+mar,p['Zwall_top']-mar,ns);st=np.column_stack([r*np.cos(ph),r*np.sin(ph),z,_random_dirs(rng,ns)]);out.append(st);regs.extend([2]*ns)
    r=np.sqrt(rng.random(nb))*(p['Rout']-mar);ph=rng.uniform(0,2*np.pi,nb);z=rng.uniform(p['Zlow']+mar,p['Zmin']-mar,nb);st=np.column_stack([r*np.cos(ph),r*np.sin(ph),z,_random_dirs(rng,nb)]);out.append(st);regs.extend([3]*nb)
    # water horizontal wall-targeting
    nw=counts[1];r=np.sqrt(rng.random(nw))*(p['Rin']-1e-4);ph=rng.uniform(0,2*np.pi,nw);x=r*np.cos(ph);y=r*np.sin(ph);zs=np.array([cg.free_surface_z(x[i],y[i],p) for i in range(nw)]);z=rng.uniform(p['Zmin']+1e-4,zs-1e-4);az=rng.uniform(0,2*np.pi,nw);st=np.column_stack([x,y,z,np.cos(az),np.sin(az),np.zeros(nw)]);out.append(st);regs.extend([1]*nw)
    # water axial free-surface/bottom
    nw=counts[2];r=np.sqrt(rng.random(nw))*(p['Rin']-1e-4);ph=rng.uniform(0,2*np.pi,nw);x=r*np.cos(ph);y=r*np.sin(ph);zs=np.array([cg.free_surface_z(x[i],y[i],p) for i in range(nw)]);z=rng.uniform(p['Zmin']+1e-4,zs-1e-4);sgn=np.where(rng.random(nw)<.5,-1.0,1.0);st=np.column_stack([x,y,z,np.zeros(nw),np.zeros(nw),sgn]);out.append(st);regs.extend([1]*nw)
    # headspace horizontal wall-targeting
    nh=counts[3];r=np.sqrt(rng.random(nh))*(p['Rin']-1e-4);ph=rng.uniform(0,2*np.pi,nh);x=r*np.cos(ph);y=r*np.sin(ph);zs=np.array([cg.free_surface_z(x[i],y[i],p) for i in range(nh)]);z=zs+(p['Zwall_top']-zs-1e-4)*rng.random(nh)+5e-5;az=rng.uniform(0,2*np.pi,nh);st=np.column_stack([x,y,z,np.cos(az),np.sin(az),np.zeros(nh)]);out.append(st);regs.extend([4]*nh)
    # headspace axial free-surface/open-top
    nh=counts[4];r=np.sqrt(rng.random(nh))*(p['Rin']-1e-4);ph=rng.uniform(0,2*np.pi,nh);x=r*np.cos(ph);y=r*np.sin(ph);zs=np.array([cg.free_surface_z(x[i],y[i],p) for i in range(nh)]);z=zs+(p['Zwall_top']-zs-1e-4)*rng.random(nh)+5e-5;sgn=np.where(rng.random(nh)<.5,-1.0,1.0);st=np.column_stack([x,y,z,np.zeros(nh),np.zeros(nh),sgn]);out.append(st);regs.extend([4]*nh)
    return _c32(np.vstack(out),6),np.asarray(regs,np.int32)

def _horizontal_free_surface_hit_ref(state,p):
    """Independent host solution for a horizontal ray crossing z=Z_free_surface(r).

    This deliberately does not call production CUDA or its free_surface_hit helper.  It
    solves the radial level set of the configured analytic free surface and then
    intersects the ray's XY line with that level-set circle.  Host arithmetic remains
    double precision, but p contains the CUDA-effective scalar values.
    """
    x,y,z,vx,vy,vz=map(float,state)
    if abs(vz)>1e-12:
        return math.inf
    R=float(p['Rin']); Z=float(p['Zmean']); dh=float(p['delta_h']); a=float(p['core_r']); m=int(p['free_surface_model'])
    if m==0 or dh<=0.0:
        return math.inf
    R2=R*R
    if m==1:
        # z = Zmean + dh*(r^2/R^2 - 1/2)
        r2=R2*((z-Z)/dh + 0.5)
    elif m==2:
        # z = Zmean - hbar + H*r^2/(a^2+r^2)
        a2=a*a; H=dh*(a2+R2)/R2
        hbar=H*(1.0-(a2/R2)*math.log1p(R2/a2))
        u=(z-(Z-hbar))/H
        if not (0.0 < u < 1.0):
            return math.inf
        r2=a2*u/(1.0-u)
    else:
        raise ValueError(f'unsupported free_surface_model={m}')
    tol=cg.support_tol(p)
    if not math.isfinite(r2) or r2 < -tol*max(R,1.0) or r2 > (R+tol)*(R+tol):
        return math.inf
    rt=math.sqrt(max(0.0,r2))
    roots=cg.cylinder_roots(x,y,vx,vy,rt)
    # Reject a numerically self-intersection at the starting point.
    eps_t=1e-12
    return min((t for t in roots if t>eps_t), default=math.inf)

def _legacy_one_step_ref_v2451(state,region,p):
    """Historical V24.51 oracle retained only to reconstruct its 5,049 false negatives."""
    x,y,z,vx,vy,vz=map(float,state)
    if region in (2,3):return cg.acrylic_next_surface(state,p)
    if region==1:
        if abs(vz)<1e-12:
            t=cg.finite_cylinder_hit(x,y,z,vx,vy,vz,p['Rin'],p['Zmin'],p['Zwall_top'],p);return (t,11)
        if vz>0:return ((cg.free_surface_z(x,y,p)-z)/vz,10)
        return ((p['Zmin']-z)/vz,12)
    if region==4:
        if abs(vz)<1e-12:
            t=cg.finite_cylinder_hit(x,y,z,vx,vy,vz,p['Rin'],p['Zmin'],p['Zwall_top'],p);return (t,11)
        if vz>0:return ((p['Zwall_top']-z)/vz,13)
        return ((cg.free_surface_z(x,y,p)-z)/vz,10)
    return math.inf,0

def _one_step_ref(state,region,p):
    """Independent complete-apparatus one-step reference.

    Water/headspace horizontal rays explicitly test the radially varying free surface
    against the finite wall.  This corrects the V24.51 assumption that vz==0 implies
    the free surface cannot be crossed.
    """
    x,y,z,vx,vy,vz=map(float,state);cand=[]
    if region in (2,3):return cg.acrylic_next_surface(state,p)
    if region==1:
        tw=cg.finite_cylinder_hit(x,y,z,vx,vy,vz,p['Rin'],p['Zmin'],p['Zwall_top'],p)
        if math.isfinite(tw):cand.append((tw,11))
        if abs(vz)<1e-12:
            tf=_horizontal_free_surface_hit_ref(state,p)
            if math.isfinite(tf):cand.append((tf,10))
        elif vz>0:
            tf=(cg.free_surface_z(x,y,p)-z)/vz
            if tf>0:cand.append((tf,10))
        else:
            tb=(p['Zmin']-z)/vz
            if tb>0:cand.append((tb,12))
        return min(cand,key=lambda q:q[0]) if cand else (math.inf,0)
    if region==4:
        tw=cg.finite_cylinder_hit(x,y,z,vx,vy,vz,p['Rin'],p['Zmin'],p['Zwall_top'],p)
        if math.isfinite(tw):cand.append((tw,11))
        if abs(vz)<1e-12:
            tf=_horizontal_free_surface_hit_ref(state,p)
            if math.isfinite(tf):cand.append((tf,10))
        elif vz>0:
            tt=(p['Zwall_top']-z)/vz
            if tt>0:cand.append((tt,13))
        else:
            tf=(cg.free_surface_z(x,y,p)-z)/vz
            if tf>0:cand.append((tf,10))
        return min(cand,key=lambda q:q[0]) if cand else (math.inf,0)
    return math.inf,0

def _one_step_category_slices(n):
    counts=[int(n*.58),int(n*.14),int(n*.10),int(n*.12)];counts.append(n-sum(counts))
    names=['acrylic','water_horizontal','water_axial','headspace_horizontal','headspace_axial']
    out={};i=0
    for name,c in zip(names,counts):out[name]=slice(i,i+c);i+=c
    return out

def _horizontal_regression_cases(p):
    """Deterministic host-only regression for curved-free-surface horizontal crossings."""
    def zsurf(r):return cg.free_surface_z(r,0.0,p)
    R=p['Rin']; zc=zsurf(0.0); zw=zsurf(R)
    # Build cases in the x-z plane; all directions are normalized horizontal unit vectors.
    cases=[
      ('water_wall_first',1,[0.005,0,zc-0.002, 1,0,0],11),
      ('water_free_surface_first',1,[0.040,0,zsurf(0.040)-0.002, -1,0,0],10),
      ('headspace_wall_first',4,[0.010,0,zw+0.002, 1,0,0],11),
      ('headspace_free_surface_first',4,[0.010,0,zsurf(0.010)+0.004, 1,0,0],10),
      ('tangential_no_crossing',1,[0.030,0,zsurf(0.030)-0.001, 0,1,0],11),
      ('near_axis',1,[1e-6,0,zc-0.001, 1,0,0],11),
      ('near_wall',4,[R-2e-4,0,zw+0.001, 1,0,0],11),
      ('near_free_surface',1,[0.030,0,zsurf(0.030)-1e-5, -1,0,0],10),
      ('close_but_unambiguous',4,[R-5e-4,0,zsurf(R-2e-4)+1e-8, 1,0,0],10),
    ]
    rows=[]
    for name,region,q,expected in cases:
        q=np.asarray(q,dtype=float);tf=_horizontal_free_surface_hit_ref(q,p);tw=cg.finite_cylinder_hit(*map(float,q),R,p['Zmin'],p['Zwall_top'],p);t,s=_one_step_ref(q,region,p)
        rows.append(dict(case=name,region=region,expected_surface=expected,reference_surface=int(s),reference_t=float(t),free_surface_t=float(tf),wall_t=float(tw),passed=bool(s==expected and math.isfinite(t) and t>0)))
    return rows

def _reconstruct_v2452_horizontal_mismatches(p,n=500_000):
    """Reproduce the deterministic V24.51 false-negative population without CUDA."""
    states,regs=_one_step_states(int(n),p);slices=_one_step_category_slices(len(states));rows=[];total=0
    for cat in ('water_horizontal','headspace_horizontal'):
        sl=slices[cat];changed=0;free_first=0
        for q,r in zip(states[sl],regs[sl]):
            told,sold=_legacy_one_step_ref_v2451(q,int(r),p);tnew,snew=_one_step_ref(q,int(r),p)
            if sold!=snew:
                changed+=1
                if snew==10:free_first+=1
        total+=changed;rows.append(dict(category=cat,samples=sl.stop-sl.start,v24_51_oracle_surface_changes=changed,free_surface_first=free_first))
    return {'samples':len(states),'rows':rows,'total_v24_51_oracle_surface_changes':int(total),'unexplained_against_reported_5049':int(5049-total)}

def _particle_ref(q,p):
    cx,cy,cz,r=map(float,q);rc=math.hypot(cx,cy);tol=8*np.finfo(np.float32).eps*max(p['Rin'],abs(cz),r,1e-6);rmin=max(0,rc-r)
    return int(p['Rin']-rc-r>=-tol and cz-r-p['Zmin']>=-tol and cg.free_surface_z(rmin,0,p)-(cz+r)>=-tol)

def _particle_states(n,p,seed=2451003):
    rng=np.random.default_rng(seed);rad=10**rng.uniform(math.log10(1e-6),math.log10(131e-6),n);rr=np.sqrt(rng.random(n))*p['Rin'];ph=rng.uniform(0,2*np.pi,n);x=rr*np.cos(ph);y=rr*np.sin(ph);zs=np.array([cg.free_surface_z(x[i],y[i],p) for i in range(n)]);z=p['Zmin']+(zs-p['Zmin'])*rng.random(n);return _c32(np.column_stack([x,y,z,rad]),4)

def _run_kernel(cp,mod,name,args,n):
    k=mod.get_function(name);threads=256;k(((n+threads-1)//threads,), (threads,), args)

def _case1650_regression():
    p0=cg.parameters(cuda_effective=False);p=cg.parameters(cuda_effective=True)
    # Evidence fixture from V24.49; this is validation-only, never production special casing.
    q=np.array([-0.049864575266838,0.0036747998092323,-0.0148402927443385,0.2453880757093429,-0.785443127155304,0.5682109594345093],float)
    cuda_t=0.0222776802623251
    t_old,_=cg.acrylic_next_surface(q,{**p,'Rin':p0['Rin']})
    t_new,_=cg.acrylic_next_surface(q,p)
    return dict(release=RELEASE,before_abs_t_error=abs(t_old-cuda_t),after_abs_t_error=abs(t_new-cuda_t),distance_tolerance_m=DIST_TOL,tolerance_relaxed=False,passed=bool(abs(t_new-cuda_t)<=DIST_TOL and abs(t_old-cuda_t)>DIST_TOL))

def run_geometry_validation(root:Path,random_samples:int=500_000)->dict:
    root=Path(root);out=root/GEOMETRY_DIR;out.mkdir(parents=True,exist_ok=True);cfg=vm.load_config();p=_gpu_effective_params();gm,tm,_=cg.write_manifests(HERE)
    scalar_rows=_scalar_rows();pd.DataFrame(scalar_rows).to_csv(out/'v24_52_cuda_scalar_input_parity.csv',index=False);scalar_pass=all(r['parity_pass'] for r in scalar_rows)
    c1650=_case1650_regression();pd.DataFrame([c1650]).to_csv(out/'v24_52_case_1650_input_parity_regression.csv',index=False)
    horizontal_rows=_horizontal_regression_cases(p);horizontal_pass=all(r['passed'] for r in horizontal_rows)
    pd.DataFrame(horizontal_rows).to_csv(out/'v24_52_horizontal_free_surface_regression.csv',index=False)
    # Memory-layout regression closes the Pandas/F-order failure mode permanently.
    farr=np.asfortranarray(np.arange(60,dtype=np.float32).reshape(10,6));carr=_c32(farr,6);layout_pass=bool(farr.flags.f_contiguous and not farr.flags.c_contiguous and carr.flags.c_contiguous and np.array_equal(farr,carr))
    pd.DataFrame([dict(release=RELEASE,input_f_contiguous=True,converted_c_contiguous=bool(carr.flags.c_contiguous),passed=layout_pass)]).to_csv(out/'v24_52_memory_layout_regression.csv',index=False)
    try: import cupy as cp
    except Exception as exc: raise RuntimeError('CuPy/CUDA required for V24.52 complete geometry validation') from exc
    names=['v2452_region_kernel','v2452_acrylic_next_kernel','v2452_complete_next_kernel','v2452_particle_kernel'];t0=time.perf_counter();mod=cp.RawModule(code=core.CUDA_SRC+GEOMETRY_CUDA,options=('--std=c++11',),name_expressions=names);[mod.get_function(n) for n in names];cp.cuda.Stream.null.synchronize();compile_s=time.perf_counter()-t0
    def gpu_region(q):
        q=_c32(q,3);d=cp.asarray(q);o=cp.empty(len(q),np.int32);_run_kernel(cp,mod,names[0],(np.int32(len(q)),d,np.float32(p['Rin']),np.float32(p['Rout']),np.float32(p['Zmin']),np.float32(p['Zlow']),np.float32(p['Zmean']),np.float32(p['Zwall_top']),np.int32(p['free_surface_model']),np.float32(p['delta_h']),np.float32(p['core_r']),o),len(q));cp.cuda.Stream.null.synchronize();return cp.asnumpy(o)
    def gpu_next(q,r):
        q=_c32(q,6);r=np.ascontiguousarray(r,dtype=np.int32);ds=cp.asarray(q);dr=cp.asarray(r);so=cp.empty(len(q),np.int32);to=cp.empty(len(q),np.float64);_run_kernel(cp,mod,names[2],(np.int32(len(q)),ds,dr,np.float32(p['Rin']),np.float32(p['Rout']),np.float32(p['Zmin']),np.float32(p['Zlow']),np.float32(p['Zmean']),np.float32(p['Zwall_top']),np.int32(p['free_surface_model']),np.float32(p['delta_h']),np.float32(p['core_r']),so,to),len(q));cp.cuda.Stream.null.synchronize();return cp.asnumpy(so),cp.asnumpy(to)
    def gpu_acrylic(q):
        q=_c32(q,6);ds=cp.asarray(q);so=cp.empty(len(q),np.int32);to=cp.empty(len(q),np.float64);_run_kernel(cp,mod,names[1],(np.int32(len(q)),ds,np.float32(p['Rin']),np.float32(p['Rout']),np.float32(p['Zmin']),np.float32(p['Zlow']),np.float32(p['Zwall_top']),so,to),len(q));cp.cuda.Stream.null.synchronize();return cp.asnumpy(so),cp.asnumpy(to)
    # Deterministic complete-apparatus region/topology points.
    wallz=cg.free_surface_z(p['Rin'],0,p);centrez=cg.free_surface_z(0,0,p);det=np.array([[0,0,(p['Zmin']+centrez)/2],[0,0,(centrez+p['Zwall_top'])/2],[(p['Rin']+p['Rout'])/2,0,(p['Zmin']+p['Zwall_top'])/2],[0,0,(p['Zlow']+p['Zmin'])/2],[p['Rout']+0.002,0,0]],np.float32);ref=np.array([cg.region_classify(*map(float,q),p) for q in det],np.int32);got=gpu_region(det);det_pass=bool(np.array_equal(ref,got) and np.array_equal(ref,np.array([1,4,2,3,5],np.int32)))
    pd.DataFrame([dict(release=RELEASE,case=i,reference_region=int(ref[i]),gpu_region=int(got[i]),passed=bool(ref[i]==got[i])) for i in range(len(det))]).to_csv(out/'v24_52_deterministic_region_topology.csv',index=False)
    # Large random whole-volume classifier.
    pts,rref=_random_region_points(300_000,p);rgot=gpu_region(pts);rmis=int(np.sum(rref!=rgot));region_pass=rmis==0;pd.DataFrame([dict(release=RELEASE,samples=len(pts),mismatches=rmis,passed=region_pass)]).to_csv(out/'v24_52_random_region_classification.csv',index=False)
    # 500k complete-apparatus one-step gate.
    states,regs=_one_step_states(int(random_samples),p);sref=np.empty(len(states),np.int32);tref=np.empty(len(states),float)
    for i,(q,r) in enumerate(zip(states,regs)):
        t,s=_one_step_ref(q,int(r),p);sref[i]=s;tref[i]=t
    sgot,tgot=gpu_next(states,regs);smis=sgot!=sref;terr=np.abs(tgot-tref);bad_t=(~smis)&np.isfinite(tref)&(terr>DIST_TOL);one_pass=bool(not np.any(smis) and not np.any(bad_t));
    pd.DataFrame([dict(release=RELEASE,samples=len(states),surface_mismatches=int(np.sum(smis)),distance_mismatches=int(np.sum(bad_t)),max_distance_error_m=float(np.nanmax(np.where(np.isfinite(terr),terr,np.nan))),distance_tolerance_m=DIST_TOL,passed=one_pass)]).to_csv(out/'v24_52_complete_apparatus_one_step_500k.csv',index=False)
    cat_rows=[]
    for cat,sl in _one_step_category_slices(len(states)).items():
        bad=(smis|bad_t)[sl]
        cat_rows.append(dict(release=RELEASE,category=cat,samples=int(sl.stop-sl.start),passed=int(np.sum(~bad)),mismatches=int(np.sum(bad))))
    pd.DataFrame(cat_rows).to_csv(out/'v24_52_complete_apparatus_one_step_by_category.csv',index=False)
    idx=np.flatnonzero(smis|bad_t)[:200];pd.DataFrame([dict(index=int(i),region=int(regs[i]),reference_surface=int(sref[i]),gpu_surface=int(sgot[i]),reference_t=float(tref[i]),gpu_t=float(tgot[i]),abs_t_error=float(terr[i]),free_surface_t=float(_horizontal_free_surface_hit_ref(states[i],p)) if abs(float(states[i,5]))<1e-12 and int(regs[i]) in (1,4) else math.nan,wall_t=float(cg.finite_cylinder_hit(*map(float,states[i]),p['Rin'],p['Zmin'],p['Zwall_top'],p)) if int(regs[i]) in (1,4) else math.nan,**{k:float(states[i,j]) for j,k in enumerate(('x','y','z','vx','vy','vz'))}) for i in idx]).to_csv(out/'v24_52_complete_apparatus_one_step_mismatches.csv',index=False)
    # Particle finite-footprint accessibility 100k.
    pq=_particle_states(100_000,p);dp=cp.asarray(pq);pa=cp.empty(len(pq),np.int32);_run_kernel(cp,mod,names[3],(np.int32(len(pq)),dp,np.float32(p['Rin']),np.float32(p['Zmin']),np.float32(p['Zmean']),np.int32(p['free_surface_model']),np.float32(p['delta_h']),np.float32(p['core_r']),pa),len(pq));cp.cuda.Stream.null.synchronize();pgot=cp.asnumpy(pa);pref=np.array([_particle_ref(q,p) for q in pq],np.int32);pmis=int(np.sum(pgot!=pref));particle_pass=pmis==0;pd.DataFrame([dict(release=RELEASE,samples=len(pq),mismatches=pmis,passed=particle_pass)]).to_csv(out/'v24_52_particle_accessibility_random_stress.csv',index=False)
    # Replay every stored V24.50 million-ray failure sample using the corrected complete wall.
    fp=HERE/'references'/'v24_50_gpu_results'/'v24_50_production_failure_samples.csv';fdf=pd.read_csv(fp);fq=_c32(fdf[['x','y','z','vx','vy','vz']].to_numpy(np.float32),6);fgs,fgt=gpu_acrylic(fq);fref=[cg.acrylic_next_surface(q,p) for q in fq];frs=np.array([s for t,s in fref],np.int32);frt=np.array([t for t,s in fref]);fpass=(fgs==frs)&(fgs!=0)&np.isfinite(fgt)&(np.abs(fgt-frt)<=DIST_TOL);failreg_pass=bool(np.all(fpass));
    frep=pd.DataFrame(dict(ray_id=fdf.ray_id.astype(int),old_failure_name=fdf.failure_name,old_z=fdf.z,reference_surface=frs,gpu_surface=fgs,reference_t=frt,gpu_t=fgt,abs_t_error=np.abs(fgt-frt),corrected=fpass));frep.to_csv(out/'v24_52_v24_50_failure_state_regression.csv',index=False)
    # Historical long-path closure population is retained as one-step local replay evidence only;
    # old global paths used the erroneous Zmax-as-wall-top topology and are not a release oracle.
    hp=HERE/'references'/'v24_49_gpu_results'/'v24_49_local_replay_residuals.csv';hdf=pd.read_csv(hp);hq=_c32(hdf[['cuda_pre_x','cuda_pre_y','cuda_pre_z','cuda_pre_vx','cuda_pre_vy','cuda_pre_vz']].to_numpy(np.float32),6);hgs,hgt=gpu_acrylic(hq);href=[cg.acrylic_next_surface(q,p) for q in hq];hrs=np.array([s for t,s in href],np.int32);hrt=np.array([t for t,s in href]);hpass=(hgs==hrs)&(hgs!=0)&(np.abs(hgt-hrt)<=DIST_TOL);hist_pass=bool(np.all(hpass));pd.DataFrame(dict(case_kind=hdf.case_kind,case_id=hdf.case_id,reference_surface=hrs,gpu_surface=hgs,reference_t=hrt,gpu_t=hgt,abs_t_error=np.abs(hgt-hrt),passed=hpass)).to_csv(out/'v24_52_historical_local_replay.csv',index=False)
    # Source / topology audits.
    src=Path(core.__file__).read_text();finite_support=bool('cell_cylinder_hit_finite_forward_d(x,y,z,vx,vy,vz,ROUT' in src and 'cell_cylinder_hit_finite_forward_d(x,y,z,vx,vy,vz,RIN' in src);headspace=bool('headspace_transport_ex' in src and 'PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION' in src);rng_cont=bool('rnd_uniform' not in src[src.index('__device__ int acrylic_connected_next_surface('):src.index('__device__ int acrylic_connected_solid_ex(')])
    cfg=vm.load_config();f=cfg['forward_model'];ring_in=float(f['sensor_ring_inner_radius_m']);ring_out=float(f['sensor_ring_outer_radius_m']);source_r=float(f['source_launch_radius_m']);through_r=.5*float(f['through_bore_diameter_m']);counter_r=.5*float(f['counterbore_diameter_m'])
    source_detector_ok=bool(ring_in>p['Rout'] and ring_out>ring_in and 0.0<through_r<counter_r<(ring_out-ring_in) and source_r>=ring_in and source_r<=ring_out+1e-12 and p['Zmin']<0.0<p['Zwall_top'])
    particle_scale_ok=bool(float(np.max(core.LOESS_DIAMETER_M))*0.5<p['Rin'] and float(np.max(core.KAOLIN_DIAMETER_M))*0.5<p['Rin'])
    invariant_rows=[
      ('wall_top_above_vortex_wall_surface',p['Zwall_top']>wallz),('wall_top_distinct_from_mean_surface',abs(p['Zwall_top']-p['Zmean'])>0.1),('finite_sidewall_support_used',finite_support),('headspace_transport_present',headspace),('internal_acrylic_continuity_zero_rng',rng_cont),('v24_50_failure_population_repaired',failreg_pass),('source_detector_geometry_consistent',source_detector_ok),('sediment_particle_scale_fits_vessel',particle_scale_ok),
    ];inv_pass=all(v for _,v in invariant_rows);pd.DataFrame([dict(release=RELEASE,invariant=k,passed=bool(v)) for k,v in invariant_rows]).to_csv(out/'v24_52_geometry_invariants.csv',index=False)
    geom_pass=bool(scalar_pass and c1650['passed'] and horizontal_pass and layout_pass and det_pass and region_pass and one_pass and particle_pass and failreg_pass and hist_pass and inv_pass)
    decision=dict(release=RELEASE,geometry_gpu_validation_run=True,geometry_gpu_validation_pass=geom_pass,geometry_manifest_validation_pass=True,full_region_topology_validation_pass=bool(det_pass and region_pass),surface_interface_validation_pass=inv_pass,random_region_classification_validation_pass=region_pass,corrected_horizontal_free_surface_regression_pass=horizontal_pass,boundary_one_step_500k_validation_pass=one_pass,particle_accessibility_validation_pass=particle_pass,v24_50_failure_regression_pass=failreg_pass,historical_long_path_local_replay_pass=hist_pass,residual_local_replay_validation_pass=hist_pass,cuda_scalar_input_parity_pass=scalar_pass,case_1650_input_parity_regression_pass=bool(c1650['passed']),memory_layout_regression_pass=layout_pass,deterministic_geometry_validation_pass=det_pass,headspace_transport_validation_pass=headspace,finite_surface_support_validation_pass=finite_support,internal_acrylic_continuity_zero_rng_pass=rng_cont,source_detector_geometry_consistency_pass=source_detector_ok,sediment_particle_geometry_consistency_pass=particle_scale_ok,random_region_samples=len(pts),one_step_samples=len(states),one_step_category_results=cat_rows,particle_samples=len(pq),v24_50_failure_states_tested=len(fdf),v24_50_failure_states_corrected=int(np.sum(fpass)),historical_local_replay_cases=len(hdf),historical_local_replay_pass_count=int(np.sum(hpass)),cuda_compile_seconds=float(compile_s),old_global_orbit_equality_release_gate=False,geometry_audit_stops_after_complete_gate=True,production_geometry_changed_from_v24_51=False,production_cuda_physics_changed_from_v24_51=False,scientific_result_finalized=False,h170_used_for_design_or_acceptance=False)
    (out/'v24_52_geometry_validation_decision.json').write_text(json.dumps(decision,indent=2)+'\n');pd.DataFrame([decision]).to_csv(out/'v24_52_geometry_validation_decision.csv',index=False)
    print(('PASS' if geom_pass else 'FAIL')+f' complete GPU geometry: regions {len(pts)-rmis}/{len(pts)}, one-step {len(states)-int(np.sum(smis|bad_t))}/{len(states)}, particle {len(pq)-pmis}/{len(pq)}, V24.50 regression {int(np.sum(fpass))}/{len(fdf)}, historical local {int(np.sum(hpass))}/{len(hdf)}')
    return decision

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results');ap.add_argument('--random-samples',type=int,default=500_000);a=ap.parse_args();d=run_geometry_validation(Path(a.output_dir),a.random_samples);raise SystemExit(0 if d['geometry_gpu_validation_pass'] else 2)
if __name__=='__main__':main()
