from __future__ import annotations
import math,time
import numpy as np
import pandas as pd
from v24_36_12_release import RELEASE

STATE_NAMES=('outside_ordinary','outside_nearcritical','tir_nearcritical','tir_grazing')

COMPLETE_KERNEL_CUDA=r'''
extern "C" {
__device__ int ck_state_bin(float mu,float muc){
    float d=0.15f; float hi=fminf(1.0f,muc+d), lo=fmaxf(0.0f,muc-d);
    if(mu>=hi) return 0; if(mu>=muc) return 1; if(mu>=lo) return 2; return 3;
}
__device__ void ck_add(unsigned long long* a,int i){atomicAdd(&a[i],1ULL);}
__global__ void complete_kernel_kernel(const int N,const float NM,const float NP,const float KP,const float ALPHA,const unsigned int SEED,
    unsigned long long* SRC, unsigned long long* REFL_TOTAL,unsigned long long* REFL_ACCEPT,unsigned long long* REFL_FALLBACK,
    unsigned long long* ESC_TOTAL,unsigned long long* ESC_ACCEPT,unsigned long long* ESC_FALLBACK,
    unsigned long long* ENTRY_TOTAL,unsigned long long* ENTRY_ACCEPT,unsigned long long* ENTRY_FALLBACK,
    double* CAND_MASS,double* REJ_MASS,double* REJ_TIR_MASS){
    int sid=(int)(blockDim.x*blockIdx.x+threadIdx.x); if(sid>=N) return;
    float muc=0.0f; if(NP>NM){float r=NM/NP; muc=sqrtf(fmaxf(0.0f,1.0f-r*r));}
    // Internal equilibrium-flux incidence: p(mu)=2mu, mu in [0,1].
    unsigned int s=SEED+(unsigned int)sid*74729u+17u; if(s==0)s=1u;
    float mu=sqrtf(fminf(fmaxf(rnd_uniform(&s),0.0f),0.99999994f)); float ph=2.0f*PI_F*rnd_uniform(&s); float st=sqrtf(fmaxf(0.0f,1.0f-mu*mu));
    float vx=st*cosf(ph),vy=st*sinf(ph),vz=mu; int a=ck_state_bin(mu,muc); ck_add(SRC,a);
    float mx=0,my=0,mz=1; beckmann_microfacet_normal(&s,ALPHA,0,0,1,vx,vy,vz,+1,&mx,&my,&mz);
    float cr=fminf(fmaxf(vx*mx+vy*my+vz*mz,0.0f),1.0f); float eta=NP/NM; int rough_tir=(eta*eta*fmaxf(0.0f,1.0f-cr*cr)>=1.0f);
    float tx=0,ty=0,tz=0; int rok=refract3(vx,vy,vz,-mx,-my,-mz,NP,NM,&tx,&ty,&tz); int compat=rok && tz>1e-7f;
    float Rr=rough_tir?1.0f:fresnel_R_complex(cr,NP,KP,NM,0.0f); double emass=rough_tir?0.0:(double)(1.0f-Rr);
    if(!rough_tir){atomicAdd(CAND_MASS,emass); if(!compat)atomicAdd(REJ_MASS,emass);}
    int fb=0; float fnx=mx,fny=my,fnz=mz; float tf_x=tx,tf_y=ty,tf_z=tz;
    if(!compat){fb=1;fnx=0;fny=0;fnz=1;rok=refract3(vx,vy,vz,0,0,-1,NP,NM,&tf_x,&tf_y,&tf_z);}
    float cf=fb?mu:cr; int final_tir=(eta*eta*fmaxf(0.0f,1.0f-cf*cf)>=1.0f); if(!rough_tir && !compat && final_tir)atomicAdd(REJ_TIR_MASS,emass);
    float Rf=final_tir?1.0f:fresnel_R_complex(cf,NP,KP,NM,0.0f); int transmit=0; if(!final_tir && rok){float u=rnd_uniform(&s); if(u>=Rf)transmit=1;}
    if(transmit){ck_add(ESC_TOTAL,a); if(fb)ck_add(ESC_FALLBACK,a);else ck_add(ESC_ACCEPT,a);} else {
        float rx,ry,rz;reflect3(vx,vy,vz,fnx,fny,fnz,&rx,&ry,&rz);int rfb=0;if(rz>=-1e-7f){reflect3(vx,vy,vz,0,0,1,&rx,&ry,&rz);rfb=1;}
        // Unit sphere, current point=(0,0,1). Reflected ray must point inward.
        float chord=-2.0f*rz; if(chord>1e-12f){float nx=chord*rx,ny=chord*ry,nz=1.0f+chord*rz;float nl=sqrtf(nx*nx+ny*ny+nz*nz);nx/=nl;ny/=nl;nz/=nl;float mu2=fminf(fmaxf(rx*nx+ry*ny+rz*nz,0.0f),1.0f);int b=ck_state_bin(mu2,muc);int q=a*4+b;ck_add(REFL_TOTAL,q);if(fb||rfb)ck_add(REFL_FALLBACK,q);else ck_add(REFL_ACCEPT,q);}
    }
    // Independent external equilibrium-flux incidence, for n^2-weighted transmission reciprocity.
    unsigned int e=SEED+0x9E3779B9u+(unsigned int)sid*11939u+31u; if(e==0)e=1u;
    float me=sqrtf(fminf(fmaxf(rnd_uniform(&e),0.0f),0.99999994f));float pe=2.0f*PI_F*rnd_uniform(&e);float se=sqrtf(fmaxf(0.0f,1.0f-me*me));
    float ex=se*cosf(pe),ey=se*sinf(pe),ez=-me;float emx=0,emy=0,emz=1;beckmann_microfacet_normal(&e,ALPHA,0,0,1,ex,ey,ez,-1,&emx,&emy,&emz);
    float ci=fminf(fmaxf(-(ex*emx+ey*emy+ez*emz),0.0f),1.0f);float ix=0,iy=0,iz=0;int iok=refract3(ex,ey,ez,emx,emy,emz,NM,NP,&ix,&iy,&iz);int icompat=iok && iz<-1e-7f;int ifb=0;if(!icompat){ifb=1;emx=0;emy=0;emz=1;ci=me;iok=refract3(ex,ey,ez,0,0,1,NM,NP,&ix,&iy,&iz);}float Ri=fresnel_R_complex(ci,NM,0.0f,NP,KP);if(iok && rnd_uniform(&e)>=Ri){float mui=fminf(fmaxf(-iz,0.0f),1.0f);int b=ck_state_bin(mui,muc);ck_add(ENTRY_TOTAL,b);if(ifb)ck_add(ENTRY_FALLBACK,b);else ck_add(ENTRY_ACCEPT,b);}
}
}
'''

def _state_label(i:int)->str:return STATE_NAMES[int(i)]

def compile_kernel(cp,core_cuda:str):
    t=time.perf_counter();m=cp.RawModule(code=core_cuda+COMPLETE_KERNEL_CUDA,options=('-std=c++11',));f=m.get_function('complete_kernel_kernel');cp.cuda.Stream.null.synchronize();return f,time.perf_counter()-t

def run_gpu_bin(cp,kernel,nm,np_,kp,alpha,samples,seed):
    n=int(samples);u=lambda k:cp.zeros(k,dtype=cp.uint64);d=lambda k:cp.zeros(k,dtype=cp.float64)
    src=u(4);rt=u(16);ra=u(16);rf=u(16);et=u(4);ea=u(4);ef=u(4);it=u(4);ia=u(4);iff=u(4);cm=d(1);rm=d(1);rtm=d(1)
    threads=128;blocks=(n+threads-1)//threads
    kernel((blocks,),(threads,),(np.int32(n),np.float32(nm),np.float32(np_),np.float32(kp),np.float32(alpha),np.uint32(seed),src,rt,ra,rf,et,ea,ef,it,ia,iff,cm,rm,rtm))
    cp.cuda.Stream.null.synchronize()
    return {k:cp.asnumpy(v) for k,v in {'src':src,'refl_total':rt,'refl_accept':ra,'refl_fallback':rf,'escape_total':et,'escape_accept':ea,'escape_fallback':ef,'entry_total':it,'entry_accept':ia,'entry_fallback':iff,'candidate_mass':cm,'rejected_mass':rm,'rejected_tir_mass':rtm}.items()}

def pair_rows(material,bin_index,qsca,event_weight,nm,np_,samples,out,rel_floor=0.01,z_limit=5.0):
    N=float(samples); rows=[]
    rt=out['refl_total'].reshape(4,4);ra=out['refl_accept'].reshape(4,4);rf=out['refl_fallback'].reshape(4,4)
    def add(kind,a,b,fc,rc,fa,ff,ra_,rf_,wf,wr,var):
        se=math.sqrt(max(var,0.0));absr=abs(wf-wr);rel=absr/max(abs(wf),abs(wr),1e-30);z=absr/se if se>0 else (0.0 if absr==0 else float('inf'));passed=bool(rel<=rel_floor or z<z_limit)
        rows.append({'release':RELEASE,'material':material,'bin_index':int(bin_index),'qsca':float(qsca),'event_rate_weight':float(event_weight),'pair_kind':kind,'state_a':a,'state_b':b,'forward_total_count':int(fc),'reverse_total_count':int(rc),'forward_accepted_rough_count':int(fa),'forward_fallback_count':int(ff),'reverse_accepted_rough_count':int(ra_),'reverse_fallback_count':int(rf_),'forward_weighted_flux':float(wf),'reverse_weighted_flux':float(wr),'absolute_residual':float(absr),'relative_residual':float(rel),'standard_error':float(se),'z_score':float(z),'engineering_relative_floor':float(rel_floor),'z_limit':float(z_limit),'passed':passed})
    # All off-diagonal internal reflection pairs; same n^2 equilibrium measure.
    for a in range(4):
        for b in range(a+1,4):
            fc,rc=int(rt[a,b]),int(rt[b,a]);wf=(np_**2)*fc/N;wr=(np_**2)*rc/N;var=(np_**4)*(fc+rc)/(N*N)
            add('internal_reflection',_state_label(a),_state_label(b),fc,rc,int(ra[a,b]),int(rf[a,b]),int(ra[b,a]),int(rf[b,a]),wf,wr,var)
    # Aggregate outside <-> macro-TIR is the primary decision pair.
    O=(0,1);T=(2,3);fc=sum(int(rt[a,b]) for a in O for b in T);rc=sum(int(rt[a,b]) for a in T for b in O);fa=sum(int(ra[a,b]) for a in O for b in T);ff=sum(int(rf[a,b]) for a in O for b in T);racc=sum(int(ra[a,b]) for a in T for b in O);rfall=sum(int(rf[a,b]) for a in T for b in O);wf=(np_**2)*fc/N;wr=(np_**2)*rc/N;var=(np_**4)*(fc+rc)/(N*N)
    add('macro_tir_primary','outside_macro_tir','inside_macro_tir',fc,rc,fa,ff,racc,rfall,wf,wr,var)
    # Cross-interface transmission reciprocity. Each side is sampled from normalized cosine flux; multiply by n^2.
    for b in range(4):
        fc=int(out['escape_total'][b]);rc=int(out['entry_total'][b]);wf=(np_**2)*fc/N;wr=(nm**2)*rc/N
        pf=fc/N;pr=rc/N;var=(np_**4)*pf*(1-pf)/N+(nm**4)*pr*(1-pr)/N
        add('cross_interface',_state_label(b),'external_medium',fc,rc,int(out['escape_accept'][b]),int(out['escape_fallback'][b]),int(out['entry_accept'][b]),int(out['entry_fallback'][b]),wf,wr,var)
    return rows

def flux_decomposition_rows(material,bin_index,qsca,event_weight,samples,out):
    rt=out['refl_total'].reshape(4,4);ra=out['refl_accept'].reshape(4,4);rf=out['refl_fallback'].reshape(4,4);O=(0,1);T=(2,3)
    rows=[]
    for name,A,B in [('outside_to_macro_tir',O,T),('macro_tir_to_outside',T,O)]:
        total=sum(int(rt[a,b]) for a in A for b in B);acc=sum(int(ra[a,b]) for a in A for b in B);fb=sum(int(rf[a,b]) for a in A for b in B)
        rows.append({'release':RELEASE,'material':material,'bin_index':int(bin_index),'qsca':float(qsca),'event_rate_weight':float(event_weight),'transition':name,'total_count':total,'accepted_rough_count':acc,'fallback_count':fb,'accepted_rough_fraction':acc/total if total else np.nan,'fallback_fraction':fb/total if total else np.nan})
    cand=float(out['candidate_mass'][0]);rej=float(out['rejected_mass'][0]);rtir=float(out['rejected_tir_mass'][0])
    rows.append({'release':RELEASE,'material':material,'bin_index':int(bin_index),'qsca':float(qsca),'event_rate_weight':float(event_weight),'transition':'candidate_escape_probability_mass','total_count':int(samples),'accepted_rough_count':0,'fallback_count':0,'accepted_rough_fraction':(cand-rej)/cand if cand else np.nan,'fallback_fraction':rej/cand if cand else np.nan})
    rows.append({'release':RELEASE,'material':material,'bin_index':int(bin_index),'qsca':float(qsca),'event_rate_weight':float(event_weight),'transition':'rejected_candidate_escape_to_geom_tir','total_count':int(samples),'accepted_rough_count':0,'fallback_count':0,'accepted_rough_fraction':np.nan,'fallback_fraction':rtir/rej if rej else np.nan})
    return rows

def summarize_pairs(pair_df:pd.DataFrame):
    rows=[]
    if pair_df.empty:return rows
    for (mat,kind,a,b),g in pair_df.groupby(['material','pair_kind','state_a','state_b'],sort=True):
        w=g.event_rate_weight.to_numpy(float);w=w/w.sum() if w.sum()>0 else np.ones(len(g))/len(g);fw=float(np.sum(w*g.forward_weighted_flux));rw=float(np.sum(w*g.reverse_weighted_flux));se=float(math.sqrt(np.sum((w*g.standard_error.to_numpy(float))**2)));absr=abs(fw-rw);rel=absr/max(abs(fw),abs(rw),1e-30);z=absr/se if se>0 else (0.0 if absr==0 else float('inf'));passed=bool(rel<=0.01 or z<5.0)
        rows.append({'release':RELEASE,'material':mat,'pair_kind':kind,'state_a':a,'state_b':b,'event_weighted_forward_flux':fw,'event_weighted_reverse_flux':rw,'absolute_residual':absr,'relative_residual':rel,'combined_standard_error':se,'z_score':z,'passed':passed})
    return rows
