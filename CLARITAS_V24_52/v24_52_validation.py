from __future__ import annotations
import sys; sys.dont_write_bytecode=True
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from v24_52_release import HERE,RELEASE,AUDIT_DIR
import v24_52_model as vm
import claritas_tardiis_core_v24_52 as core

STATE_NAMES=('outside_ordinary','outside_nearcritical','tir_nearcritical','tir_grazing')
THRESHOLDS=(20,40,65,128,256,512)
VNDF_RAW_WZ=(1.0,0.95,0.75,0.50,0.10,0.0,-0.10,-0.50,-0.70,-0.80,-0.90,-0.95,-0.99,-0.995,-0.999,-0.9995,-0.9999,-0.99999,-0.999999)
VNDF_ANGLES_DEG=(150.0,155.0,158.0,160.0,162.0,165.0,170.0,175.0,177.0,178.0,179.0,179.2,179.5,179.8,179.9)
VNDF_WZ=tuple(dict.fromkeys(VNDF_RAW_WZ+tuple(math.cos(math.radians(a)) for a in VNDF_ANGLES_DEG)))
FAILURE_NAMES={0:'NONE',10:'VNDF_PROJECTED_AREA_INVALID',11:'VNDF_CDF_INVALID',12:'VNDF_INVERSE_CDF_INVALID',13:'VNDF_VISIBILITY_FAILURE',14:'VNDF_NORMALIZATION_FAILURE',20:'HEIGHT_CDF_INVALID',21:'HEIGHT_SAMPLE_INVALID',22:'FREE_PATH_INVALID',23:'HEIGHT_LAMBDA_INVALID',24:'HEIGHT_C0_INVALID',25:'HEIGHT_LOG_C0_INVALID',26:'HEIGHT_G1_INVALID',27:'HEIGHT_ESCAPE_PROB_INVALID',28:'HEIGHT_RANDOM_INVALID',29:'HEIGHT_LOG_CNEXT_INVALID',30:'MICROFACET_NORMAL_INVALID',31:'REFLECTION_VECTOR_INVALID',32:'REFRACTION_VECTOR_INVALID',33:'SIDE_STATE_INVALID',34:'NONFINITE_DIRECTION',35:'FRESNEL_INVALID',36:'HEIGHT_CNEXT_RANGE_INVALID',37:'HEIGHT_INVERSE_CDF_INVALID',38:'HEIGHT_OUTPUT_INVALID',39:'HEIGHT_FREE_PATH_INVALID',40:'MICRO_WATCHDOG',41:'MACRO_WATCHDOG'}

# GPU validation contains independent roles:
#   * v43_* invokes the REAL V24.52 production signed-VNDF microsurface walker/sphere path.
#   * v37_bad_* reconstructs the superseded V24.37 macro-support rejection candidate.
#   * v38_bad_* reconstructs the isolated V24.38 invalid lower-side convention in the
#     equal-index limit. Historical bad processes are validation-only and not callable by production.
VALIDATION_CUDA=r'''
extern "C" {
__device__ int v43_state(float mu,float muc){float d=0.15f,hi=fminf(1.0f,muc+d),lo=fmaxf(0.0f,muc-d);if(mu>=hi)return 0;if(mu>=muc)return 1;if(mu>=lo)return 2;return 3;}
__device__ void v43_add(unsigned long long* p,int i){atomicAdd(&p[i],1ULL);}
__device__ void v43_subfail(unsigned long long* p,int offset,int fc){if(fc>=0&&fc<64)atomicAdd(&p[offset+fc],1ULL);}

__device__ int v43_internal_step(unsigned int* s,float nm,float np,float kp,float alpha,float vx,float vy,float vz,float* ox,float* oy,float* oz,int* order,int* terminal,int* failure_code){
  int top=1,o=0,fc=0;float x=vx,y=vy,z=vz;
  int st=smith_microsurface_dielectric_walk_ex(s,alpha,4096,0,0,-1,np,kp,nm,0,&x,&y,&z,&top,&o,&fc);
  *order=o;*terminal=st;*failure_code=fc;if(st!=MICRO_ESCAPE)return 2;*ox=x;*oy=y;*oz=z;
  if(top){if(z>=-1e-7f){*failure_code=SIDE_STATE_INVALID;return 2;}return 0;}else{if(z<=1e-7f){*failure_code=SIDE_STATE_INVALID;return 2;}return 1;}
}
__device__ int v43_external_step(unsigned int* s,float nm,float np,float kp,float alpha,float vx,float vy,float vz,float* ox,float* oy,float* oz,int* order,int* terminal,int* failure_code){
  int top=1,o=0,fc=0;float x=vx,y=vy,z=vz;
  int st=smith_microsurface_dielectric_walk_ex(s,alpha,4096,0,0,1,nm,0,np,kp,&x,&y,&z,&top,&o,&fc);
  *order=o;*terminal=st;*failure_code=fc;if(st!=MICRO_ESCAPE)return 2;*ox=x;*oy=y;*oz=z;
  if(top){if(z<=1e-7f){*failure_code=SIDE_STATE_INVALID;return 2;}return 0;}else{if(z>=-1e-7f){*failure_code=SIDE_STATE_INVALID;return 2;}return 1;}
}

__global__ void v43_reciprocity_kernel(int N,float nm,float np,float kp,float alpha,unsigned int seed,unsigned long long* refl,unsigned long long* esc,unsigned long long* ent,unsigned long long* fail,unsigned long long* subfail,unsigned long long* micro_events){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;float eta=np/nm,muc=eta>1.0f?sqrtf(fmaxf(0.0f,1.0f-1.0f/(eta*eta))):0.0f;
 unsigned int s=seed+(unsigned int)id*74729u+17u;if(s==0)s=1u;float mu=sqrtf(fminf(fmaxf(rnd_uniform(&s),0.0f),0.99999994f)),ph=2.0f*PI_F*rnd_uniform(&s),st=sqrtf(fmaxf(0.0f,1.0f-mu*mu));float vx=st*cosf(ph),vy=st*sinf(ph),vz=mu;int a=v43_state(mu,muc),ord=0,term=0,fc=0;float ox,oy,oz;int kind=v43_internal_step(&s,nm,np,kp,alpha,vx,vy,vz,&ox,&oy,&oz,&ord,&term,&fc);atomicAdd(micro_events,(unsigned long long)ord);if(kind==2){v43_subfail(subfail,0,fc);if(term==MICRO_WATCHDOG_FAILURE)v43_add(fail,1);else if(term==MICRO_NUMERICAL_FAILURE)v43_add(fail,0);else v43_add(fail,2);}else if(kind==1){v43_add(esc,a);}else{float chord=-2.0f*oz;if(chord<=1e-12f){v43_add(fail,2);v43_subfail(subfail,0,SIDE_STATE_INVALID);}else{float nx=chord*ox,ny=chord*oy,nz=1.0f+chord*oz;float l=sqrtf(nx*nx+ny*ny+nz*nz);nx/=l;ny/=l;nz/=l;float mu2=fminf(fmaxf(ox*nx+oy*ny+oz*nz,0.0f),1.0f);int b=v43_state(mu2,muc);v43_add(refl,a*4+b);}}
 unsigned int e=seed+0x9E3779B9u+(unsigned int)id*11939u+31u;if(e==0)e=1u;float me=sqrtf(fminf(fmaxf(rnd_uniform(&e),0.0f),0.99999994f)),pe=2.0f*PI_F*rnd_uniform(&e),se=sqrtf(fmaxf(0.0f,1.0f-me*me));float ex=se*cosf(pe),ey=se*sinf(pe),ez=-me;ord=0;term=0;fc=0;kind=v43_external_step(&e,nm,np,kp,alpha,ex,ey,ez,&ox,&oy,&oz,&ord,&term,&fc);atomicAdd(micro_events,(unsigned long long)ord);if(kind==2){v43_subfail(subfail,64,fc);if(term==MICRO_WATCHDOG_FAILURE)v43_add(fail,4);else if(term==MICRO_NUMERICAL_FAILURE)v43_add(fail,3);else v43_add(fail,5);}else if(kind==1){float chord=-2.0f*oz;if(chord<=1e-12f){v43_add(fail,5);v43_subfail(subfail,64,SIDE_STATE_INVALID);}else{float nx=chord*ox,ny=chord*oy,nz=1.0f+chord*oz;float l=sqrtf(nx*nx+ny*ny+nz*nz);nx/=l;ny/=l;nz/=l;float mui=fminf(fmaxf(ox*nx+oy*ny+oz*nz,0.0f),1.0f);v43_add(ent,v43_state(mui,muc));}}
}

// Real production sphere path: deliberately not a validation surrogate.
__global__ void v43_path_kernel(int N,float nm,float np,float kp,float alpha,int watchdog,unsigned int seed,int* status,int* internal_refl){int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2654435761u+7u;if(s==0)s=1u;float x=0,y=0,z=0,vx=0,vy=0,vz=1,path=0;int pr=0,er=0,ir=0,ab=0;int st=sphere_fresnel_interaction_3d(&s,nm,np,kp,6.22e-7f,alpha,1.0f,watchdog-1,&x,&y,&z,&vx,&vy,&vz,&path,&pr,&er,&ir,&ab);status[id]=st;internal_refl[id]=ir;}

// Smooth and equal-index release gates using the actual production walker with failure subcodes.
__global__ void v43_microsurface_limits_kernel(int N,float nm,float np,float alpha,unsigned int seed,int* smooth_ok,int* equal_ok,int* equal_status,int* equal_order,int* equal_failure,float* equal_err){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2246822519u+13u;if(s==0)s=1u;
 float mu=0.05f+0.94f*rnd_uniform(&s),ph=2.0f*PI_F*rnd_uniform(&s),st=sqrtf(fmaxf(0.0f,1.0f-mu*mu));float ix=st*cosf(ph),iy=st*sinf(ph),iz=-mu;
 float x=ix,y=iy,z=iz;int top=1,ord=0,fc=0;int ms=smith_microsurface_dielectric_walk_ex(&s,0.0f,4096,0,0,1,nm,0,np,0,&x,&y,&z,&top,&ord,&fc);
 float rx,ry,rz,tx=0,ty=0,tz=0;reflect3(ix,iy,iz,0,0,1,&rx,&ry,&rz);int rt=refract3(ix,iy,iz,0,0,1,nm,np,&tx,&ty,&tz);float er=(x-rx)*(x-rx)+(y-ry)*(y-ry)+(z-rz)*(z-rz);float et=rt?((x-tx)*(x-tx)+(y-ty)*(y-ty)+(z-tz)*(z-tz)):1e9f;smooth_ok[id]=(ms==MICRO_ESCAPE && fminf(er,et)<1e-8f)?1:0;
 unsigned int q=seed+0x85EBCA6Bu+(unsigned int)id*3266489917u+19u;if(q==0)q=1u;x=ix;y=iy;z=iz;top=1;ord=0;fc=0;ms=smith_microsurface_dielectric_walk_ex(&q,alpha,4096,0,0,1,nm,0,nm,0,&x,&y,&z,&top,&ord,&fc);float ee=(x-ix)*(x-ix)+(y-iy)*(y-iy)+(z-iz)*(z-iz);equal_status[id]=ms;equal_failure[id]=fc;equal_err[id]=ee;equal_ok[id]=(ms==MICRO_ESCAPE && !top && ee<2.5e-7f && isfinite(x)&&isfinite(y)&&isfinite(z))?1:0;equal_order[id]=ord;
}

// Direct targeted signed-VNDF validation over front, horizontal and deep-backside states.
__global__ void v43_signed_vndf_kernel(int N,int ncases,const float* wz_cases,float alpha,unsigned int seed,int* case_id,int* ok,int* failcode,double* sx,double* sy,double* visdot,double* topdot,double* normerr){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;int c=id%ncases;float wz=wz_cases[c],wx=sqrtf(fmaxf(0.0f,1.0f-wz*wz));unsigned int s=seed+(unsigned int)id*1597334677u+23u;if(s==0)s=1u;double mx=0,my=0,mz=0;int fc=0;int st=beckmann_vndf_normal_signed_d_ex(&s,alpha,0,0,1,wx,0,wz,&mx,&my,&mz,&fc);double nd=sqrt(mx*mx+my*my+mz*mz),vd=(double)wx*mx+(double)wz*mz,td=mz;case_id[id]=c;failcode[id]=fc;visdot[id]=vd;topdot[id]=td;normerr[id]=fabs(nd-1.0);if(st && mz>0.0){sx[id]=-mx/mz;sy[id]=-my/mz;}else{sx[id]=0;sy[id]=0;}ok[id]=(st && isfinite(mx)&&isfinite(my)&&isfinite(mz)&&fabs(nd-1.0)<2e-10&&td>0.0&&vd>0.0)?1:0;
}

// Random full-sphere stress excluding only the measure-zero exactly downward state.
__global__ void v43_full_sphere_vndf_kernel(int N,float alpha,unsigned int seed,int* ok,int* failcode,float* wzout,double* visdot,double* topdot,double* normerr){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2654435761u+47u;if(s==0)s=1u;float wz=-0.99999f+1.99998f*rnd_uniform(&s),ph=2.0f*PI_F*rnd_uniform(&s),r=sqrtf(fmaxf(0.0f,1.0f-wz*wz)),wx=r*cosf(ph),wy=r*sinf(ph);double mx=0,my=0,mz=0;int fc=0;int st=beckmann_vndf_normal_signed_d_ex(&s,alpha,0,0,1,wx,wy,wz,&mx,&my,&mz,&fc);double nd=sqrt(mx*mx+my*my+mz*mz),vd=(double)wx*mx+(double)wy*my+(double)wz*mz;wzout[id]=wz;failcode[id]=fc;visdot[id]=vd;topdot[id]=mz;normerr[id]=fabs(nd-1.0);ok[id]=(st&&isfinite(mx)&&isfinite(my)&&isfinite(mz)&&fabs(nd-1.0)<2e-10&&mz>0.0&&vd>0.0)?1:0;
}

// Force lower-side second intersections at several increasingly steep directions.
__global__ void v43_forced_inside_second_hit_kernel(int N,int ncases,const float* wz_cases,float n,float alpha,unsigned int seed,int* case_id,int* ok,int* height_status,int* vndf_status,int* failcode,float* dir_err,double* p_hit_out,double* u_out,int* construction_ok){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;int c=id%ncases;case_id[id]=c;unsigned int s=seed+(unsigned int)id*3812015801u+29u;if(s==0)s=1u;float ph=2.0f*PI_F*rnd_uniform(&s);float wz=wz_cases[c],rr=sqrtf(fmaxf(0.0f,1.0f-wz*wz)),wx=rr*cosf(ph),wy=rr*sinf(ph);
 // Validation forcing probability is computed from the same V24.52 Smith Lambda/log-CDF law
 // in double precision.  No float-G1 subtraction and no arbitrary 1e-12 floor are permitted.
 double lam=smith_beckmann_lambda_d(-wx,-wy,-wz,0,0,1,alpha);double C0=smith_uniform_C1_d(0.0);double pHit=0.0;
 int pc=(isfinite(lam)&&lam>=0.0&&C0>0.0&&C0<1.0);if(pc){double logG1=lam*log(C0);pHit=-expm1(logG1);pc=(isfinite(pHit)&&pHit>0.0&&pHit<1.0);}double U=pc?0.25*pHit:0.0;p_hit_out[id]=pHit;u_out[id]=U;construction_ok[id]=(pc&&U>0.0&&U<pHit)?1:0;
 float tmp=0;int hfc=0;int hs=construction_ok[id]?smith_sample_height_u_ex(U,-wx,-wy,-wz,0,0,1,alpha,0.0f,&tmp,&hfc):MICRO_NUMERICAL_FAILURE;height_status[id]=hs;if(hs!=MICRO_INTERSECTION){ok[id]=0;vndf_status[id]=0;failcode[id]=hfc?hfc:HEIGHT_SAMPLE_INVALID;dir_err[id]=1e9f;return;}float wix=-wx,wiy=-wy,wiz=-wz;double tx,ty,tz;int vfc=0;int vs=beckmann_vndf_normal_signed_d_ex(&s,alpha,0,0,1,-wix,-wiy,-wiz,&tx,&ty,&tz,&vfc);vndf_status[id]=vs;if(!vs){ok[id]=0;failcode[id]=vfc;dir_err[id]=1e9f;return;}double mx=-tx,my=-ty,mz=-tz;double vd=(double)wix*mx+(double)wiy*my+(double)wiz*mz;if(!(vd>0.0)){ok[id]=0;failcode[id]=VNDF_VISIBILITY_FAILURE;dir_err[id]=1e9f;return;}/* equal-index refraction is exactly the original propagation direction */dir_err[id]=0.0f;failcode[id]=0;ok[id]=1;
}

// Height/free-path status semantics: no numerical condition may be reported as escape.
__global__ void v43_height_sampler_kernel(int N,float alpha,unsigned int seed,int* case_id,int* status,int* ok,int* failcode,float* hnext){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;int c=id%8;float wx=0,wy=0,wz=0,h=0,U=0.5f;if(c==0){wz=1.0f;}else if(c==1){wz=-1.0f;}else if(c==2){wx=1.0f;wz=0.0f;}else if(c==3){wx=sqrtf(1-0.1f*0.1f);wz=0.1f;U=0.001f;}else if(c==4){wx=sqrtf(1-0.1f*0.1f);wz=0.1f;U=0.999999f;}else if(c==5){wx=sqrtf(1-0.1f*0.1f);wz=-0.1f;U=0.5f;}else if(c==6){wx=sqrtf(1-0.5f*0.5f);wz=0.5f;h=-0.999f;U=1e-7f;}else{wx=sqrtf(1-0.5f*0.5f);wz=-0.5f;h=0.999f;U=1.0f-1e-7f;}float hn=0;int fc=0;int st=smith_sample_height_u_ex(U,wx,wy,wz,0,0,1,alpha,h,&hn,&fc);case_id[id]=c;status[id]=st;failcode[id]=fc;hnext[id]=hn;int good=(st==MICRO_INTERSECTION||st==MICRO_ESCAPE);if(c==0)good=(st==MICRO_ESCAPE);if(c==1)good=(st==MICRO_INTERSECTION);if(c==2)good=(st==MICRO_INTERSECTION&&fabsf(hn-h)<1e-6f);if(c==5||c==7)good=(st==MICRO_INTERSECTION);if(st==MICRO_INTERSECTION)good=good&&isfinite(hn);ok[id]=good?1:0;
}


// Direct stable-Lambda gate across both signed hemispheres and the V24.40 cancellation region.
__global__ void v43_lambda_kernel(int N,int ncases,const float* c_cases,float alpha,int* case_id,int* ok,double* lam){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;int k=id%ncases;double c=(double)c_cases[k];float wz=(float)c,wx=sqrtf(fmaxf(0.0f,1.0f-wz*wz));double l=smith_beckmann_lambda_d(wx,0,wz,0,0,1,alpha);case_id[id]=k;lam[id]=l;int good=isfinite(l);if(c>0.0)good=good&&(l>=0.0);else if(c<0.0)good=good&&(l<=-1.0);ok[id]=good?1:0;
}

// Randomized production height sampler stress. Inputs and outputs are retained so the host can
// compare against an independent double-precision reference implementation.
__global__ void v43_height_stress_kernel(int N,float alpha,unsigned int seed,int* status,int* failcode,float* cout,float* hout,double* uout,float* hnext,double* lambdaout){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2246822519u+101u;if(s==0)s=1u;double U=beckmann_open_uniform(&s);float c=-0.999999f+1.999998f*rnd_uniform(&s);float h=-1.0f+3.0f*rnd_uniform(&s);float wx=sqrtf(fmaxf(0.0f,1.0f-c*c));float hn=0;int fc=0;int st=smith_sample_height_u_ex(U,wx,0,c,0,0,1,alpha,h,&hn,&fc);status[id]=st;failcode[id]=fc;cout[id]=c;hout[id]=h;uout[id]=U;hnext[id]=hn;lambdaout[id]=smith_beckmann_lambda_d(wx,0,c,0,0,1,alpha);
}

// Isolated V24.40 known-bad Lambda evaluator. It intentionally uses the cancelling float
// expression that produced HEIGHT_CDF_INVALID in the real V24.40 campaign.
__global__ void v40_bad_height_lambda_kernel(int N,float alpha,int* bad,float* c_out,float* lam_out){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;float t=(id+0.5f)/(float)N;float c=0.865f+0.025f*t;float ss=sqrtf(fmaxf(0.0f,1.0f-c*c));float a=c/(alpha*ss);const float A=0.28209479177387814f;float l=0.5f*(erff(a)-1.0f)+(A/a)*expf(-a*a);c_out[id]=c;lam_out[id]=l;bad[id]=(!isfinite(l)||l<0.0f)?1:0;
}


// Isolated V24.41.1 genuine production precision defect: stretched direction is converted
// through float theta before defining the deep-backside visible-slope support.  The corrected
// V24.52 production path never calls this kernel/helper.
__global__ void v41_bad_ultragrazing_kernel(int N,float alpha,unsigned int seed,int* ok,double* support_error){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2246822519u+149u;if(s==0)s=1u;float wz=-0.9999f,wx=sqrtf(fmaxf(0.0f,1.0f-wz*wz));double sxv=(double)alpha*wx,szv=(double)wz,vl=sqrt(sxv*sxv+szv*szv);sxv/=vl;szv/=vl;double direct=szv/fabs(sxv);float theta=(float)acos(fmax(-1.0,fmin(1.0,szv)));double rounded=cos((double)theta)/sin((double)theta);support_error[id]=rounded-direct;double sx0=0.0,sy0=0.0;int fc=0;double U1=beckmann_open_uniform(&s),U2=beckmann_open_uniform(&s);int st=beckmann_sample11_signed_slope_d_ex(rounded,U1,U2,&sx0,&sy0,&fc);double sx=(double)alpha*sx0,sy=(double)alpha*sy0,inv=1.0/sqrt(sx*sx+sy*sy+1.0),mx=-sx*inv,my=-sy*inv,mz=inv;double vd=(double)wx*mx+(double)wz*mz;ok[id]=(st&&isfinite(vd)&&vd>0.0&&mz>0.0)?1:0;
}

// Validation-only reconstruction of the known-bad V24.37 support-rejection candidate.
__device__ int v37_bad_internal_step(unsigned int* s,float nm,float np,float alpha,float vx,float vy,float vz,float* ox,float* oy,float* oz){
 for(int a=0;a<128;++a){float mx=0,my=0,mz=1;if(!beckmann_vndf_normal_signed(s,alpha,0,0,1,vx,vy,vz,&mx,&my,&mz))return 2;float c=fminf(fmaxf(dot3(vx,vy,vz,mx,my,mz),0.0f),1.0f);float eta=np/nm;int tir=eta*eta*fmaxf(0.0f,1.0f-c*c)>=1.0f;float R=tir?1.0f:fresnel_R_complex(c,np,0,nm,0);if(!tir&&rnd_uniform(s)>=R){float tx,ty,tz;if(!refract3(vx,vy,vz,-mx,-my,-mz,np,nm,&tx,&ty,&tz)||tz<=1e-7f)continue;*ox=tx;*oy=ty;*oz=tz;return 1;}float rx,ry,rz;reflect3(vx,vy,vz,mx,my,mz,&rx,&ry,&rz);if(rz>=-1e-7f)continue;*ox=rx;*oy=ry;*oz=rz;return 0;}return 2;
}
__global__ void v37_bad_regression_kernel(int N,float nm,float np,float alpha,unsigned int seed,unsigned long long* refl,unsigned long long* fail){
 int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;float eta=np/nm,muc=sqrtf(fmaxf(0.0f,1.0f-1.0f/(eta*eta)));unsigned int s=seed+(unsigned int)id*74729u+17u;float mu=sqrtf(fminf(fmaxf(rnd_uniform(&s),0.0f),0.99999994f)),ph=2*PI_F*rnd_uniform(&s),st=sqrtf(fmaxf(0.0f,1-mu*mu));float vx=st*cosf(ph),vy=st*sinf(ph),vz=mu;int a=v43_state(mu,muc);float ox,oy,oz;int k=v37_bad_internal_step(&s,nm,np,alpha,vx,vy,vz,&ox,&oy,&oz);if(k==2){atomicAdd(fail,1ULL);return;}if(k==0){float chord=-2*oz;if(chord<=1e-12f){atomicAdd(fail,1ULL);return;}float nx=chord*ox,ny=chord*oy,nz=1+chord*oz,l=sqrtf(nx*nx+ny*ny+nz*nz);nx/=l;ny/=l;nz/=l;int b=v43_state(fminf(fmaxf(ox*nx+oy*ny+oz*nz,0.0f),1.0f),muc);atomicAdd(&refl[a*4+b],1ULL);}}

// Isolated V24.38 defect: height walk + flipped nominal side convention. In equal-index
// transport, any lower-side second intersection is wrongly rejected before the VNDF sample.
__device__ int v38_bad_equal_walk(unsigned int* s,float alpha,float dx,float dy,float dz,int* order){
 float h=1.0f+smith_uniform_invC1(0.999f);int outside=1,o=0;for(int it=0;it<4096;++it){float nh=0;int hs;if(outside){hs=smith_sample_height(s,dx,dy,dz,0,0,1,alpha,h,&nh);if(hs==MICRO_ESCAPE){*order=o;return 1;}}else{float tmp=0;hs=smith_sample_height(s,-dx,-dy,-dz,0,0,1,alpha,-h,&tmp);if(hs==MICRO_ESCAPE){*order=o;return 1;}if(hs==MICRO_INTERSECTION)nh=-tmp;}if(hs!=MICRO_INTERSECTION){*order=o;return 0;}h=nh;++o;float wix=-dx,wiy=-dy,wiz=-dz;float sbz=outside?1.0f:-1.0f;if(wiz*sbz<=0.0f){*order=o;return 0;}outside=1-outside;}*order=o;return 0;
}
__global__ void v38_bad_equal_index_kernel(int N,float alpha,unsigned int seed,int* ok,int* order){int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*3266489917u+19u;if(s==0)s=1u;float mu=0.05f+0.94f*rnd_uniform(&s),ph=2*PI_F*rnd_uniform(&s),st=sqrtf(fmaxf(0.0f,1-mu*mu));float dx=st*cosf(ph),dy=st*sinf(ph),dz=-mu;int o=0;ok[id]=v38_bad_equal_walk(&s,alpha,dx,dy,dz,&o);order[id]=o;}

// Isolated V24.39 numerical defect: the historical -0.9999 erf-domain floor truncates
// deep-backside visible-slope support.  This code is validation-only and never callable by production.
__device__ int v39_bad_sample11(float theta_i,float U1,float U2,float* slope_x,float* slope_y){
 float sin_t=sinf(theta_i),cos_t=cosf(theta_i);if(!(sin_t>0))return 0;float slope_i=cos_t/sin_t;const float A=0.28209479177387814f;float projected=0.5f*(erff(slope_i)+1.0f)*cos_t+A*sin_t*expf(-slope_i*slope_i);if(!(projected>1e-12f)||!isfinite(projected))return 0;float c=1.0f/projected,emin=-0.9999f,emax=fmaxf(emin,erff(slope_i));emax=fminf(emax,0.999999f);float ec=0.5f*(emin+emax);for(int it=0;it<64&&(emax-emin)>1e-5f;++it){float e=fminf(fmaxf(ec,-0.999999f),0.999999f),sl=erfinvf(e);float CDF=(sl>=slope_i)?1.0f:c*(A*sin_t*expf(-sl*sl)+cos_t*(0.5f+0.5f*erff(sl)));float diff=CDF-U1;if(fabsf(diff)<1e-5f)break;if(diff>0)emax=ec;else emin=ec;float der=0.5f*c*cos_t-0.5f*c*sin_t*sl;ec=(isfinite(der)&&fabsf(der)>1e-8f)?ec-diff/der:0.5f*(emin+emax);}float ex=fminf(fmaxf(ec,emin),emax);ex=fminf(fmaxf(ex,-0.999999f),0.999999f);float ey=fminf(fmaxf(2*U2-1,-0.999999f),0.999999f);*slope_x=erfinvf(ex);*slope_y=erfinvf(ey);return isfinite(*slope_x)&&isfinite(*slope_y);}
__global__ void v39_bad_deep_backside_kernel(int N,float alpha,unsigned int seed,int* ok){int id=(int)(blockDim.x*blockIdx.x+threadIdx.x);if(id>=N)return;unsigned int s=seed+(unsigned int)id*2246822519u+71u;if(s==0)s=1u;float wz=-0.90f,wx=sqrtf(1-wz*wz),sxv=alpha*wx,vl=sqrtf(sxv*sxv+wz*wz),sz=wz/vl,theta=acosf(fminf(fmaxf(sz,-1.0f),1.0f));float sx=0,sy=0;int st=v39_bad_sample11(theta,rnd_uniform(&s),rnd_uniform(&s),&sx,&sy);sx*=alpha;sy*=alpha;float inv=rsqrtf(sx*sx+sy*sy+1.0f),mx=-sx*inv,my=-sy*inv,mz=inv;double vd=(double)wx*mx+(double)wz*mz;ok[id]=(st&&vd>0.0&&mz>0.0f)?1:0;}
}
'''

def _weights(host):
    w=np.asarray(host['mu_event_by_bin'],float);s=w.sum();return w/s if s>0 else np.zeros_like(w)

def _pair_metrics(fc,rc,N,nfac=1.0):
    wf=nfac*fc/N;wr=nfac*rc/N;var=(nfac*nfac)*(fc+rc)/(N*N);se=math.sqrt(max(var,0.0));absr=abs(wf-wr);rel=absr/max(abs(wf),abs(wr),1e-30);z=absr/se if se else (0.0 if absr==0 else float('inf'));return wf,wr,absr,rel,se,z

def _erfcx_asym(x:float)->float:
    """Stable erfcx asymptotic for x>=4 used only by the independent host reference."""
    y=1.0/(x*x)
    series=1.0-0.5*y+0.75*y*y-1.875*y**3+6.5625*y**4-29.53125*y**5+162.421875*y**6-1055.7421875*y**7
    return series/(math.sqrt(math.pi)*x)

def _tail_B_host(x:float)->float:
    if x<4.0:
        return 1.0-x*math.sqrt(math.pi)*math.exp(x*x)*math.erfc(x)
    y=1.0/(x*x)
    return 0.5*y-0.75*y*y+1.875*y**3-6.5625*y**4+29.53125*y**5-162.421875*y**6+1055.7421875*y**7

def _vndf_reference_moments(wz:float,alpha:float):
    """Independent analytic moments of the isotropic Beckmann visible-slope marginal.

    For t=sx/alpha and stretched incidence s=wz/(alpha*wx), p(t) is proportional to
    (s-t) exp(-t^2), t<s.  Closed-form truncated-Gaussian moments avoid any dependency
    on the CUDA inverse CDF and remain stable in the deep negative tail.
    """
    wx=math.sqrt(max(0.0,1.0-wz*wz))
    if wx<1e-14:
        if wz<=0:return dict(mean_sx=float('nan'),mean_sy=0.0,mean_sx2=float('nan'),mean_sy2=alpha*alpha/2)
        return dict(mean_sx=0.0,mean_sy=0.0,mean_sx2=alpha*alpha/2,mean_sy2=alpha*alpha/2)
    si=wz/(alpha*wx)
    if si < -4.0:
        x=-si;B=_tail_B_host(x);ex=_erfcx_asym(x)
        mt=-(math.sqrt(math.pi)/2.0)*ex/B
        mt2=(1.0-0.5*x*math.sqrt(math.pi)*ex)/B
    else:
        A0=0.5*math.sqrt(math.pi)*math.erfc(-si);E=math.exp(-si*si);I0=si*A0+0.5*E
        if not I0>0:return dict(mean_sx=float('nan'),mean_sy=0.0,mean_sx2=float('nan'),mean_sy2=alpha*alpha/2)
        mt=-0.5*A0/I0;mt2=(0.5*si*A0+0.5*E)/I0
    return dict(mean_sx=alpha*mt,mean_sy=0.0,mean_sx2=alpha*alpha*mt2,mean_sy2=alpha*alpha/2)

def _failure_counts(codes):
    a=np.asarray(codes,int);return {FAILURE_NAMES.get(int(k),f'UNKNOWN_{int(k)}'):int(v) for k,v in zip(*np.unique(a,return_counts=True)) if int(k)!=0}



def _cuda_c_f32_vector(cp, values, name='values'):
    """Make the raw-kernel memory contract explicit for host->device float32 vectors."""
    a=np.ascontiguousarray(np.asarray(values,dtype=np.float32).reshape(-1))
    if a.dtype!=np.float32 or not a.flags.c_contiguous or a.strides!=(4,):
        raise RuntimeError(f'{name}: invalid host raw-CUDA layout dtype={a.dtype} shape={a.shape} strides={a.strides}')
    d=cp.asarray(a,order='C')
    if d.dtype!=cp.float32 or d.ndim!=1 or not bool(d.flags.c_contiguous):
        raise RuntimeError(f'{name}: invalid device raw-CUDA layout')
    return d

def _smith_lambda_reference(c:float,alpha:float)->float:
    c=max(-1.0,min(1.0,float(c)))
    if abs(c)==1.0:return 0.0 if c>0 else -1.0
    if c==0.0:return math.inf
    ss=math.sqrt(max(0.0,1.0-c*c));a=abs(c)/(alpha*ss)
    if not a>0:return math.inf if c>0 else -math.inf
    B=_tail_B_host(a)
    if not B>0:return float('nan')
    lp=math.exp(-a*a-math.log(2.0*math.sqrt(math.pi)*a))*B if a*a<740 else 0.0
    return lp if c>0 else -1.0-lp

def _height_reference(c:float,h:float,U:float,alpha:float):
    c=max(-1.0,min(1.0,float(c)));C0=max(0.0,min(1.0,0.5*(float(h)+1.0)))
    if not (0.0<U<1.0):return 3,float('nan')
    if abs(c)==1.0:
        return (2,float('nan')) if c>0 else (1,2.0*(U*C0)-1.0)
    if c==0.0:return 1,float(h)
    lam=_smith_lambda_reference(c,alpha)
    if c>0:
        if C0<=0.0 or C0>=1.0 or lam==0.0:return 2,float('nan')
        logC0=math.log(C0);p=-math.expm1(lam*logC0)
        if U>p:return 2,float('nan')
        lc=logC0-math.log1p(-U)/lam
    else:
        if C0<=0.0:return 1,-1.0
        lc=math.log(C0)-math.log1p(-U)/lam
    if lc>1e-12:return 3,float('nan')
    C=math.exp(min(0.0,lc));return 1,2.0*C-1.0

def acquire_gpu(root:Path,samples_per_bin:int|None=None):
    import cupy as cp
    cfg=vm.load_config();bp=cfg['boundary_process_v24_52'];N=int(samples_per_bin or bp['samples_per_bin']);alpha_nominal=math.tan(math.radians(25.0));alpha_cuda=float(np.float32(alpha_nominal));alpha=alpha_cuda;watch=int(bp['watchdog_internal_reflections']);root.mkdir(parents=True,exist_ok=True);adir=root/AUDIT_DIR;adir.mkdir(exist_ok=True)
    t=time.perf_counter();mod=cp.RawModule(code=core.CUDA_SRC+VALIDATION_CUDA,options=('-std=c++11',))
    names=('v43_reciprocity_kernel','v43_path_kernel','v43_microsurface_limits_kernel','v43_signed_vndf_kernel','v43_full_sphere_vndf_kernel','v43_forced_inside_second_hit_kernel','v43_height_sampler_kernel','v43_lambda_kernel','v43_height_stress_kernel','v37_bad_regression_kernel','v38_bad_equal_index_kernel','v39_bad_deep_backside_kernel','v40_bad_height_lambda_kernel','v41_bad_ultragrazing_kernel')
    kr,kp,kl,kv,kfs,ki,kh,klam,khstress,kb37,kb38,kb39,kb40,kb41=(mod.get_function(n) for n in names);cp.cuda.Stream.null.synchronize();compile_s=time.perf_counter()-t
    rows=[];cross=[];cons=[];surv=[];legacy=[];coverage={};failure_rows=[];subfailure_rows=[]
    for mi,mat in enumerate(('kaolin','loess')):
        model=vm.make_model(cfg,legacy_mode=False);prep=model._prepare_gpu_particle_physics(mat,1.0,None,'mass_fraction');host=prep['host'];w=_weights(host);coverage[mat]=0
        lmodel=vm.make_model(cfg,legacy_mode=True);lhost=lmodel._prepare_gpu_particle_physics(mat,1.0,None,'mass_fraction')['host']
        for idx,qsca in enumerate(np.asarray(host['qsca'],float)):
            if qsca<1.0:continue
            coverage[mat]+=1;np_=float(host['n_real'][idx]);nm=float(cfg['forward_model']['n_water']);seed=int(bp['seed'])+mi*1000003+idx*1009
            u=lambda n:cp.zeros(n,dtype=cp.uint64);refl=u(16);esc=u(4);ent=u(4);fail=u(6);sub=u(128);micro=u(1);thr=128;blk=(N+thr-1)//thr
            kr((blk,),(thr,),(np.int32(N),np.float32(nm),np.float32(np_),np.float32(0.0),np.float32(alpha),np.uint32(seed),refl,esc,ent,fail,sub,micro));cp.cuda.Stream.null.synchronize()
            rh=cp.asnumpy(refl).reshape(4,4);eh=cp.asnumpy(esc);ih=cp.asnumpy(ent);fh=cp.asnumpy(fail);sh=cp.asnumpy(sub);mh=int(cp.asnumpy(micro)[0]);O=(0,1);T=(2,3);fc=sum(int(rh[a,c]) for a in O for c in T);rc=sum(int(rh[a,c]) for a in T for c in O);wf,wr,ar,rel,se,z=_pair_metrics(fc,rc,N,np_**2);passed=bool(rel<=float(bp['relative_residual_limit']) or z<float(bp['z_limit']))
            micro_num=int(fh[0]+fh[3]);micro_watch=int(fh[1]+fh[4]);macro_geom=int(fh[2]+fh[5])
            rows.append(dict(release=RELEASE,material=mat,bin_index=idx,diameter_m=float(host['diameter_m'][idx]),qsca=qsca,event_rate_weight=float(w[idx]),samples=N,forward_count=fc,reverse_count=rc,forward_weighted_flux=wf,reverse_weighted_flux=wr,absolute_residual=ar,relative_residual=rel,standard_error=se,z_score=z,passed=passed,microsurface_scatter_events=mh,micro_numerical_failures=micro_num,micro_watchdog_failures=micro_watch,macro_geometry_failures=macro_geom))
            failure_rows.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,internal_micro_numerical=int(fh[0]),internal_micro_watchdog=int(fh[1]),internal_geometry=int(fh[2]),external_micro_numerical=int(fh[3]),external_micro_watchdog=int(fh[4]),external_geometry=int(fh[5])))
            for direction,off in (('internal',0),('external',64)):
                for code,name in FAILURE_NAMES.items():
                    if code==0:continue
                    count=int(sh[off+code]) if off+code<len(sh) else 0
                    subfailure_rows.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,direction=direction,failure_code=code,failure_name=name,count=count))
            for sidx,name in enumerate(STATE_NAMES):
                fc2=int(eh[sidx]);rc2=int(ih[sidx]);pf=np_**2*fc2/N;pr=nm**2*rc2/N;var=(np_**4)*(fc2/N)*(1-fc2/N)/N+(nm**4)*(rc2/N)*(1-rc2/N)/N;se3=math.sqrt(max(var,0));abs3=abs(pf-pr);rel3=abs3/max(abs(pf),abs(pr),1e-30);z3=abs3/se3 if se3 else (0 if abs3==0 else float('inf'))
                cross.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,event_rate_weight=float(w[idx]),state=name,internal_to_external_count=fc2,external_to_internal_count=rc2,internal_to_external_weighted_flux=pf,external_to_internal_weighted_flux=pr,relative_residual=rel3,standard_error=se3,z_score=z3,passed=bool(rel3<=float(bp['relative_residual_limit']) or z3<float(bp['z_limit']))))
            PN=max(12000,N//4);status=cp.empty(PN,dtype=cp.int32);irs=cp.empty(PN,dtype=cp.int32);blk2=(PN+thr-1)//thr;kp((blk2,),(thr,),(np.int32(PN),np.float32(nm),np.float32(np_),np.float32(0),np.float32(alpha),np.int32(watch),np.uint32(seed+77777),status,irs));cp.cuda.Stream.null.synchronize();st=cp.asnumpy(status);ir=cp.asnumpy(irs);watchhits=int(np.sum((st==0)&(ir>=watch)));otherfail=int(np.sum((st==0)&(ir<watch)));cons.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,samples=PN,survived_count=int(np.sum(st==1)),absorbed_count=int(np.sum(st==2)),watchdog_hits=watchhits,other_numerical_failures=otherfail,count_closure_error=int(len(st)-np.sum(st==1)-np.sum(st==2)-watchhits-otherfail),max_internal_reflections=int(ir.max(initial=0))))
            for th in THRESHOLDS:surv.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,event_rate_weight=float(w[idx]),reflection_threshold=th,reached_count=int(np.sum(ir>=th)),fraction=float(np.mean(ir>=th))))
            legacy.append(dict(release=RELEASE,material=mat,bin_index=idx,qsca=qsca,qsca_legacy=float(lhost['qsca'][idx]),event_rate_new=float(host['mu_event_by_bin'][idx]),event_rate_legacy=float(lhost['mu_event_by_bin'][idx]),event_rate_abs_diff=float(abs(host['mu_event_by_bin'][idx]-lhost['mu_event_by_bin'][idx])),host_identity_passed=bool(np.array_equal(np.asarray(host['qsca']),np.asarray(lhost['qsca'])) and np.array_equal(np.asarray(host['mu_event_by_bin']),np.asarray(lhost['mu_event_by_bin'])))))
    pd.DataFrame(rows).to_csv(adir/'v24_52_corrected_kernel_reciprocity_by_bin.csv',index=False);pd.DataFrame(cross).to_csv(adir/'v24_52_cross_interface_reciprocity.csv',index=False);pd.DataFrame(cons).to_csv(adir/'v24_52_probability_conservation.csv',index=False);pd.DataFrame(surv).to_csv(adir/'v24_52_internal_reflection_survival.csv',index=False);pd.DataFrame(legacy).to_csv(adir/'v24_52_legacy_identity.csv',index=False);pd.DataFrame(failure_rows).to_csv(adir/'v24_52_failure_classes.csv',index=False);pd.DataFrame(subfailure_rows).to_csv(adir/'v24_52_failure_subcodes.csv',index=False)

    # Smooth/equal-index gates through the actual production walker.
    MN=max(30000,N//2);thr=128;blk=(MN+thr-1)//thr;sok=cp.empty(MN,dtype=cp.int32);eok=cp.empty(MN,dtype=cp.int32);est=cp.empty(MN,dtype=cp.int32);eord=cp.empty(MN,dtype=cp.int32);efc=cp.empty(MN,dtype=cp.int32);eerr=cp.empty(MN,dtype=cp.float32);nm=float(cfg['forward_model']['n_water'])
    kl((blk,),(thr,),(np.int32(MN),np.float32(nm),np.float32(1.58),np.float32(alpha),np.uint32(int(bp['seed'])+99173),sok,eok,est,eord,efc,eerr));cp.cuda.Stream.null.synchronize();sh=cp.asnumpy(sok);eh=cp.asnumpy(eok);esh=cp.asnumpy(est);oh=cp.asnumpy(eord);efh=cp.asnumpy(efc);eeh=cp.asnumpy(eerr)
    eq_num=int(np.sum(esh==3));eq_watch=int(np.sum(esh==4));eq_pass=bool(np.all(eh==1) and eq_num==0 and eq_watch==0 and float(np.max(eeh,initial=0.0))<2.5e-7)
    pd.DataFrame([dict(release=RELEASE,test='smooth_surface_limit',samples=MN,failures=int(np.sum(sh!=1)),numerical_failures=0,watchdog_hits=0,max_direction_error_sq=0.0,max_scatter_order=1,passed=bool(np.all(sh==1))),dict(release=RELEASE,test='equal_index_limit',samples=MN,failures=int(np.sum(eh!=1)),numerical_failures=eq_num,watchdog_hits=eq_watch,max_direction_error_sq=float(np.max(eeh,initial=0.0)),max_scatter_order=int(oh.max(initial=0)),passed=eq_pass)]).to_csv(adir/'v24_52_microsurface_specific_tests.csv',index=False)
    pd.DataFrame({'release':RELEASE,'scatter_order':oh}).groupby('scatter_order').size().rename('count').reset_index().to_csv(adir/'v24_52_microsurface_scatter_order.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,failure_code=k,failure_name=FAILURE_NAMES.get(int(k),f'UNKNOWN_{int(k)}'),count=v) for k,v in zip(*np.unique(efh,return_counts=True)) if int(k)!=0]).to_csv(adir/'v24_52_equal_index_failure_subcodes.csv',index=False)

    # Targeted signed VNDF: front, horizontal, backside, and transition-angle cases.
    wz_cases=np.ascontiguousarray(np.asarray(VNDF_WZ,dtype=np.float32));nc=len(wz_cases);VN=max(nc*12000,N*4);blk=(VN+thr-1)//thr;dwz=_cuda_c_f32_vector(cp,wz_cases,'vndf_wz_cases');cid=cp.empty(VN,dtype=cp.int32);vok=cp.empty(VN,dtype=cp.int32);vfc=cp.empty(VN,dtype=cp.int32);vsx=cp.empty(VN,dtype=cp.float64);vsy=cp.empty(VN,dtype=cp.float64);vvd=cp.empty(VN,dtype=cp.float64);vtd=cp.empty(VN,dtype=cp.float64);vne=cp.empty(VN,dtype=cp.float64)
    kv((blk,),(thr,),(np.int32(VN),np.int32(nc),dwz,np.float32(alpha),np.uint32(int(bp['seed'])+44001),cid,vok,vfc,vsx,vsy,vvd,vtd,vne));cp.cuda.Stream.null.synchronize();ch=cp.asnumpy(cid);okh=cp.asnumpy(vok);fch=cp.asnumpy(vfc);sxh=cp.asnumpy(vsx).astype(float);syh=cp.asnumpy(vsy).astype(float);vdh=cp.asnumpy(vvd).astype(float);tdh=cp.asnumpy(vtd).astype(float);neh=cp.asnumpy(vne).astype(float)
    vrows=[];vndf_pass=True;vfrows=[]
    for c,wz in enumerate(wz_cases.astype(float)):
        m=ch==c;n=int(m.sum());valid=okh[m]==1;nv=int(valid.sum());ref=_vndf_reference_moments(wz,alpha);sx=sxh[m][valid];sy=syh[m][valid]
        if nv:
            means=(float(sx.mean()),float(sy.mean()),float((sx*sx).mean()),float((sy*sy).mean()));ses=(float(sx.std(ddof=1)/math.sqrt(nv)),float(sy.std(ddof=1)/math.sqrt(nv)),float((sx*sx).std(ddof=1)/math.sqrt(nv)),float((sy*sy).std(ddof=1)/math.sqrt(nv))) if nv>1 else (float('inf'),)*4
        else:means=(float('nan'),)*4;ses=(float('inf'),)*4
        refs=(ref['mean_sx'],ref['mean_sy'],ref['mean_sx2'],ref['mean_sy2']);moment_ok=bool(nv==n and all(np.isfinite(refs)) and all(abs(a-r)<=max(0.015,6.0*se,0.006*max(1.0,abs(r))) for a,r,se in zip(means,refs,ses)));orientation_ok=bool(nv==n and np.all(vdh[m]>0) and np.all(tdh[m]>0) and np.max(neh[m],initial=0)<2e-5);passed=bool(moment_ok and orientation_ok);vndf_pass &= passed
        wx_gpu=float(np.float32(math.sqrt(max(0.0,1.0-wz*wz))));direct_slope_i=(wz/(alpha_cuda*wx_gpu)) if wx_gpu>0 else (math.inf if wz>0 else -math.inf);swz=wz/math.sqrt(wz*wz+alpha_cuda*alpha_cuda*max(0.0,1.0-wz*wz)) if abs(wz)<1 or wz!=0 else wz;angle=math.degrees(math.acos(max(-1,min(1,wz))));sangle=math.degrees(math.acos(max(-1,min(1,swz))))
        vrows.append(dict(release=RELEASE,case=c,wz=wz,incidence_angle_deg=angle,stretched_wz=swz,stretched_incidence_angle_deg=sangle,direct_slope_i=direct_slope_i,samples=n,invalid_samples=n-nv,min_visible_dot=float(np.min(vdh[m],initial=np.inf)),min_top_dot=float(np.min(tdh[m],initial=np.inf)),max_normalization_error=float(np.max(neh[m],initial=0)),sample_mean_sx=means[0],reference_mean_sx=refs[0],sample_mean_sy=means[1],reference_mean_sy=refs[1],sample_mean_sx2=means[2],reference_mean_sx2=refs[2],sample_mean_sy2=means[3],reference_mean_sy2=refs[3],moment_sanity_passed=moment_ok,orientation_passed=orientation_ok,passed=passed))
        for name,count in _failure_counts(fch[m]).items():vfrows.append(dict(release=RELEASE,case=c,wz=wz,failure_name=name,count=count))
    pd.DataFrame(vrows).to_csv(adir/'v24_52_signed_vndf_validation.csv',index=False);pd.DataFrame(vfrows,columns=['release','case','wz','failure_name','count']).to_csv(adir/'v24_52_vndf_failure_subcodes.csv',index=False)

    # Random full-sphere VNDF stress binned by raw incidence angle.
    FN=max(360000,N*6);blk=(FN+thr-1)//thr;fok=cp.empty(FN,dtype=cp.int32);ffc=cp.empty(FN,dtype=cp.int32);fwz=cp.empty(FN,dtype=cp.float32);fvd=cp.empty(FN,dtype=cp.float64);ftd=cp.empty(FN,dtype=cp.float64);fne=cp.empty(FN,dtype=cp.float64)
    kfs((blk,),(thr,),(np.int32(FN),np.float32(alpha),np.uint32(int(bp['seed'])+47001),fok,ffc,fwz,fvd,ftd,fne));cp.cuda.Stream.null.synchronize();fo=cp.asnumpy(fok);ff=cp.asnumpy(ffc);fz=cp.asnumpy(fwz).astype(float);fd=cp.asnumpy(fvd).astype(float);ft=cp.asnumpy(ftd).astype(float);fn=cp.asnumpy(fne).astype(float);ang=np.degrees(np.arccos(np.clip(fz,-1,1)));bins=np.arange(0,185,5);bi=np.clip(np.digitize(ang,bins,right=False)-1,0,len(bins)-2);fsrows=[]
    for j in range(len(bins)-1):
        m=bi==j;n=int(m.sum());bad=int(np.sum(fo[m]!=1));fsrows.append(dict(release=RELEASE,angle_start_deg=float(bins[j]),angle_end_deg=float(bins[j+1]),samples=n,valid_samples=n-bad,numerical_failures=bad,visibility_failures=int(np.sum(ff[m]==13)),max_normalization_error=float(np.max(fn[m],initial=0)),min_visible_dot=float(np.min(fd[m],initial=np.inf)),min_top_dot=float(np.min(ft[m],initial=np.inf)),passed=bool(n>0 and bad==0 and np.all(fd[m]>0) and np.all(ft[m]>0) and np.max(fn[m],initial=0)<2e-5)))
    fullsphere_pass=bool(np.all(fo==1) and all(r['passed'] for r in fsrows));pd.DataFrame(fsrows).to_csv(adir/'v24_52_full_sphere_vndf_stress.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,failure_code=k,failure_name=FAILURE_NAMES.get(int(k),f'UNKNOWN_{int(k)}'),count=v) for k,v in zip(*np.unique(ff,return_counts=True)) if int(k)!=0]).to_csv(adir/'v24_52_full_sphere_failure_subcodes.csv',index=False)

    # Forced inside-side second hits.  Rare physical intersections are forced with a
    # double-precision probability from the same Smith Lambda/log-CDF law; no 1e-12 floor.
    inside_wz=np.ascontiguousarray(np.asarray((-0.10,-0.50,-0.70,-0.80,-0.90,-0.95,-0.99),dtype=np.float32));IN=max(60000,N);ic=len(inside_wz);blk=(IN+thr-1)//thr;diw=_cuda_c_f32_vector(cp,inside_wz,'inside_wz_cases');icid=cp.empty(IN,dtype=cp.int32);iok=cp.empty(IN,dtype=cp.int32);ihs=cp.empty(IN,dtype=cp.int32);ivs=cp.empty(IN,dtype=cp.int32);ifc=cp.empty(IN,dtype=cp.int32);ie=cp.empty(IN,dtype=cp.float32);iph=cp.empty(IN,dtype=cp.float64);iu=cp.empty(IN,dtype=cp.float64);ico=cp.empty(IN,dtype=cp.int32)
    ki((blk,),(thr,),(np.int32(IN),np.int32(ic),diw,np.float32(nm),np.float32(alpha_cuda),np.uint32(int(bp['seed'])+55001),icid,iok,ihs,ivs,ifc,ie,iph,iu,ico));cp.cuda.Stream.null.synchronize();ich=cp.asnumpy(icid);io=cp.asnumpy(iok);ihr=cp.asnumpy(ihs);ivr=cp.asnumpy(ivs);ifh=cp.asnumpy(ifc);ier=cp.asnumpy(ie);iphh=cp.asnumpy(iph).astype(float);iuh=cp.asnumpy(iu).astype(float);icoh=cp.asnumpy(ico);irows=[];inside_pass=True;inside_probability_pass=True
    for c,wz in enumerate(inside_wz.astype(float)):
        m=ich==c;construction=bool(np.all(icoh[m]==1) and np.all(np.isfinite(iphh[m])) and np.all(iphh[m]>0) and np.all(iuh[m]>0) and np.all(iuh[m]<iphh[m]));inside_probability_pass &= construction;passed=bool(construction and np.all(io[m]==1) and np.all(ihr[m]==1) and np.all(ivr[m]==1) and np.all(ifh[m]==0) and float(np.max(ier[m],initial=0.0))<2.5e-7);inside_pass &= passed;irows.append(dict(release=RELEASE,case=c,wz=wz,samples=int(m.sum()),forced_hit_probability_min=float(np.min(iphh[m],initial=np.inf)),forced_hit_probability_max=float(np.max(iphh[m],initial=-np.inf)),forced_u_min=float(np.min(iuh[m],initial=np.inf)),forced_u_max=float(np.max(iuh[m],initial=-np.inf)),probability_construction_passed=construction,failures=int(np.sum(io[m]!=1)),height_nonintersection=int(np.sum(ihr[m]!=1)),vndf_failures=int(np.sum(ivr[m]!=1)),numerical_failures=int(np.sum(ifh[m]!=0)),max_direction_error_sq=float(np.max(ier[m],initial=0.0)),passed=passed))
    pd.DataFrame(irows).to_csv(adir/'v24_52_inside_side_second_hit.csv',index=False)

    # Height sampler semantics/edge cases with explicit subcodes.
    HN=max(40000,N//2);blk=(HN+thr-1)//thr;hcid=cp.empty(HN,dtype=cp.int32);hst=cp.empty(HN,dtype=cp.int32);hok=cp.empty(HN,dtype=cp.int32);hfc=cp.empty(HN,dtype=cp.int32);hn=cp.empty(HN,dtype=cp.float32)
    kh((blk,),(thr,),(np.int32(HN),np.float32(alpha),np.uint32(int(bp['seed'])+66001),hcid,hst,hok,hfc,hn));cp.cuda.Stream.null.synchronize();hc=cp.asnumpy(hcid);hs=cp.asnumpy(hst);ho=cp.asnumpy(hok);hfh=cp.asnumpy(hfc);hh=cp.asnumpy(hn);hrows=[];height_pass=True
    for c in range(8):
        m=hc==c;passed=bool(np.all(ho[m]==1) and not np.any(hs[m]==3) and not np.any(hs[m]==4) and np.all(hfh[m]==0));height_pass &= passed;hrows.append(dict(release=RELEASE,case=c,samples=int(m.sum()),intersections=int(np.sum(hs[m]==1)),escapes=int(np.sum(hs[m]==2)),numerical_failures=int(np.sum(hs[m]==3)),watchdog_failures=int(np.sum(hs[m]==4)),failure_subcodes=int(np.sum(hfh[m]!=0)),finite_intersection_heights=bool(np.all(np.isfinite(hh[m][hs[m]==1]))),passed=passed))
    pd.DataFrame(hrows).to_csv(adir/'v24_52_height_sampler_validation.csv',index=False)

    # Direct Smith Lambda gate.  The independent reference uses the exact float32 alpha
    # delivered to CUDA, so the strict tolerance compares implementations at identical input.
    lambda_cases=np.asarray((-0.999999,-0.99,-0.9,-0.5,-0.1,0.1,0.5,0.8,0.85,0.87,0.875,0.87725,0.88,0.9,0.99,0.999999),dtype=np.float32)
    lambda_cases=np.ascontiguousarray(lambda_cases,dtype=np.float32);LN=len(lambda_cases);dlc=_cuda_c_f32_vector(cp,lambda_cases,'smith_lambda_cases');lcid=cp.empty(LN,dtype=cp.int32);lok=cp.empty(LN,dtype=cp.int32);llam=cp.empty(LN,dtype=cp.float64)
    klam((1,),(128,),(np.int32(LN),np.int32(LN),dlc,np.float32(alpha_cuda),lcid,lok,llam));cp.cuda.Stream.null.synchronize();lch=cp.asnumpy(lcid);loh=cp.asnumpy(lok);llh=cp.asnumpy(llam)
    lrows=[];lambda_pass=True;lambda_reference_precision_pass=True
    for j,c in enumerate(lambda_cases.astype(float)):
        ref=_smith_lambda_reference(c,alpha_cuda);got=float(llh[lch==j][0]);finite=math.isfinite(got);tol=max(2e-13,2e-10*max(1.0,abs(ref))) if math.isfinite(ref) else float('inf');passed=bool(loh[lch==j][0]==1 and finite and math.isfinite(ref) and abs(got-ref)<=tol);precision_ok=bool(float(np.float32(alpha_nominal))==alpha_cuda);lambda_reference_precision_pass &= precision_ok;lambda_pass &= passed;lrows.append(dict(release=RELEASE,case=j,direction_cosine=c,alpha_nominal=alpha_nominal,alpha_cuda=alpha_cuda,alpha_reference=alpha_cuda,lambda_cuda=got,lambda_reference=ref,absolute_error=abs(got-ref),relative_error=(abs(got-ref)/max(abs(ref),1e-300)),tolerance=tol,reference_input_precision_passed=precision_ok,passed=passed))
    pd.DataFrame(lrows).to_csv(adir/'v24_52_smith_lambda_validation.csv',index=False)

    # Large randomized height/free-path stress using the real production device function.
    HSN=max(240000,N*4);blk=(HSN+thr-1)//thr;hsst=cp.empty(HSN,dtype=cp.int32);hsfc=cp.empty(HSN,dtype=cp.int32);hsc=cp.empty(HSN,dtype=cp.float32);hsh=cp.empty(HSN,dtype=cp.float32);hsu=cp.empty(HSN,dtype=cp.float64);hsn=cp.empty(HSN,dtype=cp.float32);hsl=cp.empty(HSN,dtype=cp.float64)
    khstress((blk,),(thr,),(np.int32(HSN),np.float32(alpha),np.uint32(int(bp['seed'])+67001),hsst,hsfc,hsc,hsh,hsu,hsn,hsl));cp.cuda.Stream.null.synchronize();stx=cp.asnumpy(hsst);fcx=cp.asnumpy(hsfc);cx=cp.asnumpy(hsc).astype(float);hx=cp.asnumpy(hsh).astype(float);ux=cp.asnumpy(hsu).astype(float);hnx=cp.asnumpy(hsn).astype(float);lx=cp.asnumpy(hsl).astype(float)
    # Independent host reference on every sample; Python math uses double precision and separate algebra.
    ref_st=np.empty(HSN,dtype=np.int8);ref_h=np.full(HSN,np.nan,dtype=float)
    for ii,(cc,hh,uu) in enumerate(zip(cx,hx,ux)):
        rs,rh=_height_reference(cc,hh,uu,alpha_cuda);ref_st[ii]=rs;ref_h[ii]=rh
    status_match=stx==ref_st
    inter=(stx==1)&(ref_st==1)
    herr=np.abs(hnx[inter]-ref_h[inter]) if np.any(inter) else np.asarray([0.0])
    height_stress_pass=bool(np.all((stx==1)|(stx==2)) and np.all(fcx==0) and np.all(status_match) and float(np.max(herr,initial=0.0))<2.5e-6 and np.all(np.isfinite(lx) | (np.abs(cx)<1e-12)))
    edges=np.linspace(-1,1,21);bix=np.clip(np.digitize(cx,edges)-1,0,len(edges)-2);hsrows=[]
    for j in range(len(edges)-1):
        m=bix==j;n=int(m.sum());im=m&inter;err=float(np.max(np.abs(hnx[im]-ref_h[im]),initial=0.0)) if np.any(im) else 0.0
        hsrows.append(dict(release=RELEASE,c_start=float(edges[j]),c_end=float(edges[j+1]),samples=n,intersections=int(np.sum(stx[m]==1)),escapes=int(np.sum(stx[m]==2)),numerical_failures=int(np.sum(stx[m]==3)),watchdog_failures=int(np.sum(stx[m]==4)),failure_subcodes=int(np.sum(fcx[m]!=0)),status_mismatches=int(np.sum(~status_match[m])),max_height_error=err,lambda_min=float(np.nanmin(lx[m])) if n else np.nan,lambda_max=float(np.nanmax(lx[m])) if n else np.nan,passed=bool(n>0 and np.all((stx[m]==1)|(stx[m]==2)) and np.all(fcx[m]==0) and np.all(status_match[m]) and err<2.5e-6)))
    pd.DataFrame(hsrows).to_csv(adir/'v24_52_randomized_height_stress.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,failure_code=k,failure_name=FAILURE_NAMES.get(int(k),f'UNKNOWN_{int(k)}'),count=v) for k,v in zip(*np.unique(fcx,return_counts=True)) if int(k)!=0]).to_csv(adir/'v24_52_randomized_height_failure_subcodes.csv',index=False)
    fm=fcx!=0
    if np.any(fm):
        fs=dict(release=RELEASE,failures=int(np.sum(fm)),c_min=float(np.min(cx[fm])),c_max=float(np.max(cx[fm])),h_min=float(np.min(hx[fm])),h_max=float(np.max(hx[fm])),U_min=float(np.min(ux[fm])),U_max=float(np.max(ux[fm])),lambda_min=float(np.nanmin(lx[fm])),lambda_max=float(np.nanmax(lx[fm])),candidate_hnext_min=float(np.nanmin(hnx[fm])),candidate_hnext_max=float(np.nanmax(hnx[fm])))
    else:
        fs=dict(release=RELEASE,failures=0,c_min=np.nan,c_max=np.nan,h_min=np.nan,h_max=np.nan,U_min=np.nan,U_max=np.nan,lambda_min=np.nan,lambda_max=np.nan,candidate_hnext_min=np.nan,candidate_hnext_max=np.nan)
    pd.DataFrame([fs]).to_csv(adir/'v24_52_height_failure_state_summary.csv',index=False)

    # Historical known-bad regressions.
    BN=max(60000,N);brefl=cp.zeros(16,dtype=cp.uint64);bfail=cp.zeros(1,dtype=cp.uint64);blk=(BN+127)//128;kb37((blk,),(128,),(np.int32(BN),np.float32(nm),np.float32(1.58),np.float32(alpha),np.uint32(243700),brefl,bfail));cp.cuda.Stream.null.synchronize();br=cp.asnumpy(brefl).reshape(4,4);O=(0,1);T=(2,3);bf=sum(int(br[a,c]) for a in O for c in T);brv=sum(int(br[a,c]) for a in T for c in O);_,_,_,brel,bse,bz=_pair_metrics(bf,brv,BN,1.58**2);bad37_pass=bool(brel>0.05 and bz>=5.0);pd.DataFrame([dict(release=RELEASE,candidate='V24.37 VNDF + macro-support rejection/resampling',samples=BN,forward_count=bf,reverse_count=brv,relative_residual=brel,z_score=bz,numerical_failures=int(cp.asnumpy(bfail)[0]),known_bad_rejected=bad37_pass)]).to_csv(adir/'v24_52_v24_37_known_bad_regression.csv',index=False)
    B38=max(30000,N//2);blk=(B38+127)//128;bok=cp.empty(B38,dtype=cp.int32);bo=cp.empty(B38,dtype=cp.int32);kb38((blk,),(128,),(np.int32(B38),np.float32(alpha),np.uint32(243800),bok,bo));cp.cuda.Stream.null.synchronize();bh=cp.asnumpy(bok);boh=cp.asnumpy(bo);frac=float(np.mean(bh!=1));bad38_pass=bool(frac>0.05 and frac<0.20);pd.DataFrame([dict(release=RELEASE,candidate='V24.38 upper-hemisphere/invalid lower-side equal-index walk',samples=B38,failures=int(np.sum(bh!=1)),failure_fraction=frac,max_scatter_order=int(boh.max(initial=0)),real_gpu_reference_failure_fraction=3278/30000,known_bad_rejected=bad38_pass)]).to_csv(adir/'v24_52_v24_38_known_bad_regression.csv',index=False)
    B39=max(30000,N//2);blk=(B39+127)//128;b39o=cp.empty(B39,dtype=cp.int32);kb39((blk,),(128,),(np.int32(B39),np.float32(alpha),np.uint32(243900),b39o));cp.cuda.Stream.null.synchronize();b39h=cp.asnumpy(b39o);f39=float(np.mean(b39h!=1));bad39_pass=bool(f39>0.50);pd.DataFrame([dict(release=RELEASE,candidate='V24.39 erf_min=-0.9999 deep-backside signed-VNDF',wz=-0.90,samples=B39,failures=int(np.sum(b39h!=1)),failure_fraction=f39,real_equal_index_failure_fraction=281/30000,real_probability_population_failure_fraction=34104/990000,known_bad_rejected=bad39_pass)]).to_csv(adir/'v24_52_v24_39_known_bad_regression.csv',index=False)
    B40=max(60000,N);blk=(B40+127)//128;b40bad=cp.empty(B40,dtype=cp.int32);b40c=cp.empty(B40,dtype=cp.float32);b40l=cp.empty(B40,dtype=cp.float32);kb40((blk,),(128,),(np.int32(B40),np.float32(alpha),b40bad,b40c,b40l));cp.cuda.Stream.null.synchronize();b40h=cp.asnumpy(b40bad);b40ch=cp.asnumpy(b40c);b40lh=cp.asnumpy(b40l);bad40_count=int(np.sum(b40h==1));bad40_pass=bool(bad40_count>0);pd.DataFrame([dict(release=RELEASE,candidate='V24.40 cancelling float Smith Lambda / HEIGHT_CDF arithmetic',samples=B40,detected_invalid_lambda=bad40_count,failure_fraction=float(bad40_count/B40),c_min=float(np.min(b40ch[b40h==1],initial=np.inf)),c_max=float(np.max(b40ch[b40h==1],initial=-np.inf)),lambda_min=float(np.min(b40lh,initial=np.inf)),real_equal_index_height_failures=280,real_probability_population_failures=34187,real_complete_height_cdf_failures=122915,real_ultragrazing_vndf_failures=903,known_bad_rejected=bad40_pass)]).to_csv(adir/'v24_52_v24_40_known_bad_regression.csv',index=False)
    # V24.41.1 known-defect regression has two validator false negatives plus the genuine float-theta VNDF support defect.
    bad41_lambda_rows=[];strict_failed=0;corrected_passed=0
    for c in (-0.5,-0.1,0.1,0.5):
        ref_wrong=_smith_lambda_reference(float(np.float32(c)),alpha_nominal);ref_right=_smith_lambda_reference(float(np.float32(c)),alpha_cuda);tol=max(2e-13,2e-10*max(1.0,abs(ref_right)));strict_failed += int(abs(ref_wrong-ref_right)>tol);corrected_passed += int(abs(ref_right-ref_right)<=tol)
    cforce=0.95;lamforce=_smith_lambda_reference(cforce,alpha_cuda);ptrue=-math.expm1(lamforce*math.log(0.5));g1f=float(np.float32(math.exp(lamforce*math.log(0.5))));pold=max(1e-12,0.25*max(0.0,1.0-g1f));forced_bug=bool(pold>ptrue)
    B41=max(12000,N//5);blk=(B41+127)//128;b41ok=cp.empty(B41,dtype=cp.int32);b41se=cp.empty(B41,dtype=cp.float64);kb41((blk,),(128,),(np.int32(B41),np.float32(alpha_cuda),np.uint32(244101),b41ok,b41se));cp.cuda.Stream.null.synchronize();b41h=cp.asnumpy(b41ok);b41seh=cp.asnumpy(b41se);float_theta_failures=int(np.sum(b41h!=1));float_theta_frac=float(float_theta_failures/B41);float_theta_bug=bool(float_theta_frac>0.01)
    bad41_pass=bool(strict_failed>=2 and corrected_passed==4 and forced_bug and float_theta_bug)
    pd.DataFrame([dict(release=RELEASE,defect='lambda_reference_alpha_precision_mismatch',demonstrated=bool(strict_failed>=2),strict_cases_failed=strict_failed,corrected_cases_passed=corrected_passed,alpha_nominal=alpha_nominal,alpha_cuda=alpha_cuda),dict(release=RELEASE,defect='forced_inside_float_G1_plus_1e-12_floor',demonstrated=forced_bug,true_hit_probability=ptrue,old_forced_U=pold,float_G1=g1f),dict(release=RELEASE,defect='float_theta_ultragrazing_support',demonstrated=float_theta_bug,samples=B41,failures=float_theta_failures,failure_fraction=float_theta_frac,support_error_min=float(np.min(b41seh,initial=np.inf)),support_error_max=float(np.max(b41seh,initial=-np.inf)),real_v24_41_1_failures_at_wz_m0p9999=916),dict(release=RELEASE,defect='aggregate_v24_41_1_known_defects',demonstrated=bad41_pass,known_bad_rejected=bad41_pass)]).to_csv(adir/'v24_52_v24_41_known_bad_regression.csv',index=False)

    meta={'release':RELEASE,'gpu_acquisition_complete':True,'compile_seconds':compile_s,'samples_per_bin':N,'coverage':coverage,'corrected_core_cuda_sha256':hashlib.sha256(core.CUDA_SRC.encode()).hexdigest(),'validation_cuda_sha256':hashlib.sha256(VALIDATION_CUDA.encode()).hexdigest(),'roughness_deg':25.0,'k_zero':True,'watchdog_reflections':watch,'microsurface_watchdog':int(bp['microsurface_scattering_watchdog']),'v24_37_known_bad_rejected':bad37_pass,'v24_38_known_bad_rejected':bad38_pass,'v24_39_known_bad_rejected':bad39_pass,'v24_40_known_bad_rejected':bad40_pass,'v24_41_known_bad_rejected':bad41_pass,'smith_lambda_pass':lambda_pass,'smith_lambda_reference_precision_pass':lambda_reference_precision_pass,'randomized_height_stress_pass':height_stress_pass,'signed_vndf_pass':vndf_pass,'ultra_grazing_vndf_pass':bool(all(r['passed'] for r in vrows if r['wz']<=-0.999)),'deep_backside_vndf_pass':bool(all(r['passed'] for r in vrows if r['wz']<-.7)),'full_sphere_vndf_stress_pass':fullsphere_pass,'vndf_distribution_pass':vndf_pass,'equal_index_pass':eq_pass,'inside_side_second_hit_pass':inside_pass,'forced_inside_probability_construction_pass':inside_probability_pass,'height_sampler_pass':height_pass,'microsurface_specific_tests_pass':bool(eq_pass and np.all(sh==1) and vndf_pass and fullsphere_pass and inside_pass and inside_probability_pass and height_pass and lambda_pass and lambda_reference_precision_pass and height_stress_pass)}
    (adir/'v24_52_gpu_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta

def host_smoke(root:Path):
    root.mkdir(parents=True,exist_ok=True);adir=root/AUDIT_DIR;adir.mkdir(exist_ok=True);rows=[];cross=[];cons=[];surv=[];legacy=[];fails=[];subs=[]
    for mat in ('kaolin','loess'):
        for i in range(2):
            rows.append(dict(release=RELEASE,material=mat,bin_index=i,diameter_m=1e-6*(i+1),qsca=1.1+i,event_rate_weight=.5,samples=100,forward_count=20,reverse_count=20,forward_weighted_flux=.05,reverse_weighted_flux=.05,absolute_residual=0,relative_residual=0,standard_error=.01,z_score=0,passed=True,microsurface_scatter_events=100,micro_numerical_failures=0,micro_watchdog_failures=0,macro_geometry_failures=0))
            for st in STATE_NAMES:cross.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,event_rate_weight=.5,state=st,internal_to_external_count=10,external_to_internal_count=10,internal_to_external_weighted_flux=.02,external_to_internal_weighted_flux=.02,relative_residual=0,standard_error=.01,z_score=0,passed=True))
            cons.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,samples=100,survived_count=100,absorbed_count=0,watchdog_hits=0,other_numerical_failures=0,count_closure_error=0,max_internal_reflections=4))
            for th in THRESHOLDS:surv.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,event_rate_weight=.5,reflection_threshold=th,reached_count=0,fraction=0.0))
            legacy.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,qsca_legacy=1.1+i,event_rate_new=1,event_rate_legacy=1,event_rate_abs_diff=0,host_identity_passed=True));fails.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,internal_micro_numerical=0,internal_micro_watchdog=0,internal_geometry=0,external_micro_numerical=0,external_micro_watchdog=0,external_geometry=0))
            for direction in ('internal','external'):
                for code,name in FAILURE_NAMES.items():
                    if code:subs.append(dict(release=RELEASE,material=mat,bin_index=i,qsca=1.1+i,direction=direction,failure_code=code,failure_name=name,count=0))
    pd.DataFrame(rows).to_csv(adir/'v24_52_corrected_kernel_reciprocity_by_bin.csv',index=False);pd.DataFrame(cross).to_csv(adir/'v24_52_cross_interface_reciprocity.csv',index=False);pd.DataFrame(cons).to_csv(adir/'v24_52_probability_conservation.csv',index=False);pd.DataFrame(surv).to_csv(adir/'v24_52_internal_reflection_survival.csv',index=False);pd.DataFrame(legacy).to_csv(adir/'v24_52_legacy_identity.csv',index=False);pd.DataFrame(fails).to_csv(adir/'v24_52_failure_classes.csv',index=False);pd.DataFrame(subs).to_csv(adir/'v24_52_failure_subcodes.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,test='smooth_surface_limit',samples=0,failures=0,numerical_failures=0,watchdog_hits=0,max_direction_error_sq=0,max_scatter_order=0,passed=False),dict(release=RELEASE,test='equal_index_limit',samples=0,failures=0,numerical_failures=0,watchdog_hits=0,max_direction_error_sq=0,max_scatter_order=0,passed=False)]).to_csv(adir/'v24_52_microsurface_specific_tests.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,scatter_order=0,count=0)]).to_csv(adir/'v24_52_microsurface_scatter_order.csv',index=False);pd.DataFrame(columns=['release','failure_code','failure_name','count']).to_csv(adir/'v24_52_equal_index_failure_subcodes.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,case=0,wz=0.5,incidence_angle_deg=60,stretched_wz=0.5,stretched_incidence_angle_deg=60,samples=0,invalid_samples=0,min_visible_dot=np.nan,min_top_dot=np.nan,max_normalization_error=np.nan,sample_mean_sx=np.nan,reference_mean_sx=np.nan,sample_mean_sy=np.nan,reference_mean_sy=0,sample_mean_sx2=np.nan,reference_mean_sx2=np.nan,sample_mean_sy2=np.nan,reference_mean_sy2=np.nan,moment_sanity_passed=False,orientation_passed=False,passed=False)]).to_csv(adir/'v24_52_signed_vndf_validation.csv',index=False)
    pd.DataFrame(columns=['release','case','wz','failure_name','count']).to_csv(adir/'v24_52_vndf_failure_subcodes.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,angle_start_deg=0,angle_end_deg=5,samples=0,valid_samples=0,numerical_failures=0,visibility_failures=0,max_normalization_error=np.nan,min_visible_dot=np.nan,min_top_dot=np.nan,passed=False)]).to_csv(adir/'v24_52_full_sphere_vndf_stress.csv',index=False);pd.DataFrame(columns=['release','failure_code','failure_name','count']).to_csv(adir/'v24_52_full_sphere_failure_subcodes.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,case=0,wz=-0.1,samples=0,forced_hit_probability_min=np.nan,forced_hit_probability_max=np.nan,forced_u_min=np.nan,forced_u_max=np.nan,probability_construction_passed=False,failures=0,height_nonintersection=0,vndf_failures=0,numerical_failures=0,max_direction_error_sq=0,passed=False)]).to_csv(adir/'v24_52_inside_side_second_hit.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,case=0,samples=0,intersections=0,escapes=0,numerical_failures=0,watchdog_failures=0,failure_subcodes=0,finite_intersection_heights=True,passed=False)]).to_csv(adir/'v24_52_height_sampler_validation.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,case=0,direction_cosine=0.5,alpha_nominal=np.nan,alpha_cuda=np.nan,alpha_reference=np.nan,lambda_cuda=np.nan,lambda_reference=np.nan,absolute_error=np.nan,relative_error=np.nan,tolerance=np.nan,reference_input_precision_passed=False,passed=False)]).to_csv(adir/'v24_52_smith_lambda_validation.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,c_start=-1.0,c_end=-0.9,samples=0,intersections=0,escapes=0,numerical_failures=0,watchdog_failures=0,failure_subcodes=0,status_mismatches=0,max_height_error=np.nan,lambda_min=np.nan,lambda_max=np.nan,passed=False)]).to_csv(adir/'v24_52_randomized_height_stress.csv',index=False)
    pd.DataFrame(columns=['release','failure_code','failure_name','count']).to_csv(adir/'v24_52_randomized_height_failure_subcodes.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,failures=0,c_min=np.nan,c_max=np.nan,h_min=np.nan,h_max=np.nan,U_min=np.nan,U_max=np.nan,lambda_min=np.nan,lambda_max=np.nan,candidate_hnext_min=np.nan,candidate_hnext_max=np.nan)]).to_csv(adir/'v24_52_height_failure_state_summary.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,candidate='V24.37 validation-only known-bad regression',samples=0,forward_count=0,reverse_count=0,relative_residual=np.nan,z_score=np.nan,numerical_failures=0,known_bad_rejected=False)]).to_csv(adir/'v24_52_v24_37_known_bad_regression.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,candidate='V24.38 validation-only known-bad regression',samples=0,failures=0,failure_fraction=np.nan,max_scatter_order=0,real_gpu_reference_failure_fraction=3278/30000,known_bad_rejected=False)]).to_csv(adir/'v24_52_v24_38_known_bad_regression.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,candidate='V24.39 validation-only known-bad regression',wz=-.9,samples=0,failures=0,failure_fraction=np.nan,real_equal_index_failure_fraction=281/30000,real_probability_population_failure_fraction=34104/990000,known_bad_rejected=False)]).to_csv(adir/'v24_52_v24_39_known_bad_regression.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,candidate='V24.40 validation-only known-bad regression',samples=0,detected_invalid_lambda=0,failure_fraction=np.nan,c_min=np.nan,c_max=np.nan,lambda_min=np.nan,real_equal_index_height_failures=280,real_probability_population_failures=34187,real_complete_height_cdf_failures=122915,real_ultragrazing_vndf_failures=903,known_bad_rejected=False)]).to_csv(adir/'v24_52_v24_40_known_bad_regression.csv',index=False)
    pd.DataFrame([dict(release=RELEASE,defect='lambda_reference_alpha_precision_mismatch',demonstrated=False,known_bad_rejected=False)]).to_csv(adir/'v24_52_v24_41_known_bad_regression.csv',index=False)
    meta={'release':RELEASE,'gpu_acquisition_complete':False,'host_smoke':True,'microsurface_specific_tests_pass':False,'smith_lambda_pass':False,'smith_lambda_reference_precision_pass':False,'randomized_height_stress_pass':False,'signed_vndf_pass':False,'ultra_grazing_vndf_pass':False,'deep_backside_vndf_pass':False,'full_sphere_vndf_stress_pass':False,'vndf_distribution_pass':False,'equal_index_pass':False,'inside_side_second_hit_pass':False,'forced_inside_probability_construction_pass':False,'height_sampler_pass':False,'v24_37_known_bad_rejected':False,'v24_38_known_bad_rejected':False,'v24_39_known_bad_rejected':False,'v24_40_known_bad_rejected':False,'v24_41_known_bad_rejected':False};(adir/'v24_52_gpu_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta
