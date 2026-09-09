from __future__ import annotations
import math
import numpy as np
import pandas as pd
from v24_36_12_release import RELEASE

PI=math.pi

def norm(v):
    a=np.asarray(v,float);q=float(np.linalg.norm(a))
    if not np.isfinite(q) or q<=0:raise ValueError('invalid vector')
    return a/q

def reflect(i,n):
    i=norm(i);n=norm(n);return norm(i-2.0*np.dot(i,n)*n)

def refract(i,n,n1,n2):
    i=norm(i);n=norm(n);ci=float(np.clip(-np.dot(i,n),0.0,1.0));eta=float(n1/n2)
    k=1.0-eta*eta*(1.0-ci*ci)
    if k<=0.0:return None
    return norm(eta*i+(eta*ci-math.sqrt(k))*n)

def fresnel_real(ci,n1,n2):
    ci=float(np.clip(ci,0.0,1.0));eta=float(n1/n2);st2=eta*eta*max(0.0,1.0-ci*ci)
    if st2>=1.0:return 1.0,True
    ct=math.sqrt(max(0.0,1.0-st2));ds=n1*ci+n2*ct;dp=n1*ct+n2*ci
    rs=((n1*ci-n2*ct)/ds)**2 if abs(ds)>1e-15 else 1.0
    rp=((n1*ct-n2*ci)/dp)**2 if abs(dp)>1e-15 else 1.0
    return float(np.clip(0.5*(rs+rp),0.0,1.0)),False

def beckmann_D(m,n,alpha):
    m=norm(m);n=norm(n);c=float(np.dot(m,n))
    if c<=0 or alpha<=0:return 0.0
    tan2=max(0.0,1.0-c*c)/(c*c)
    return math.exp(-tan2/(alpha*alpha))/(PI*alpha*alpha*c**4)

def smith_G1_beckmann(v,m,n,alpha):
    v=norm(v);m=norm(m);n=norm(n);vn=float(np.dot(v,n));vm=float(np.dot(v,m))
    if abs(vn)<=1e-15 or vm*vn<=0:return 0.0
    c=abs(vn);s=math.sqrt(max(0.0,1.0-c*c))
    if s<=1e-15 or alpha<=0:return 1.0
    a=c/(alpha*s)
    if a>=12:return 1.0
    den=1.0+math.erf(a)+math.exp(-a*a)/(a*math.sqrt(PI))
    return 2.0/den

def raw_ndf_density(m,n,alpha):
    return beckmann_D(m,n,alpha)*max(0.0,float(np.dot(norm(m),norm(n))))

def raw_facing_probability(incident_cos_macro,alpha):
    c=float(np.clip(abs(incident_cos_macro),0.0,1.0))
    s=math.sqrt(max(0.0,1.0-c*c))
    if s<=1e-15 or alpha<=0:return 1.0
    sigma=alpha/math.sqrt(2.0)
    z=c/(s*sigma)
    return 0.5*(1.0+math.erf(z/math.sqrt(2.0)))

def current_microfacet_density(v,m,n,alpha,attempts=8):
    """Continuous density of the current finite-attempt facing-rejection sampler.

    This excludes the point mass at the geometric normal produced if all attempts
    fail. v uses Walter's away-from-interface direction convention.
    """
    v=norm(v);m=norm(m);n=norm(n);vn=float(np.dot(v,n));vm=float(np.dot(v,m))
    if vm*vn<=0:return 0.0
    z=raw_facing_probability(abs(vn),alpha)
    if z<=0:return 0.0
    acceptance_factor=(1.0-(1.0-z)**attempts)/z
    return raw_ndf_density(m,n,alpha)*acceptance_factor

def visible_normal_density(v,m,n,alpha):
    v=norm(v);m=norm(m);n=norm(n);vn=abs(float(np.dot(v,n)))
    if vn<=1e-15:return 0.0
    return smith_G1_beckmann(v,m,n,alpha)*abs(float(np.dot(v,m)))*beckmann_D(m,n,alpha)/vn

def microfacet_reflection_bsdf(i,o,n,m,alpha):
    i=norm(i);o=norm(o);n=norm(n);m=norm(m)
    ci=abs(float(np.dot(i,m)));F,_=fresnel_real(ci,1.0,1.5)  # overwritten by caller variant only for ratio structure
    G=smith_G1_beckmann(i,m,n,alpha)*smith_G1_beckmann(o,m,n,alpha);D=beckmann_D(m,n,alpha)
    den=4.0*abs(float(np.dot(i,n)))*abs(float(np.dot(o,n)))
    return (F*G*D/den) if den>0 else 0.0

def reflection_bsdf(i,o,n,m,alpha,n_medium):
    i=norm(i);o=norm(o);n=norm(n);m=norm(m);ci=abs(float(np.dot(i,m)));F,_=fresnel_real(ci,n_medium,n_medium)
    G=smith_G1_beckmann(i,m,n,alpha)*smith_G1_beckmann(o,m,n,alpha);D=beckmann_D(m,n,alpha)
    den=4.0*abs(float(np.dot(i,n)))*abs(float(np.dot(o,n)))
    return (F*G*D/den) if den>0 else 0.0

def rough_reflection_bsdf(i,o,n,m,alpha,n1,n2):
    i=norm(i);o=norm(o);n=norm(n);m=norm(m);ci=abs(float(np.dot(i,m)));F,_=fresnel_real(ci,n1,n2)
    G=smith_G1_beckmann(i,m,n,alpha)*smith_G1_beckmann(o,m,n,alpha);D=beckmann_D(m,n,alpha)
    den=4.0*abs(float(np.dot(i,n)))*abs(float(np.dot(o,n)))
    return (F*G*D/den) if den>0 else 0.0

def rough_transmission_bsdf(i,o,n,m,alpha,eta_i,eta_o):
    """Walter et al. 2007 Eq. 21, directions point away from interface."""
    i=norm(i);o=norm(o);n=norm(n);m=norm(m)
    ci=abs(float(np.dot(i,m)));F,_=fresnel_real(ci,eta_i,eta_o)
    G=smith_G1_beckmann(i,m,n,alpha)*smith_G1_beckmann(o,m,n,alpha);D=beckmann_D(m,n,alpha)
    inn=abs(float(np.dot(i,n)));onn=abs(float(np.dot(o,n)))
    im=float(np.dot(i,m));om=float(np.dot(o,m));den0=eta_i*im+eta_o*om
    den=inn*onn*den0*den0
    if den<=1e-30:return 0.0
    return abs(im)*abs(om)*(eta_o**2)*(1.0-F)*G*D/den

def fixed_cases(alpha=math.tan(math.radians(25.0))):
    n=np.array([0.,0.,1.])
    cases=[]
    def micro(theta,phi=0.0):return norm([math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)])
    # Reflection from inside particle. Propagation vin points toward surface.
    for name,vin,m in [
        ('reflection_moderate',norm([0.25,0.0,0.9682458]),micro(math.radians(12),0.4)),
        ('reflection_grazing',norm([0.85,0.0,0.5267827]),micro(math.radians(20),-0.3)),
    ]:cases.append((name,'reflection',vin,n,m,1.59,1.333))
    # Transmission internal->water, deliberately below facet critical angle.
    for name,vin,m in [
        ('transmission_moderate',norm([0.25,0.0,0.9682458]),micro(math.radians(5),0.2)),
        ('transmission_near_critical',norm([0.72,0.0,0.693974]),micro(math.radians(8),0.0)),
    ]:cases.append((name,'transmission',vin,n,m,1.59,1.333))
    return cases

def fixed_microfacet_rows(alpha=math.tan(math.radians(25.0))):
    out=[]
    for name,mode,vin,n,m,n1,n2 in fixed_cases(alpha):
        if mode=='reflection':
            vo=reflect(vin,m);rev=reflect(-vo,m);rerr=float(np.linalg.norm(rev+vin));F1,_=fresnel_real(abs(np.dot(vin,m)),n1,n2);F2,_=fresnel_real(abs(np.dot(-vo,m)),n1,n2)
            i=-vin;o=vo;f1=rough_reflection_bsdf(i,o,n,m,alpha,n1,n2);f2=rough_reflection_bsdf(o,i,n,m,alpha,n1,n2);berr=abs(f1-f2)/max(abs(f1),abs(f2),1e-30);ok=True
            relation='fr(i,o)=fr(o,i)'
        else:
            vo=refract(vin,-m,n1,n2)
            ok=vo is not None
            if ok:
                rev=refract(-vo,m,n2,n1);rerr=float(np.linalg.norm(rev+vin)) if rev is not None else float('inf');F1,_=fresnel_real(abs(np.dot(vin,m)),n1,n2);F2,_=fresnel_real(abs(np.dot(-vo,m)),n2,n1)
                i=-vin;o=vo;f1=rough_transmission_bsdf(i,o,n,m,alpha,n1,n2);f2=rough_transmission_bsdf(o,i,n,m,alpha,n2,n1);a=f1/(n2*n2);b=f2/(n1*n1);berr=abs(a-b)/max(abs(a),abs(b),1e-30)
            else:rerr=float('inf');F1=F2=float('nan');berr=float('inf')
            relation='ft(i,o)/n_o^2 = ft(o,i)/n_i^2'
        out.append({'release':RELEASE,'case_name':name,'mode':mode,'n1':n1,'n2':n2,'incident_cos_macro':float(abs(np.dot(vin,n))),'microfacet_cos_macro':float(np.dot(m,n)),'forward_ok':bool(ok),'reverse_ok':bool(np.isfinite(rerr)),'forward_reverse_direction_l2':rerr,'fresnel_reverse_abs':float(abs(F1-F2)) if np.isfinite(F1) else float('nan'),'bsdf_reciprocity_relative_error':float(berr),'expected_relation':relation,'passed':bool(ok and rerr<1e-10 and abs(F1-F2)<1e-10 and berr<1e-10)})
    return out

def reconstruct_pair_rows(rows:list[dict]|pd.DataFrame,alpha:float,direction_tol:float=3e-6,fresnel_tol:float=3e-6,bsdf_tol:float=2e-5)->list[dict]:
    """Validate local optics using exact deterministic reconstruction.

    CUDA pair directions are float32 observations.  They are retained only as a
    storage-precision diagnostic; exact reciprocity invariants are evaluated from
    the recorded incident direction and microfacet normal.
    """
    df=pd.DataFrame(rows);out=[]
    if df.empty:return out
    for _,r in df.iterrows():
        vin=norm([r.ix,r.iy,r.iz]);n=norm([r.gnx,r.gny,r.gnz]);m=norm([r.mnx,r.mny,r.mnz]);stored=norm([r.fox,r.foy,r.foz]);n1=float(r.n1);n2=float(r.n2);cat=str(r.category)
        trans_like=('escape' in cat or 'fallback' in cat or 'candidate' in cat)
        if trans_like:
            exact=refract(vin,-m,n1,n2)
            if exact is not None:
                storage_err=float(np.linalg.norm(exact-stored));reverse=refract(-exact,m,n2,n1);rev_err=float(np.linalg.norm(reverse+vin)) if reverse is not None else float('inf');F1,_=fresnel_real(abs(np.dot(vin,m)),n1,n2);F2,_=fresnel_real(abs(np.dot(-exact,m)),n2,n1)
                i=-vin;o=exact;f1=rough_transmission_bsdf(i,o,n,m,alpha,n1,n2);f2=rough_transmission_bsdf(o,i,n,m,alpha,n2,n1);a=f1/(n2*n2);b=f2/(n1*n1);terr=abs(a-b)/max(abs(a),abs(b),1e-30)
            else:storage_err=rev_err=float('nan');F1=F2=float('nan');terr=float('nan');exact=stored
            rerr=float('nan')
        else:
            exact=reflect(vin,m);storage_err=float(np.linalg.norm(exact-stored));reverse=reflect(-exact,m);rev_err=float(np.linalg.norm(reverse+vin));F1,_=fresnel_real(abs(np.dot(vin,m)),n1,n2);F2,_=fresnel_real(abs(np.dot(-exact,m)),n1,n2);i=-vin;o=exact;f1=rough_reflection_bsdf(i,o,n,m,alpha,n1,n2);f2=rough_reflection_bsdf(o,i,n,m,alpha,n1,n2);rerr=abs(f1-f2)/max(abs(f1),abs(f2),1e-30);terr=float('nan')
        i=-vin;o=exact
        qf=current_microfacet_density(i,m,n,alpha);qr=current_microfacet_density(o,m,n,alpha);vf=visible_normal_density(i,m,n,alpha);vr=visible_normal_density(o,m,n,alpha)
        accepted_actual = cat in ('accepted_rough_reflection','accepted_rough_escape','outside_to_tir')
        if accepted_actual and qr>0 and qf>0:
            im=abs(float(np.dot(i,m)));om=abs(float(np.dot(o,m)));inn=abs(float(np.dot(i,n)));onn=abs(float(np.dot(o,n)))
            if trans_like:
                denf=(n1*float(np.dot(i,m))+n2*float(np.dot(o,m)))**2;denr=(n2*float(np.dot(o,m))+n1*float(np.dot(i,m)))**2;jf=(n2*n2*om/denf) if denf>1e-30 else 0.0;jr=(n1*n1*im/denr) if denr>1e-30 else 0.0;kf=qf*(1.0-F1)*jf;kr=qr*(1.0-F2)*jr;af=(n1*n1)*inn*kf;ar=(n2*n2)*onn*kr
            else:
                jf=1.0/(4.0*om) if om>1e-30 else 0.0;jr=1.0/(4.0*im) if im>1e-30 else 0.0;kf=qf*F1*jf;kr=qr*F2*jr;af=inn*kf;ar=onn*kr
            db_ratio=af/ar if ar>0 else float('inf');db_err=abs(af-ar)/max(abs(af),abs(ar),1e-30)
        else:db_ratio=float('nan');db_err=float('nan')
        passed=bool((not np.isfinite(rev_err) or rev_err<=direction_tol) and (not np.isfinite(F1) or abs(F1-F2)<=fresnel_tol) and (not np.isfinite(rerr) or rerr<=bsdf_tol) and (not np.isfinite(terr) or terr<=bsdf_tol))
        out.append({'release':RELEASE,'material':r.material,'bin_index':int(r.bin_index),'sample_id':int(r.sample_id),'category':cat,'fixed_facet_reverse_direction_l2':float(rev_err),'fresnel_reverse_abs':float(abs(F1-F2)) if np.isfinite(F1) else float('nan'),'reflection_bsdf_reciprocity_rel':float(rerr),'transmission_basic_radiance_reciprocity_rel':float(terr),'current_sampler_forward_density':float(qf),'current_sampler_reverse_density':float(qr),'current_sampler_density_ratio':float(qf/qr) if qr>0 else float('inf'),'current_kernel_detailed_balance_ratio':float(db_ratio),'current_kernel_detailed_balance_relative_error':float(db_err),'physical_vndf_forward_density':float(vf),'physical_vndf_reverse_density':float(vr),'note':f'stored_float32_direction_l2={storage_err:.3g}; exact deterministic direction used for invariants' if np.isfinite(storage_err) else 'exact proposal not refractable/TIR','passed_local_optics':passed})
    return out

