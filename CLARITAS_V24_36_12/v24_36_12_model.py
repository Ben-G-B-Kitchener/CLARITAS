from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from v24_36_12_release import HERE,CONFIG_NAME,RELEASE
import claritas_tardiis_core_v24_36 as core

def load_config(path: str|Path|None=None)->dict:
    p=HERE/CONFIG_NAME if path is None else Path(path)
    cfg=json.loads(p.read_text())
    if cfg.get('version')!=RELEASE:
        raise RuntimeError(f'config release mismatch: {cfg.get("version")} != {RELEASE}')
    b=cfg.get('reciprocity_fallback_audit_v24_36_12',{})
    if not b.get('diagnostic_release',False) or b.get('production_ray_campaign_enabled',True):
        raise RuntimeError(f'{RELEASE} must remain diagnostic-only')
    if cfg['forward_model'].get('particle_event_model')!='hybrid_mie_exact_rate':
        raise RuntimeError('audit must select the V24.29 exact-rate event model')
    if int(cfg['forward_model'].get('max_internal_bounces',-1))+1 != int(b['production_reflection_limit']):
        raise RuntimeError('production reflection-limit semantic mismatch')
    if int(b['diagnostic_reflection_limit']) < int(b['production_reflection_limit']):
        raise RuntimeError('diagnostic reflection limit cannot be smaller than production reflection limit')
    if float(cfg['forward_model'].get('particle_surface_rms_slope_deg',-1)) != 25.0:
        raise RuntimeError('V24.36.12 must audit, not alter, the inherited 25-degree particle roughness')
    return cfg

def particle_optics_override()->dict:
    out={}
    for mat in ('loess','kaolin'):
        n,k=core.get_material_optical_constants(mat,None,1.59)
        out[mat]={'n_real':np.asarray(n,float).tolist(),'k_imag':np.zeros_like(np.asarray(k,float)).tolist()}
    return out

def model_kwargs(cfg:dict, roughness_deg:float|None=None)->dict:
    d=dict(cfg['forward_model'])
    d.update(
        source_bore_surface_model='absorbing',detector_bore_surface_model='absorbing',ring_inner_face_surface_model='absorbing',
        photodiode_response_model='ideal_hard_aperture',sediment_transport_model='uniform',meridional_circulation_ratio=0.0,
        aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
        particle_event_model='hybrid_mie_exact_rate',
        particle_surface_rms_slope_deg=float(cfg['forward_model']['particle_surface_rms_slope_deg']) if roughness_deg is None else float(roughness_deg),
        wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.0,wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
        source_launch_radius_m=0.0655,alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d')
    d['particle_optical_constants_by_material']=particle_optics_override()
    return d

def event_and_csa_weights(host:dict)->tuple[np.ndarray,np.ndarray]:
    ev=np.asarray(host['mu_event_by_bin'],float);csa=np.asarray(host['mu_geom_by_bin'],float)
    ev=np.divide(ev,ev.sum(),out=np.zeros_like(ev),where=ev.sum()>0)
    csa=np.divide(csa,csa.sum(),out=np.zeros_like(csa),where=csa.sum()>0)
    return ev,csa

def qsca_in_audit(qsca:float,cfg:dict)->bool:
    return float(qsca)>=float(cfg['reciprocity_fallback_audit_v24_36_12']['audit_qsca_min'])
