#!/usr/bin/env python3
"""Canonical complete-apparatus geometry for CLARITAS V24.52.

This module is deliberately host-only and independent of the CUDA transport.  It
provides the canonical geometry manifest, region/surface topology, and reference
classifiers used by V24.52 verification.  Production CUDA uses the same physical
parameters through the normal model configuration but does not call these helpers.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from v24_52_release import HERE, RELEASE

HARDWAREX_URL='https://eprints.whiterose.ac.uk/id/eprint/141481/1/1-s2.0-S2468067218300762-main.pdf'
HARDWAREX_CITATION='Kitchener et al., HardwareX 5 (2019) e00052, Section 5.1'

REGION_NAMES={0:'INTERFACE_OR_AMBIGUOUS',1:'WATER',2:'ACRYLIC_SIDEWALL',3:'ACRYLIC_BOTTOM',4:'HEADSPACE_AIR',5:'EXTERNAL_AIR'}
SURFACE_NAMES={
 10:'WATER_AIR_FREE_SURFACE',11:'INNER_ACRYLIC_CYLINDER',12:'BOTTOM_WATER_APERTURE',13:'OPEN_BORE_TOP',
 21:'ACRYLIC_INNER_CYLINDER',22:'ACRYLIC_OUTER_CYLINDER',23:'ACRYLIC_BOTTOM_EXTERNAL_FACE',24:'ACRYLIC_ANNULAR_TOP_FACE',25:'ACRYLIC_BOTTOM_WATER_APERTURE',
}

def load_cfg(path:Path|None=None):
    p=HERE/'claritas_v24_52_config.json' if path is None else Path(path)
    return json.loads(p.read_text())

def parameters(cfg=None, *, cuda_effective=False):
    cfg=load_cfg() if cfg is None else cfg
    f=cfg['forward_model']
    p=dict(
      Rin=float(f['tube_inner_radius_m']), Rout=float(f['tube_outer_radius_m']), tube_length=float(f['tube_length_m']),
      sensor_height=float(f['sensor_height_above_bottom_m']), water_height=float(f['water_height_m']),
      bottom_thickness=float(f['bottom_disc_thickness_m']),
      free_surface_model={'flat':0,'volume_preserving_paraboloid':1,'localized_scully_vortex':2}[f['free_surface_model']],
      delta_h=float(f['vortex_wall_center_height_difference_m']), core_r=float(f['vortex_core_radius_m']),
      n_water=float(f['n_water']),n_acrylic=float(f['n_acrylic']),n_air=float(f['n_air']),n_bottom=float(f.get('bottom_external_refractive_index',f['n_air'])),
    )
    p['Zmin']=-p['sensor_height'];p['Zlow']=p['Zmin']-p['bottom_thickness'];p['Zmean']=p['water_height']-p['sensor_height'];p['Zwall_top']=p['tube_length']-p['sensor_height']
    if cuda_effective:
        for k in ('Rin','Rout','Zmin','Zlow','Zmean','Zwall_top','delta_h','core_r','n_water','n_acrylic','n_air','n_bottom'):
            p[k]=float(np.float32(p[k]))
    return p

def support_tol(p):
    scale=max(abs(p['Rout']),abs(p['Zlow']),abs(p['Zwall_top']),abs(p['Rin']),abs(p['Zmin']),1e-3)
    return 8.0*np.finfo(np.float32).eps*scale

def free_surface_z(x,y,p):
    Z=p['Zmean'];R=p['Rin'];dh=p['delta_h'];a=p['core_r'];m=int(p['free_surface_model']);q=float(x)*float(x)+float(y)*float(y)
    if m==0 or dh<=0:return Z
    if m==1:return Z+dh*(q/(R*R)-0.5)
    a2=a*a;R2=R*R;H=dh*(a2+R2)/R2;hbar=H*(1.0-(a2/R2)*math.log1p(R2/a2));return Z-hbar+H*q/(a2+q)

def region_classify(x,y,z,p):
    rr=math.hypot(float(x),float(y));tol=support_tol(p)
    if z>p['Zlow']+tol and z<p['Zmin']-tol and rr<p['Rout']-tol:return 3
    if z>p['Zmin']+tol and z<p['Zwall_top']-tol and rr>p['Rin']+tol and rr<p['Rout']-tol:return 2
    if z>=p['Zmin']-tol and z<=p['Zwall_top']+tol and rr<p['Rin']-tol:
        zs=free_surface_z(x,y,p)
        if z<zs-tol:return 1
        if z>zs+tol:return 4
        return 0
    if z<p['Zlow']-tol or z>p['Zwall_top']+tol or rr>p['Rout']+tol:return 5
    return 0

def cylinder_roots(x,y,vx,vy,R):
    A=vx*vx+vy*vy;B=2.0*(x*vx+y*vy);C=x*x+y*y-R*R
    if abs(A)<1e-24:
        return []
    D=B*B-4*A*C
    if D<0:return []
    sd=math.sqrt(max(D,0.0));q=-0.5*(B+math.copysign(sd,B));
    if abs(q)>1e-300:t1=q/A;t2=C/q
    else:t1=(-B-sd)/(2*A);t2=(-B+sd)/(2*A)
    return sorted([t for t in (t1,t2) if t>0 and math.isfinite(t)])

def finite_cylinder_hit(x,y,z,vx,vy,vz,R,zlo,zhi,p):
    tol=support_tol(p);best=math.inf
    for t in cylinder_roots(x,y,vx,vy,R):
        zh=z+t*vz
        if zlo-tol<=zh<=zhi+tol:best=min(best,t)
    return best

def acrylic_next_surface(state,p):
    x,y,z,vx,vy,vz=map(float,state);tol=support_tol(p);cand=[]
    for t in cylinder_roots(x,y,vx,vy,p['Rin']):
        zh=z+t*vz
        if p['Zmin']-tol<=zh<=p['Zwall_top']+tol:cand.append((t,21))
    for t in cylinder_roots(x,y,vx,vy,p['Rout']):
        zh=z+t*vz
        if p['Zlow']-tol<=zh<=p['Zwall_top']+tol:cand.append((t,22))
    if vz<0:
        t=(p['Zlow']-z)/vz
        if t>0 and math.hypot(x+t*vx,y+t*vy)<=p['Rout']+tol:cand.append((t,23))
    if vz>0:
        t=(p['Zwall_top']-z)/vz
        rr=math.hypot(x+t*vx,y+t*vy)
        if t>0 and p['Rin']-tol<=rr<=p['Rout']+tol:cand.append((t,24))
        t=(p['Zmin']-z)/vz
        rr=math.hypot(x+t*vx,y+t*vy)
        if t>0 and rr<p['Rin']-tol:cand.append((t,25))
    return min(cand,key=lambda q:q[0]) if cand else (math.inf,0)

def canonical_manifest(cfg=None):
    cfg=load_cfg() if cfg is None else cfg
    p=parameters(cfg);wall_z=free_surface_z(p['Rin'],0,p);centre_z=free_surface_z(0,0,p)
    dims=[
      ('tube_inner_radius_m',p['Rin'],'m','physical acrylic tube inner radius','HardwareX: 93 mm internal diameter','authoritative apparatus literature'),
      ('tube_outer_radius_m',p['Rout'],'m','physical acrylic tube outer radius','HardwareX: 100 mm external diameter','authoritative apparatus literature'),
      ('tube_length_m',p['tube_length'],'m','physical acrylic tube length','HardwareX: 491 mm tube length','authoritative apparatus literature'),
      ('sensor_height_above_bottom_m',p['sensor_height'],'m','sensor plane height above internal bottom','CLARITAS experimental configuration','project configuration'),
      ('water_height_m',p['water_height'],'m','mean water depth above internal bottom','CLARITAS experiment configuration','project configuration'),
      ('bottom_disc_thickness_m',p['bottom_thickness'],'m','acrylic bottom disc thickness used by optical model','CLARITAS apparatus configuration','project assumption'),
      ('Z_cell_bottom_m',p['Zmin'],'m','internal water/acrylic bottom elevation relative to sensor plane','derived: -sensor height','derived'),
      ('Z_outer_bottom_m',p['Zlow'],'m','external bottom acrylic face','derived: Z_cell_bottom-bottom thickness','derived'),
      ('Z_water_mean_m',p['Zmean'],'m','mean free-surface elevation relative to sensor plane','derived: water height-sensor height','derived'),
      ('Z_wall_top_m',p['Zwall_top'],'m','physical acrylic tube top relative to sensor plane','derived: tube length-sensor height','derived'),
      ('free_surface_centre_m',centre_z,'m','localized vortex centre surface elevation','derived from configured Scully surface','derived'),
      ('free_surface_wall_m',wall_z,'m','localized vortex wall surface elevation','derived from configured Scully surface','derived'),
      ('sensor_ring_inner_radius_m',float(cfg['forward_model']['sensor_ring_inner_radius_m']),'m','inner radius of source/detector ring','CLARITAS apparatus machining configuration','project apparatus configuration'),
      ('sensor_ring_outer_radius_m',float(cfg['forward_model']['sensor_ring_outer_radius_m']),'m','outer radius of source/detector ring','CLARITAS apparatus machining configuration','project apparatus configuration'),
      ('source_launch_radius_m',float(cfg['forward_model']['source_launch_radius_m']),'m','source launch reference radius','CLARITAS source geometry configuration','project configuration / documented assumption'),
      ('through_bore_radius_m',0.5*float(cfg['forward_model']['through_bore_diameter_m']),'m','source/detector through-bore radius','CLARITAS apparatus machining configuration','project apparatus configuration'),
      ('counterbore_radius_m',0.5*float(cfg['forward_model']['counterbore_diameter_m']),'m','source/detector counterbore radius','CLARITAS apparatus machining configuration','project apparatus configuration'),
      ('counterbore_depth_m',float(cfg['forward_model']['counterbore_depth_m']),'m','source/detector counterbore depth','CLARITAS apparatus machining configuration','project apparatus configuration'),
    ]
    ring_in=float(cfg['forward_model']['sensor_ring_inner_radius_m']);ring_out=float(cfg['forward_model']['sensor_ring_outer_radius_m']);source_r=float(cfg['forward_model']['source_launch_radius_m'])
    return {
      'release':RELEASE,'coordinate_reference':'sensor/detector plane z=0; +z upward along tube',
      'authoritative_apparatus_reference':{'citation':HARDWAREX_CITATION,'url':HARDWAREX_URL,'supported_dimensions':'tube length 491 mm; ID 93 mm; OD 100 mm; acrylic disc closes one end'},
      'dimensions':[{'parameter':a,'value':b,'unit':c,'physical_meaning':d,'source_provenance':e,'status':f,'cuda_representation':'float32 where passed to transport','host_reference_representation':'same CUDA-effective scalar value, double-precision calculations'} for a,b,c,d,e,f in dims],
      'derived_relationships':{'wall_top_above_mean_surface_m':p['Zwall_top']-p['Zmean'],'wall_top_above_vortex_wall_surface_m':p['Zwall_top']-wall_z,'free_surface_contained_by_wall':bool(wall_z<p['Zwall_top']),'detector_ring_radial_clearance_from_acrylic_m':ring_in-p['Rout'],'source_launch_at_or_outside_ring_inner':bool(source_r>=ring_in),'source_launch_z_reference_within_wall_support':bool(p['Zmin']<0.0<p['Zwall_top'])},
      'source_detector_geometry':{'sensor_plane_z_m':0.0,'ring_inner_radius_m':ring_in,'ring_outer_radius_m':ring_out,'source_launch_radius_m':source_r,'through_bore_radius_m':0.5*float(cfg['forward_model']['through_bore_diameter_m']),'counterbore_radius_m':0.5*float(cfg['forward_model']['counterbore_diameter_m']),'counterbore_depth_m':float(cfg['forward_model']['counterbore_depth_m']),'detector_angles_deg':[10*i for i in range(18)],'consistency_assertions':['ring_inner_radius > acrylic_outer_radius','ring_outer_radius > ring_inner_radius','through_bore_radius < counterbore_radius','sensor/source z reference lies inside finite sidewall support']},
      'sediment_particle_geometry':{'particle_domain':'finite sphere centres proposed in water only','required_clearances':['full finite sphere inside Rin','sphere above Zmin bottom','sphere below spatially varying free surface over finite footprint'],'free_surface_criterion':'lowest free-surface point under the finite spherical footprint'},
      'assumptions':['tube is open at its upper end unless a future apparatus-specific closure is explicitly configured','external medium is air for sidewall/top; configured bottom external refractive index applies to outer bottom face'],
    }

def topology_manifest(cfg=None):
    p=parameters(cfg)
    regions=[
      dict(region_id=1,name='WATER',radial_support='r<Rin',axial_support='Zmin<z<local free surface',material='water',neighbors=['headspace','acrylic sidewall','acrylic bottom']),
      dict(region_id=4,name='HEADSPACE_AIR',radial_support='r<Rin',axial_support='local free surface<z<Zwall_top',material='air',neighbors=['water','acrylic sidewall','external air through open top']),
      dict(region_id=2,name='ACRYLIC_SIDEWALL',radial_support='Rin<r<Rout',axial_support='Zmin<z<Zwall_top',material='acrylic',neighbors=['water or headspace at inner wall','external air at outer wall','acrylic bottom at seam','external air at annular top']),
      dict(region_id=3,name='ACRYLIC_BOTTOM',radial_support='r<Rout',axial_support='Zlow<z<Zmin',material='acrylic',neighbors=['water over r<Rin at top','acrylic sidewall over Rin<r<Rout at seam','external medium at bottom/outer wall']),
      dict(region_id=5,name='EXTERNAL_AIR',radial_support='outside physical vessel or above open top',axial_support='unbounded outside finite vessel',material='air',neighbors=['acrylic outer surfaces','headspace through open bore']),
    ]
    surfaces=[
      dict(surface_id=10,name='water_air_free_surface',definition='z=Z_free_surface(x,y)',finite_support='r<Rin',side_a='water',side_b='headspace air',optical=True,fresnel=True,rng=True),
      dict(surface_id=11,name='inner_acrylic_cylinder',definition='r=Rin',finite_support='Zmin<=z<=Zwall_top',side_a='water below local free surface; headspace above',side_b='acrylic sidewall',optical=True,fresnel=True,rng=True),
      dict(surface_id=22,name='outer_acrylic_cylinder',definition='r=Rout',finite_support='Zlow<=z<=Zwall_top',side_a='acrylic',side_b='external air',optical=True,fresnel=True,rng=True),
      dict(surface_id=25,name='bottom_water_aperture',definition='z=Zmin',finite_support='r<Rin',side_a='water',side_b='acrylic bottom',optical=True,fresnel=True,rng=True),
      dict(surface_id='seam',name='bottom_sidewall_internal_continuity',definition='z=Zmin',finite_support='Rin<=r<=Rout',side_a='acrylic sidewall',side_b='acrylic bottom',optical=False,fresnel=False,rng=False),
      dict(surface_id=23,name='outer_bottom_face',definition='z=Zlow',finite_support='r<=Rout',side_a='acrylic bottom',side_b='configured bottom external medium',optical=True,fresnel=True,rng=True),
      dict(surface_id=24,name='annular_tube_top_face',definition='z=Zwall_top',finite_support='Rin<=r<=Rout',side_a='acrylic sidewall',side_b='external air',optical=True,fresnel=True,rng=True),
      dict(surface_id=13,name='open_bore_top',definition='z=Zwall_top',finite_support='r<Rin',side_a='headspace air',side_b='external air',optical=False,fresnel=False,rng=False),
    ]
    return {'release':RELEASE,'regions':regions,'surfaces':surfaces,'canonical_parameters':p}

def sediment_inventory():
    """Inventory current executable sediment cases and packaged historical study families.

    Historical rows are deliberately non-executable in V24.52; they are retained so the
    user can see every sediment campaign/validation definition still packaged with the
    release rather than mistaking archived studies for silently skipped current cases.
    """
    cfg=load_cfg();f=cfg['forward_model'];rows=[]
    common=dict(
      discovered=True,executed=False,skipped_reason='',campaign_family='V24.52 canonical sediment diagnostics',
      discovered_source='claritas_v24_52_config.json + references/v24_29_campaign_grid.csv',
      configuration_file='claritas_v24_52_config.json',default_ray_count=1_000_000,
      wavelength_nm=1e9*float(f['particle_wavelength_m']),free_surface_model=f['free_surface_model'],
      vortex_wall_center_height_difference_m=float(f['vortex_wall_center_height_difference_m']),vortex_core_radius_m=float(f['vortex_core_radius_m']),
      sediment_transport_model=f['sediment_transport_model'],particle_event_model=f['particle_event_model'],
      source_angular_model=f['source_angular_model'],source_launch_radius_m=float(f['source_launch_radius_m']),
      photodiode_response_model=f['photodiode_response_model'],detector_count=18,detector_angles_deg='0,10,...,170',
      particle_geometry='finite spherical particle proposals constrained to the physical water volume',
      h170_semantics='170-degree detector channel within this case')
    for mi,mat in enumerate(('loess','kaolin')):
      for ci,conc in enumerate((0.5,2.0,4.0)):
        rows.append(dict(case_id=f'{mat}_{conc:g}gL',material=mat,concentration_g_per_L=conc,purpose='canonical V24.52 sediment diagnostic case; same six material/concentration combinations as the historical V24.29 campaign grid',psd_definition=f'packaged {mat} PSD in core',optical_constants_file=f'particle_optical_constants_{mat}_v24_5.csv',seed=2_441_000+mi*10_000+ci*100,executable=True,**common))

    blank={k:'' for k in common}
    historical_specs=[
      dict(case_id='historical_v24_18_high_stat_validation_family',campaign_family='V24.18 high-stat sediment validation',discovered_source='references/claritas_v24_36_9_config.json:validation_v24_18',purpose='archived 15-case uniform/finite-Re drift-diffusion validation family (3 concentrations x 5 transport settings); historical physics/reference only',configuration_file='references/claritas_v24_36_9_config.json',skipped_reason='historical validation family; no V24.52 executable runner and physics is not the canonical V24.52 sediment configuration'),
      dict(case_id='historical_v24_19_meridional_first_screen_family',campaign_family='V24.19 meridional-advection first screen',discovered_source='references/claritas_v24_36_9_config.json:sediment_transport_v24_19',purpose='archived first-screen meridional circulation study with chi={0.025,0.05,0.1,0.2} at fixed diffusivity',configuration_file='references/claritas_v24_36_9_config.json',skipped_reason='historical model-screening family; retained for provenance, not a current V24.52 executable test'),
      dict(case_id='historical_v24_29_campaign_grid',campaign_family='V24.29 six-case adaptive sediment campaign',discovered_source='references/v24_29_campaign_grid.csv',purpose='archived six material/concentration cases with adaptive 5M-30M ray convergence targets; provides provenance for the current six-case material/concentration grid',configuration_file='references/v24_29_campaign_grid.csv',skipped_reason='historical adaptive campaign definition; V24.52 executes the same six material/concentration combinations through its current diagnostic runner instead'),
      dict(case_id='historical_v24_33_h170_final_turn_family',campaign_family='V24.33 H170 final-turn diagnostic',discovered_source='references/claritas_v24_36_9_config.json:h170_final_turn_angular_state_v24_33',purpose='archived diagnostic-only H170 final-turn angular-state study for loess/kaolin at 0.5, 2 and 4 g/L',configuration_file='references/claritas_v24_36_9_config.json',skipped_reason='historical diagnostic definition; current V24.52 records H170 for every executable sediment case but does not reactivate the V24.33-specific history experiment')]
    for h in historical_specs:
      d=dict(blank);d.update(discovered=True,executed=False,executable=False,material='multiple',concentration_g_per_L='',psd_definition='historical packaged definitions',optical_constants_file='',seed='',default_ray_count='',h170_semantics='historical/reference semantics');d.update(h);rows.append(d)
    return rows

def write_manifests(root:Path|None=None):
    root=HERE if root is None else Path(root)
    gm=canonical_manifest();tm=topology_manifest();inv=sediment_inventory()
    (root/'v24_52_geometry_manifest.json').write_text(json.dumps(gm,indent=2)+'\n')
    (root/'v24_52_region_surface_manifest.json').write_text(json.dumps(tm,indent=2)+'\n')
    import csv
    with (root/'sediment_case_inventory.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
    return gm,tm,inv

if __name__=='__main__':
    write_manifests();print(json.dumps(canonical_manifest(),indent=2))
