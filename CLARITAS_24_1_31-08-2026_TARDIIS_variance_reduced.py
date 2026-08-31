#!/usr/bin/env python3
"""CLARITAS V24.1: TARDIIS-specific 3-D cylindrical-cell optical transport.

Particle physics remains the successful Mie-free V23.2 model: geometric
sphere encounters plus full 3-D Snell/Fresnel physics.  V24.1 retains the V24 replacement of the artificial spherical sample
boundary and abstract detector caps with the actual TARDIIS radial apparatus: water cylinder, acrylic tube wall, air gap,
and physical 4-mm detector/source collimator apertures.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from claritas_tardiis_core_v24_1 import TardiisForwardModel, DETECTOR_ANGLES_DEG, get_material_psd, build_geometric_transport

DEFAULTS = dict(
    n_water=1.33,
    n_particle=1.59,
    n_acrylic=1.4906,
    n_air=1.0,
    tube_inner_radius_m=0.0465,
    tube_outer_radius_m=0.0500,
    water_height_m=0.426,
    sensor_height_above_bottom_m=0.102,
    collimator_inner_plane_radius_m=0.0505,
    collimator_outer_plane_radius_m=0.0595,
    collimator_diameter_m=0.004,
    source_beam_sigma_m=1.0e-5,
    alpha1=1.0,
    alpha2=100.0,
    density_kg_per_m3=2600.0,
    max_internal_bounces=64,
    max_cell_bounces=64,
    chunk_size=250000,
    detector_scoring_mode='variance_reduced',
    detector_vr_bandwidth_factor=3.0,
)

def make_model(args):
    d=DEFAULTS.copy()
    if args.config:
        cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
        d.update(cfg.get('forward_model', cfg))
    if getattr(args,'detector_scoring',None): d['detector_scoring_mode']=args.detector_scoring
    if getattr(args,'vr_bandwidth_factor',None) is not None: d['detector_vr_bandwidth_factor']=args.vr_bandwidth_factor
    return TardiisForwardModel(**d), d

def save_geometry_plot(outdir: Path, geom: dict):
    rin=geom['tube_inner_radius_m']; rout=geom['tube_outer_radius_m']
    ci=geom['collimator_inner_plane_radius_m']; co=geom['collimator_outer_plane_radius_m']
    fig,ax=plt.subplots(figsize=(7,7))
    for r,label,ls in [(rin,'water / acrylic','-'),(rout,'acrylic / air','-'),(ci,'ring inner plane','--'),(co,'collimator outer plane','--')]:
        ax.add_patch(plt.Circle((0,0),r*1000,fill=False,linestyle=ls,label=f'{label}: {2*r*1000:.1f} mm dia'))
    for deg in DETECTOR_ANGLES_DEG:
        th=np.deg2rad(deg); x=co*np.sin(th)*1000; y=co*np.cos(th)*1000
        ax.plot(x,y,'o',markersize=2)
    # source at 180 deg
    ax.plot(0,-co*1000,'s',markersize=5,label='LED source (180°)')
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('CLARITAS V24.1 TARDIIS radial optical geometry')
    ax.grid(True,alpha=.25); ax.legend(fontsize=8,loc='upper right'); fig.tight_layout()
    fig.savefig(outdir/'tardiis_radial_geometry.png',dpi=200); plt.close(fig)

def save_heatmap(hm, path: Path, title: str, xlabel: str, extent):
    if hm is None: return
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(np.log10(hm+1.0),origin='upper',extent=extent,aspect='auto')
    ax.set_xlabel(xlabel); ax.set_ylabel('beam axis y (mm)'); ax.set_title(title)
    fig.colorbar(im,ax=ax,label='log10(path samples + 1)'); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description='CLARITAS V24.1 TARDIIS cylindrical-cell transport with variance-reduced physical detector scoring')
    ap.add_argument('--material',choices=['loess','kaolin'],default='loess')
    ap.add_argument('--concentration',type=float,default=0.5,help='nominal/effective g/L used by this forward run')
    ap.add_argument('--n-rays',type=int,default=500000)
    ap.add_argument('--seed',type=int,default=24681357)
    ap.add_argument('--output-dir',help='default: claritas_v24_1_results/<material>_<concentration>gL')
    ap.add_argument('--config',help='optional JSON containing a forward_model object or forward-model keys')
    ap.add_argument('--dgb-only',action='store_true',help='force zero sediment concentration to model water-only device geometry baseline')
    ap.add_argument('--heatmap-size',type=int,default=1024,help='0 disables projected water-path heatmaps')
    ap.add_argument('--save-rays',action='store_true',help='save per-ray external exit/state CSV (large)')
    ap.add_argument('--detector-scoring',choices=['variance_reduced','exact'],default='variance_reduced',help='selected detector response estimator; exact binary hits are always saved')
    ap.add_argument('--vr-bandwidth-factor',type=float,default=3.0,help='phase-space kernel bandwidth / physical aperture radius; >=1, default 3.0')
    args=ap.parse_args()
    concentration=0.0 if args.dgb_only else args.concentration
    tag='DGB_water_only' if args.dgb_only else f'{args.material}_{concentration:g}gL'
    outdir=Path(args.output_dir or f'claritas_v24_1_results/{tag}'); outdir.mkdir(parents=True,exist_ok=True)
    model,geom=make_model(args)
    print('\n=========== CLARITAS V24.1 — TARDIIS CELL OPTICS + VR DETECTORS ===========')
    print(f'material={args.material}; concentration={concentration:g} g/L; rays={args.n_rays:,}')
    print(f'water ID={2*model.rin*1000:.1f} mm; acrylic OD={2*model.rout*1000:.1f} mm')
    print(f'n_water={model.n_water:.5f}; n_acrylic={model.n_acrylic:.5f}; n_air={model.n_air:.5f}; n_particle={model.n_particle:.5f}')
    print('Particle physics: geometric pi*r^2 encounter + full 3-D Snell/Fresnel (no Mie)')
    print('Cell physics: cylindrical water/acrylic/air interfaces + multiple wall reflections/TIR')
    print('Detector geometry: physical two-plane 4-mm collimator; exact binary hits retained')
    print(f'Detector estimator: {model.detector_scoring_mode}; VR bandwidth factor={model.detector_vr_bandwidth_factor:g}')
    print('Stirring interpretation: continuous; no settling-time model is applied.')
    print('===========================================================\n')
    res=model.simulate(args.material,concentration,n_rays=args.n_rays,seed=args.seed,collect_rays=args.save_rays,heatmap_size=args.heatmap_size)
    pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'exact_physical_hits':res.raw_hits}).to_csv(outdir/'detector_hits_exact.csv',index=False)
    pd.DataFrame({
        'Detector_deg':res.detector_angles_deg.astype(int),
        'normalized_response':res.normalized_response,
        'normalized_variance_reduced':res.normalized_variance_reduced_response,
        'normalized_exact':res.normalized_exact_response,
        'variance_reduced_physical_equivalent_score':res.variance_reduced_physical_equivalent_scores,
        'variance_reduced_kernel_sum':res.variance_reduced_kernel_sums,
        'variance_reduced_support_rays':res.variance_reduced_support_counts,
        'exact_physical_hits':res.raw_hits,
    }).to_csv(outdir/'detector_response_normalized.csv',index=False)
    pd.DataFrame({
        'Detector_deg':res.detector_angles_deg.astype(int),
        'physical_equivalent_score':res.variance_reduced_physical_equivalent_scores,
        'kernel_sum':res.variance_reduced_kernel_sums,
        'support_rays':res.variance_reduced_support_counts,
    }).to_csv(outdir/'detector_scores_variance_reduced.csv',index=False)
    (outdir/'diagnostics.json').write_text(json.dumps(res.to_dict(),indent=2),encoding='utf-8')
    geom_out=geom.copy(); geom_out['source_angle_deg']=180; geom_out['detector_angles_deg']=DETECTOR_ANGLES_DEG.tolist()
    geom_out['stirrer_state_for_interpretation']='continuous_on'
    (outdir/'apparatus_geometry.json').write_text(json.dumps(geom_out,indent=2),encoding='utf-8')
    save_geometry_plot(outdir,geom)
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(res.detector_angles_deg,res.normalized_variance_reduced_response,'o-',label='variance-reduced')
    ax.plot(res.detector_angles_deg,res.normalized_exact_response,'--',alpha=.65,label='exact binary')
    ax.set_xlabel('Detector angle (deg)'); ax.set_ylabel('Normalized detector response'); ax.set_xticks(DETECTOR_ANGLES_DEG)
    ax.set_title(f'CLARITAS V24.1 TARDIIS response — {tag}'); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(outdir/'detector_response_normalized.png',dpi=200); plt.close(fig)
    if args.heatmap_size>0:
        R=model.rin*1000
        save_heatmap(res.heatmap_xy,outdir/'water_path_xy.png',f'{tag}: x-y water-path projection','x (mm)',[-R,R,R,-R])
        save_heatmap(res.heatmap_zy,outdir/'water_path_zy.png',f'{tag}: z-y water-path projection','z relative to sensor plane (mm)',[model.zmin*1000,model.zmax*1000,R,-R])
    if args.save_rays and res.ray_data is not None:
        pd.DataFrame(res.ray_data).to_csv(outdir/'ray_states.csv',index=False)
    d,w=get_material_psd(args.material); tr=build_geometric_transport(d,w,concentration,model.density)
    pd.DataFrame({'diameter_um':d*1e6,'source_weight':w,'event_weight':tr['particle_event_weights'],'mu_geom_by_bin_per_m':tr['mu_geom_by_bin']}).to_csv(outdir/'particle_interaction_diagnostics.csv',index=False)
    print(f'Saved V24.1 results to {outdir}')
    print(f'Exact detected ray fraction: {res.detector_detection_fraction:.6g}')
    print(f'VR physical-equivalent detector fraction: {res.detector_vr_equivalent_fraction:.6g}')
    print(f'VR support rays per detector: min={res.variance_reduced_support_counts.min()}, median={np.median(res.variance_reduced_support_counts):.1f}, max={res.variance_reduced_support_counts.max()}')
    print(f'Entered-water fraction: {res.entered_water_fraction:.6g}; air-exit fraction: {res.air_exit_fraction:.6g}; axial-loss fraction: {res.axial_loss_fraction:.6g}')
    print(f'Mean cell inner reflections/ray: {res.mean_cell_inner_reflections:.6g}; outer: {res.mean_cell_outer_reflections:.6g}; cell-TIR ray fraction: {res.cell_tir_fraction:.6g}')

if __name__=='__main__': main()
