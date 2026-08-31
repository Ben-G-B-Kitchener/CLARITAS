#!/usr/bin/env python3
"""CLARITAS V24.2: TARDIIS cell optics with multi-bandwidth detector extrapolation.

Optical ray transport is unchanged from V24.1.  Only the post-transport detector
irradiance estimator changes: physical-equivalent Epanechnikov scores are
computed at several bandwidths from the same rays and extrapolated to h -> 0.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from claritas_tardiis_core_v24_2 import TardiisForwardModel, DETECTOR_ANGLES_DEG, get_material_psd, build_geometric_transport

DEFAULTS=dict(
 n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,
 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,water_height_m=0.426,sensor_height_above_bottom_m=0.102,
 collimator_inner_plane_radius_m=0.0505,collimator_outer_plane_radius_m=0.0595,collimator_diameter_m=0.004,
 source_beam_sigma_m=1.0e-5,alpha1=1.0,alpha2=100.0,density_kg_per_m3=2600.0,
 max_internal_bounces=64,max_cell_bounces=64,chunk_size=250000,
 detector_scoring_mode='multiband_extrapolated',detector_vr_bandwidth_factors=(2.0,3.0,4.0),detector_extrapolation_power=2.0)

def parse_factors(s):
    vals=tuple(float(x.strip()) for x in str(s).split(',') if x.strip())
    if len(vals)<2: raise argparse.ArgumentTypeError('provide at least two comma-separated bandwidth factors')
    return vals

def make_model(args):
    d=DEFAULTS.copy()
    if args.config:
        cfg=json.loads(Path(args.config).read_text(encoding='utf-8')); d.update(cfg.get('forward_model',cfg))
    if args.detector_scoring: d['detector_scoring_mode']=args.detector_scoring
    if args.vr_bandwidth_factors is not None: d['detector_vr_bandwidth_factors']=args.vr_bandwidth_factors
    if args.extrapolation_power is not None: d['detector_extrapolation_power']=args.extrapolation_power
    return TardiisForwardModel(**d),d

def label_factor(v):
    return ('%g'%float(v)).replace('.','p').replace('-','m')

def save_geometry_plot(outdir,geom):
    rin=geom['tube_inner_radius_m'];rout=geom['tube_outer_radius_m'];ci=geom['collimator_inner_plane_radius_m'];co=geom['collimator_outer_plane_radius_m']
    fig,ax=plt.subplots(figsize=(7,7))
    for r,label,ls in [(rin,'water / acrylic','-'),(rout,'acrylic / air','-'),(ci,'ring inner plane','--'),(co,'collimator outer plane','--')]:
        ax.add_patch(plt.Circle((0,0),r*1000,fill=False,linestyle=ls,label=f'{label}: {2*r*1000:.1f} mm dia'))
    for deg in DETECTOR_ANGLES_DEG:
        th=np.deg2rad(deg);ax.plot(co*np.sin(th)*1000,co*np.cos(th)*1000,'o',markersize=2)
    ax.plot(0,-co*1000,'s',markersize=5,label='LED source (180°)');ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    ax.set_title('CLARITAS V24.2 TARDIIS radial optical geometry');ax.grid(True,alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(outdir/'tardiis_radial_geometry.png',dpi=200);plt.close(fig)

def save_heatmap(hm,path,title,xlabel,extent):
    if hm is None:return
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(np.log10(hm+1.0),origin='upper',extent=extent,aspect='auto')
    ax.set_xlabel(xlabel);ax.set_ylabel('beam axis y (mm)');ax.set_title(title);fig.colorbar(im,ax=ax,label='log10(path samples + 1)');fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description='CLARITAS V24.2 TARDIIS transport + multi-bandwidth zero-bandwidth detector extrapolation')
    ap.add_argument('--material',choices=['loess','kaolin'],default='loess');ap.add_argument('--concentration',type=float,default=.5);ap.add_argument('--n-rays',type=int,default=500000)
    ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--output-dir');ap.add_argument('--config');ap.add_argument('--dgb-only',action='store_true')
    ap.add_argument('--heatmap-size',type=int,default=1024);ap.add_argument('--save-rays',action='store_true')
    ap.add_argument('--detector-scoring',choices=['multiband_extrapolated','smallest_bandwidth','exact'],default='multiband_extrapolated')
    ap.add_argument('--vr-bandwidth-factors',type=parse_factors,default=(2.0,3.0,4.0),help='comma separated, default 2,3,4')
    ap.add_argument('--extrapolation-power',type=float,default=2.0,help='leading KDE bias power; default 2')
    args=ap.parse_args();c=0.0 if args.dgb_only else args.concentration;tag='DGB_water_only' if args.dgb_only else f'{args.material}_{c:g}gL'
    outdir=Path(args.output_dir or f'claritas_v24_2_results/{tag}');outdir.mkdir(parents=True,exist_ok=True);model,geom=make_model(args)
    print('\n=========== CLARITAS V24.2 — MULTI-BANDWIDTH TARDIIS DETECTORS ===========')
    print(f'material={args.material}; concentration={c:g} g/L; rays={args.n_rays:,}')
    print(f'bandwidth factors={model.detector_vr_bandwidth_factors.tolist()}; extrapolation power={model.detector_extrapolation_power:g}')
    print('Optical physics: unchanged from V24.1; only detector scoring estimator changed.')
    print('Stirring interpretation: continuous_on; no settling-time model.\n')
    res=model.simulate(args.material,c,n_rays=args.n_rays,seed=args.seed,collect_rays=args.save_rays,heatmap_size=args.heatmap_size)
    pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'exact_physical_hits':res.raw_hits}).to_csv(outdir/'detector_hits_exact.csv',index=False)
    wide={'Detector_deg':res.detector_angles_deg.astype(int),'normalized_response':res.normalized_response,
          'normalized_multiband_extrapolated':res.normalized_multiband_extrapolated_response,'normalized_exact':res.normalized_exact_response,
          'extrapolated_physical_equivalent_score':res.multiband_extrapolated_physical_equivalent_scores,
          'extrapolated_raw_intercept':res.multiband_extrapolated_raw_scores,'extrapolation_slope':res.multiband_fit_slope,
          'extrapolation_fit_r2':res.multiband_fit_r2,'bandwidth_relative_spread':res.multiband_relative_spread,'exact_physical_hits':res.raw_hits}
    long=[]
    for bi,bf in enumerate(res.multiband_bandwidth_factors):
        lab=label_factor(bf);wide[f'normalized_bw_{lab}']=res.multiband_normalized_responses[bi];wide[f'physical_equivalent_bw_{lab}']=res.multiband_physical_equivalent_scores[bi];wide[f'support_rays_bw_{lab}']=res.multiband_support_counts[bi]
        for j,a in enumerate(res.detector_angles_deg): long.append({'Detector_deg':int(a),'bandwidth_factor':float(bf),'normalized_response':res.multiband_normalized_responses[bi,j],'physical_equivalent_score':res.multiband_physical_equivalent_scores[bi,j],'kernel_sum':res.multiband_kernel_sums[bi,j],'support_rays':int(res.multiband_support_counts[bi,j])})
    pd.DataFrame(wide).to_csv(outdir/'detector_response_normalized.csv',index=False);pd.DataFrame(long).to_csv(outdir/'detector_scores_multiband.csv',index=False)
    (outdir/'diagnostics.json').write_text(json.dumps(res.to_dict(),indent=2),encoding='utf-8');geom_out=geom.copy();geom_out['stirrer_state_for_interpretation']='continuous_on';geom_out['detector_angles_deg']=DETECTOR_ANGLES_DEG.tolist();(outdir/'apparatus_geometry.json').write_text(json.dumps(geom_out,indent=2),encoding='utf-8')
    save_geometry_plot(outdir,geom)
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(res.detector_angles_deg,res.normalized_multiband_extrapolated_response,'o-',label='h→0 extrapolated');ax.plot(res.detector_angles_deg,res.normalized_exact_response,'--',alpha=.65,label='exact')
    for bi,bf in enumerate(res.multiband_bandwidth_factors):ax.plot(res.detector_angles_deg,res.multiband_normalized_responses[bi],alpha=.55,label=f'bw={bf:g}')
    ax.set_xlabel('Detector angle (deg)');ax.set_ylabel('Normalized detector response');ax.set_xticks(DETECTOR_ANGLES_DEG);ax.set_title(f'CLARITAS V24.2 TARDIIS response — {tag}');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(outdir/'detector_response_normalized.png',dpi=200);plt.close(fig)
    if args.heatmap_size>0:
        R=model.rin*1000;save_heatmap(res.heatmap_xy,outdir/'water_path_xy.png',f'{tag}: x-y water-path projection','x (mm)',[-R,R,R,-R]);save_heatmap(res.heatmap_zy,outdir/'water_path_zy.png',f'{tag}: z-y water-path projection','z relative to sensor plane (mm)',[model.zmin*1000,model.zmax*1000,R,-R])
    if args.save_rays and res.ray_data is not None:pd.DataFrame(res.ray_data).to_csv(outdir/'ray_states.csv',index=False)
    d,w=get_material_psd(args.material);tr=build_geometric_transport(d,w,c,model.density);pd.DataFrame({'diameter_um':d*1e6,'source_weight':w,'event_weight':tr['particle_event_weights'],'mu_geom_by_bin_per_m':tr['mu_geom_by_bin']}).to_csv(outdir/'particle_interaction_diagnostics.csv',index=False)
    print(f'Saved V24.2 results to {outdir}');print(f'Exact hits={int(res.raw_hits.sum())}; extrapolated physical-equivalent total={res.multiband_extrapolated_physical_equivalent_scores.sum():.3f}')
    for bi,bf in enumerate(res.multiband_bandwidth_factors):print(f'  bw={bf:g}: support min={res.multiband_support_counts[bi].min()}, median={np.median(res.multiband_support_counts[bi]):.1f}, max={res.multiband_support_counts[bi].max()}')
    print(f'Extrapolation fit R2 median={np.nanmedian(res.multiband_fit_r2):.4f}; raw negative intercept channels={np.count_nonzero(res.multiband_extrapolated_raw_scores<0)}')

if __name__=='__main__':main()
