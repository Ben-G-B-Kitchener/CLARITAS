#!/usr/bin/env python3
"""CLARITAS V24.4: reconstructed TARDIIS sensor-ring collimator geometry.

V24.4 is a narrow apparatus-geometry release. It preserves the V23.2 particle
interaction law and V24 cylindrical water/acrylic/air optics. The selected
response uses the hard reconstructed sensor-ring bore from the TARDIIS drawing.
The original CLARITAS +/-6.5-degree exit-position acceptance is exported from
the same rays as a diagnostic comparison only.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from claritas_tardiis_core_v24_4 import TardiisForwardModel,DETECTOR_ANGLES_DEG,get_material_psd,build_geometric_transport

DEFAULTS=dict(
 n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,
 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
 water_height_m=0.426,sensor_height_above_bottom_m=0.102,
 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,
 counterbore_depth_m=0.008,through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,
 source_launch_radius_m=0.0655,legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1e-5,
 alpha1=1.0,alpha2=100.0,density_kg_per_m3=2600.0,max_internal_bounces=64,max_cell_bounces=64,
 chunk_size=250000,statistics_batch_rays=250000,symmetry_average_exact=True)

def make_model(args):
 d=DEFAULTS.copy()
 if args.config:
  cfg=json.loads(Path(args.config).read_text());d.update(cfg.get('forward_model',cfg))
 return TardiisForwardModel(**d),d

def main():
 ap=argparse.ArgumentParser(description='CLARITAS V24.4 reconstructed TARDIIS collimator + exact hard-wall scoring')
 ap.add_argument('--material',choices=['loess','kaolin'],default='loess');ap.add_argument('--concentration',type=float,default=0.5)
 ap.add_argument('--n-rays',type=int,default=1000000,help='fixed ray count unless --target-detector-score is used')
 ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--output-dir');ap.add_argument('--config');ap.add_argument('--dgb-only',action='store_true')
 ap.add_argument('--heatmap-size',type=int,default=0,help='0 is recommended for high-statistics detector runs')
 ap.add_argument('--save-rays',action='store_true');ap.add_argument('--target-detector-score',type=float,help='adaptive selected-hardware score target')
 ap.add_argument('--min-rays',type=int);ap.add_argument('--max-rays',type=int);ap.add_argument('--stability-l1-tolerance',type=float)
 args=ap.parse_args();conc=0.0 if args.dgb_only else args.concentration;tag='DGB_water_only' if args.dgb_only else f'{args.material}_{conc:g}gL'
 out=Path(args.output_dir or f'claritas_v24_4_results/{tag}');out.mkdir(parents=True,exist_ok=True);model,geom=make_model(args)
 throat_outer=model.throat_out
 half_angle=math.degrees(math.atan2(model.throat_radius,model.source_launch_radius-model.ring_in))
 print('\n=========== CLARITAS V24.4 — RECONSTRUCTED TARDIIS COLLIMATOR ===========')
 print(f'material={args.material}; concentration={conc:g} g/L; requested rays={args.n_rays:,}')
 print('Particle optics: V23.2 geometric encounters + 3-D Snell/Fresnel (unchanged)')
 print('Cell optics: V24 cylindrical water/acrylic/air Fresnel/reflection/TIR (unchanged)')
 print(f'Ring: ID={2*model.ring_in*1e3:.1f} mm; OD={2*model.ring_out*1e3:.1f} mm')
 print(f'Bore: {2*model.throat_radius*1e3:.1f} mm dia through, narrow length={(throat_outer-model.ring_in)*1e3:.1f} mm; counterbore {2*model.counterbore_radius*1e3:.1f} mm dia x {model.counterbore_depth*1e3:.1f} mm')
 print(f'Source launch radius={model.source_launch_radius*1e3:.1f} mm (configurable; paper does not locate the LED die plane)')
 print(f'Legacy comparison: +/-{model.legacy_acceptance_deg:g} deg original-CLARITAS exit-position scorer')
 print(f'Centered source-to-inner-throat geometric half-angle ~{half_angle:.3f} deg (diagnostic only)')
 print('Selected detector response: reconstructed hard mechanical bore + x-reflection symmetry average. No KDE.')
 print('Stirring interpretation: continuous-on ensemble; no settling clock.')
 res=model.simulate(args.material,conc,n_rays=args.n_rays,seed=args.seed,collect_rays=args.save_rays,heatmap_size=args.heatmap_size,
  target_detector_score=args.target_detector_score,min_rays=args.min_rays,max_rays=args.max_rays,stability_l1_tolerance=args.stability_l1_tolerance)
 df=pd.DataFrame({
  'Detector_deg':res.detector_angles_deg.astype(int),
  'normalized_response':res.normalized_response,
  'normalized_hardware_symmetry_exact':res.normalized_hardware_symmetry_response,
  'normalized_hardware_native_exact':res.normalized_hardware_native_response,
  'normalized_hardware_mirror_exact':res.normalized_hardware_mirror_response,
  'hardware_jackknife_se_normalized':res.hardware_normalized_jackknife_se,
  'hardware_symmetry_physical_equivalent_score':res.hardware_symmetry_scores,
  'hardware_native_hits':res.hardware_native_hits,'hardware_mirror_hits':res.hardware_mirror_hits,
  'normalized_legacy_6p5_symmetry':res.normalized_legacy_symmetry_response,
  'normalized_legacy_6p5_native':res.normalized_legacy_native_response,
  'normalized_legacy_6p5_mirror':res.normalized_legacy_mirror_response,
  'legacy_6p5_jackknife_se_normalized':res.legacy_normalized_jackknife_se,
  'legacy_6p5_symmetry_channel_score':res.legacy_symmetry_channel_scores,
  'legacy_6p5_native_channel_count':res.legacy_native_channel_counts,
  'legacy_6p5_mirror_channel_count':res.legacy_mirror_channel_counts,
 })
 df.to_csv(out/'detector_response_normalized.csv',index=False)
 pd.DataFrame({
  'Detector_deg':res.detector_angles_deg.astype(int),
  'hardware_native_hits':res.hardware_native_hits,'hardware_mirror_hits':res.hardware_mirror_hits,
  'hardware_symmetry_physical_equivalent_score':res.hardware_symmetry_scores,
  'legacy_6p5_native_channel_count':res.legacy_native_channel_counts,
  'legacy_6p5_mirror_channel_count':res.legacy_mirror_channel_counts,
  'legacy_6p5_symmetry_channel_score':res.legacy_symmetry_channel_scores,
 }).to_csv(out/'detector_hits_exact.csv',index=False)
 pd.DataFrame(res.batch_diagnostics or []).to_csv(out/'batch_convergence.csv',index=False)
 (out/'diagnostics.json').write_text(json.dumps(res.to_dict(),indent=2))
 geom.update({
  'stirrer_state_for_interpretation':'continuous_on',
  'detector_estimator':'reconstructed_hard_bore_symmetry_average_exact',
  'through_bore_outer_radius_m':model.throat_out,
  'through_bore_length_m':model.throat_out-model.ring_in,
  'source_launch_radius_m':model.source_launch_radius,
  'centered_source_to_inner_throat_half_angle_deg_diagnostic':half_angle,
  'legacy_comparison':'original CLARITAS +/-6.5 deg boundary-exit-position channel scorer; overlapping channels; not selected',
  'paper_dimension_choice':'Fig.3/Fig.4 machining values: 4 mm through bore, 8.7 mm counterbore x 8 mm deep; section 5.3 prose states 3 mm/10 mm and is treated as conflicting text.'})
 (out/'apparatus_geometry.json').write_text(json.dumps(geom,indent=2))
 d,w=get_material_psd(args.material);tr=build_geometric_transport(d,w,conc,model.density)
 pd.DataFrame({'diameter_um':d*1e6,'source_weight':w,'event_weight':tr['particle_event_weights'],'mu_geom_by_bin_per_m':tr['mu_geom_by_bin']}).to_csv(out/'particle_interaction_diagnostics.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,5));ax.errorbar(res.detector_angles_deg,res.normalized_hardware_symmetry_response,yerr=res.hardware_normalized_jackknife_se,fmt='o-',capsize=2,label='V24.4 reconstructed hardware')
 ax.plot(res.detector_angles_deg,res.normalized_legacy_symmetry_response,'--',label=f'legacy +/-{res.legacy_acceptance_deg:g} deg comparison')
 ax.set_xticks(DETECTOR_ANGLES_DEG);ax.set_xlabel('Detector angle (deg)');ax.set_ylabel('Normalized response');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'detector_response_normalized.png',dpi=200);plt.close(fig)
 if args.save_rays and res.ray_data is not None: pd.DataFrame(res.ray_data).to_csv(out/'ray_states.csv',index=False)
 print(f'Saved V24.4 results to {out}')
 print(f'actual rays={res.n_rays:,}; stop={res.stop_reason}; hardware native={res.hardware_native_hits.sum():,}; mirror={res.hardware_mirror_hits.sum():,}; equivalent={res.hardware_symmetry_scores.sum():.1f}')
 print(f'hardware detection fraction selected={res.detector_detection_fraction:.6g}; legacy channel-score/ray={res.legacy_channel_score_per_ray:.6g}')
if __name__=='__main__':main()
