#!/usr/bin/env python3
"""Fast inference-oriented CLARITAS V24.1 / TARDIIS apparatus forward model.

This module re-exports the shared V24.1 physics core and provides a small CLI.
It intentionally contains no separate optical implementation: production V24.1
and PSD Inference V3 therefore use the same CUDA kernel.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from claritas_tardiis_core_v24_1 import (
    TardiisForwardModel, ForwardResult, DETECTOR_ANGLES_DEG,
    get_material_psd, build_geometric_transport,
)
ClaritasForwardModel = TardiisForwardModel

def main():
    ap=argparse.ArgumentParser(description='Fast CLARITAS V24.1 TARDIIS forward evaluation with VR detector scoring')
    ap.add_argument('--material',choices=['kaolin','loess'],required=True)
    ap.add_argument('--concentration',type=float,required=True)
    ap.add_argument('--n-rays',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=24681357)
    ap.add_argument('--config',default='psd_inference_v3_1_config.json')
    ap.add_argument('--output',default='forward_v24_1_response.csv')
    ap.add_argument('--diagnostics-json',default='forward_v24_1_diagnostics.json')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8')) if Path(args.config).exists() else {}
    model=ClaritasForwardModel(**cfg.get('forward_model',{}))
    res=model.simulate(args.material,args.concentration,n_rays=args.n_rays,seed=args.seed)
    pd.DataFrame({
        'Detector_deg':res.detector_angles_deg.astype(int),
        'normalized_response':res.normalized_response,
        'normalized_variance_reduced':res.normalized_variance_reduced_response,
        'normalized_exact':res.normalized_exact_response,
        'physical_equivalent_score':res.variance_reduced_physical_equivalent_scores,
        'support_rays':res.variance_reduced_support_counts,
        'exact_physical_hits':res.raw_hits,
    }).to_csv(args.output,index=False)
    Path(args.diagnostics_json).write_text(json.dumps(res.to_dict(),indent=2),encoding='utf-8')
    print(f'Saved {args.output}')
    print(f'mu_geom={res.mu_geom_per_m:.6g}; optical_depth={res.optical_depth_diameter:.6g}; exact_hits={res.raw_hits.sum()}; vr_equivalent={res.variance_reduced_physical_equivalent_scores.sum():.3f}')

if __name__=='__main__': main()
