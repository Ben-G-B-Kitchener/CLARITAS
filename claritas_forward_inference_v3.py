#!/usr/bin/env python3
"""Fast inference-oriented CLARITAS V24 / TARDIIS apparatus forward model.

This module re-exports the shared V24 physics core and provides a small CLI.
It intentionally contains no separate optical implementation: production V24
and PSD Inference V3 therefore use the same CUDA kernel.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from claritas_tardiis_core_v24 import (
    TardiisForwardModel, ForwardResult, DETECTOR_ANGLES_DEG,
    get_material_psd, build_geometric_transport,
)
ClaritasForwardModel = TardiisForwardModel

def main():
    ap=argparse.ArgumentParser(description='Fast CLARITAS V24 TARDIIS forward evaluation')
    ap.add_argument('--material',choices=['kaolin','loess'],required=True)
    ap.add_argument('--concentration',type=float,required=True)
    ap.add_argument('--n-rays',type=int,default=100000)
    ap.add_argument('--seed',type=int,default=24681357)
    ap.add_argument('--config',default='psd_inference_v3_config.json')
    ap.add_argument('--output',default='forward_v24_response.csv')
    ap.add_argument('--diagnostics-json',default='forward_v24_diagnostics.json')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8')) if Path(args.config).exists() else {}
    model=ClaritasForwardModel(**cfg.get('forward_model',{}))
    res=model.simulate(args.material,args.concentration,n_rays=args.n_rays,seed=args.seed)
    pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'raw_hits':res.raw_hits,'normalized_response':res.normalized_response}).to_csv(args.output,index=False)
    Path(args.diagnostics_json).write_text(json.dumps(res.to_dict(),indent=2),encoding='utf-8')
    print(f'Saved {args.output}')
    print(f'mu_geom={res.mu_geom_per_m:.6g}; optical_depth={res.optical_depth_diameter:.6g}; detected={res.raw_hits.sum()}')

if __name__=='__main__': main()
