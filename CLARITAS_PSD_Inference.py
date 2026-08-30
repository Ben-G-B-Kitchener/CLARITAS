#!/usr/bin/env python3
"""Infer smooth concentration-dependent effective PSDs from detector responses.

This program treats CLARITAS as a forward model and changes only the PSD shape.
Nominal mass concentration is held fixed for every experimental condition.

Default parameterisation
------------------------
For each material, five control points are placed uniformly in log particle diameter.
The log multiplier applied to the source PSD is

    log S(d, C) = A(d) + B(d) * log(C / C_ref)

where A and B are linearly interpolated between the control points. The control
multipliers are centred before application because a uniform multiplier vanishes
when the PSD is renormalised. This gives a smooth, low-dimensional PSD change that
is shared across all three concentrations rather than fitting three unrelated PSDs.

The objective combines linear detector-response error and a weak log-response error,
plus regularisation for excessive departure from the source PSD, rough size-to-size
variation and unnecessarily large concentration dependence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from claritas_forward_inference import ClaritasForwardModel, get_material_psd


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_measurements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"material", "concentration_g_per_L", "detector_deg", "measured_normalized"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measurement CSV missing columns: {sorted(missing)}")
    df["material"] = df["material"].str.lower()
    return df


def centered_controls(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


def effective_psd(diameters_m: np.ndarray, base_weights: np.ndarray,
                  concentration: float, parameters: np.ndarray,
                  n_control: int, c_ref: float):
    a = centered_controls(parameters[:n_control])
    b = centered_controls(parameters[n_control:2*n_control])
    log_c = math.log(float(concentration) / float(c_ref))
    ctrl = centered_controls(a + b * log_c)
    logd = np.log(diameters_m)
    ctrl_x = np.linspace(logd.min(), logd.max(), n_control)
    log_factor = np.interp(logd, ctrl_x, ctrl)
    factor = np.exp(log_factor)
    w = base_weights * factor
    w /= w.sum()
    return w, factor, ctrl, ctrl_x


def detector_metrics(measured: np.ndarray, modelled: np.ndarray):
    diff = modelled - measured
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    if np.std(measured) > 0 and np.std(modelled) > 0:
        corr = float(np.corrcoef(measured, modelled)[0,1])
    else:
        corr = float("nan")
    return rmse, mae, corr


class PSDInference:
    def __init__(self, material: str, measurements: pd.DataFrame, config: dict,
                 output_dir: Path, quick: bool = False):
        self.material = material.lower()
        self.df = measurements[measurements.material == self.material].copy()
        if self.df.empty:
            raise ValueError(f"No measurements found for {self.material}")
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.diameters, self.base_weights = get_material_psd(self.material)
        self.concentrations = np.sort(self.df.concentration_g_per_L.unique().astype(float))
        self.n_control = int(config["n_control_points"])
        self.c_ref = float(np.exp(np.mean(np.log(self.concentrations))))
        self.seed = int(config["seed"])
        self.n_rays_coarse = int(config["n_rays_coarse"])
        self.n_rays_final = int(config["n_rays_final"])
        self.opt_cfg = config["optimizer"].copy()
        if quick:
            self.n_rays_coarse = min(self.n_rays_coarse, 25_000)
            self.n_rays_final = min(self.n_rays_final, 100_000)
            self.opt_cfg["maxiter"] = min(int(self.opt_cfg.get("maxiter", 30)), 6)
        fm = config["forward_model"]
        self.forward = ClaritasForwardModel(
            n_medium=fm["n_medium"], n_particle=fm["n_particle"],
            sample_radius_m=fm["sample_radius_m"], ray_offset_m=fm["ray_offset_m"],
            detector_acceptance_deg=fm["detector_acceptance_deg"],
            alpha1=fm["alpha1"], alpha2=fm["alpha2"],
            density_kg_per_m3=fm["density_kg_per_m3"],
            max_internal_bounces=fm["max_internal_bounces"],
            chunk_size=fm["chunk_size"])
        self.eval_count = 0
        self.best_score = float("inf")
        self.history = []
        self.cache = {}

    def measured_vector(self, concentration: float):
        d = self.df[np.isclose(self.df.concentration_g_per_L, concentration)].sort_values("detector_deg")
        return d.detector_deg.to_numpy(float), d.measured_normalized.to_numpy(float)

    def regularisation(self, p: np.ndarray):
        reg = self.config["regularization"]
        a = centered_controls(p[:self.n_control])
        b = centered_controls(p[self.n_control:2*self.n_control])
        baseline = float(reg["baseline_strength"]) * np.mean(a*a)
        slope = float(reg["concentration_slope_strength"]) * np.mean(b*b)
        rough = 0.0
        for c in self.concentrations:
            _, _, ctrl, _ = effective_psd(self.diameters, self.base_weights, c, p, self.n_control, self.c_ref)
            if ctrl.size >= 3:
                rough += np.mean(np.diff(ctrl, n=2)**2)
        rough *= float(reg["size_smoothness_strength"]) / max(len(self.concentrations),1)
        return baseline + slope + rough

    def evaluate(self, p: np.ndarray, n_rays: int, record: bool = True):
        # Rounded cache is safe only because common random numbers make evaluation deterministic.
        key = (int(n_rays), tuple(np.round(np.asarray(p, float), 8)))
        if key in self.cache:
            return self.cache[key]
        obj_cfg = self.config["objective"]
        lin_w = float(obj_cfg["linear_weight"])
        log_w = float(obj_cfg["log_weight"])
        floor = float(obj_cfg["log_floor"])
        total = 0.0
        condition_outputs = []
        for j, c in enumerate(self.concentrations):
            w, factor, ctrl, ctrl_x = effective_psd(
                self.diameters, self.base_weights, c, p, self.n_control, self.c_ref)
            seed = self.seed + j*10007 + (0 if self.material == "kaolin" else 500003)
            res = self.forward.simulate(self.material, c, weights=w, n_rays=n_rays, seed=seed)
            angles, measured = self.measured_vector(c)
            modelled = res.normalized_response
            if modelled.size != measured.size:
                raise RuntimeError("Detector channel mismatch between forward model and measurements")
            lin = modelled - measured
            log_res = np.log(modelled + floor) - np.log(measured + floor)
            total += lin_w * float(np.mean(lin*lin)) + log_w * float(np.mean(log_res*log_res))
            rmse, mae, corr = detector_metrics(measured, modelled)
            condition_outputs.append((c, w, factor, ctrl, res, measured, rmse, mae, corr))
        total /= len(self.concentrations)
        total += self.regularisation(np.asarray(p,float))
        self.cache[key] = (total, condition_outputs)
        if record:
            self.eval_count += 1
            if total < self.best_score:
                self.best_score = total
                mark = " *best*"
            else:
                mark = ""
            self.history.append({"evaluation":self.eval_count,"objective":total,"n_rays":n_rays,"best_so_far":self.best_score})
            if self.eval_count == 1 or self.eval_count % 5 == 0 or mark:
                print(f"[{self.material}] eval {self.eval_count:4d}: objective={total:.7g}{mark}")
        return total, condition_outputs

    def objective(self, p):
        return self.evaluate(p, self.n_rays_coarse, record=True)[0]

    def optimise(self):
        max_factor = float(self.config["max_psd_factor"])
        bound = math.log(max_factor)
        n_param = 2*self.n_control
        x0 = np.zeros(n_param, dtype=np.float64)
        bounds = [(-bound, bound)] * n_param
        print(f"\n=== {self.material.upper()} PSD inference ===")
        print(f"Concentrations: {self.concentrations.tolist()} g/L")
        print(f"Control points: {self.n_control}; parameters: {n_param}; C_ref={self.c_ref:.6g} g/L")
        print(f"Coarse rays/evaluation: {self.n_rays_coarse:,}")
        print("Nominal concentration is fixed; only PSD shape is changed.\n")
        result = minimize(
            self.objective, x0, method=self.opt_cfg.get("method","Powell"), bounds=bounds,
            options={"maxiter":int(self.opt_cfg.get("maxiter",30)),
                     "xtol":float(self.opt_cfg.get("xtol",0.03)),
                     "ftol":float(self.opt_cfg.get("ftol",1e-4)), "disp":True})
        p_best = np.asarray(result.x,float)
        # Final high-statistics result; do not mix this into optimisation history.
        final_score, final_conditions = self.evaluate(p_best, self.n_rays_final, record=False)
        self.write_outputs(result, p_best, final_score, final_conditions)
        return result

    def baseline_only(self):
        p = np.zeros(2*self.n_control)
        score, cond = self.evaluate(p, self.n_rays_final, record=False)
        self.write_outputs(None, p, score, cond, prefix="baseline_check_")
        return score

    def write_outputs(self, opt_result, p_best, final_score, conditions, prefix=""):
        # PSD table
        psd_rows=[]; det_rows=[]; summary=[]
        for c,w,factor,ctrl,res,measured,rmse,mae,corr in conditions:
            for d,bw,ew,fac in zip(self.diameters,self.base_weights,w,factor):
                psd_rows.append({"material":self.material,"concentration_g_per_L":c,
                                 "diameter_m":d,"diameter_um":d*1e6,
                                 "source_weight":bw,"effective_weight":ew,
                                 "effective_to_source_ratio":ew/bw if bw>0 else np.nan,
                                 "raw_multiplier_before_renormalization":fac})
            for a,mv,mod,h in zip(res.detector_angles_deg,measured,res.normalized_response,res.raw_hits):
                det_rows.append({"material":self.material,"concentration_g_per_L":c,"detector_deg":int(a),
                                 "measured_normalized":mv,"modelled_normalized":mod,"raw_model_hits":int(h),
                                 "residual":mod-mv})
            summary.append({"material":self.material,"concentration_g_per_L":c,
                            "rmse":rmse,"mae":mae,"pearson_r":corr,
                            "mu_geom_per_m":res.mu_geom_per_m,
                            "optical_depth_diameter":res.optical_depth_diameter,
                            "mean_interactions":res.mean_interactions,
                            "median_interactions":res.median_interactions,
                            "ballistic_fraction":res.ballistic_fraction,
                            "entry_reflection_fraction":res.entry_reflection_fraction,
                            "internal_reflection_fraction":res.internal_reflection_fraction})
        pd.DataFrame(psd_rows).to_csv(self.output_dir/f"{prefix}effective_psds.csv",index=False)
        pd.DataFrame(det_rows).to_csv(self.output_dir/f"{prefix}detector_fit.csv",index=False)
        pd.DataFrame(summary).to_csv(self.output_dir/f"{prefix}fit_summary.csv",index=False)
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_dir/"evaluation_history.csv",index=False)

        payload={"material":self.material,"final_objective":final_score,"c_ref_g_per_L":self.c_ref,
                 "n_control_points":self.n_control,"parameters":p_best.tolist(),
                 "parameterization":"log S(d,C)=A(d)+B(d)*log(C/C_ref), linearly interpolated in log diameter",
                 "nominal_concentration_fixed":True,"mass_fraction_renormalized":True}
        if opt_result is not None:
            payload["optimizer_success"]=bool(opt_result.success)
            payload["optimizer_message"]=str(opt_result.message)
            payload["optimizer_fun_coarse"]=float(opt_result.fun)
            payload["optimizer_nfev"]=int(opt_result.nfev)
        (self.output_dir/f"{prefix}best_fit_parameters.json").write_text(json.dumps(payload,indent=2))

        # PSD ratio plot
        plt.figure(figsize=(8,5))
        for c,w,factor,ctrl,res,measured,rmse,mae,corr in conditions:
            plt.plot(self.diameters*1e6, w/self.base_weights, marker='o', markersize=2, label=f"{c:g} g/L")
        plt.xscale('log'); plt.yscale('log')
        plt.axhline(1.0,linewidth=1)
        plt.xlabel('Particle diameter (µm)'); plt.ylabel('Effective/source PSD weight ratio')
        plt.title(f'{self.material.capitalize()} inferred optical PSD modification')
        plt.grid(True,which='both',alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir/f"{prefix}psd_ratio.png",dpi=200); plt.close()

        # PSD absolute plot
        plt.figure(figsize=(8,5))
        plt.plot(self.diameters*1e6,self.base_weights,linewidth=2,label='Source PSD')
        for c,w,factor,ctrl,res,measured,rmse,mae,corr in conditions:
            plt.plot(self.diameters*1e6,w,label=f'Effective {c:g} g/L')
        plt.xscale('log'); plt.xlabel('Particle diameter (µm)'); plt.ylabel('Mass-fraction weight')
        plt.title(f'{self.material.capitalize()} source and inferred effective PSDs')
        plt.grid(True,which='both',alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir/f"{prefix}effective_psds.png",dpi=200); plt.close()

        # Detector overlay plot
        plt.figure(figsize=(9,6))
        for c,w,factor,ctrl,res,measured,rmse,mae,corr in conditions:
            plt.plot(res.detector_angles_deg, measured, 'o-', label=f'Measured {c:g} g/L')
            plt.plot(res.detector_angles_deg, res.normalized_response, '--', label=f'Inferred PSD {c:g} g/L')
        plt.xlabel('Detector angle (deg)'); plt.ylabel('Normalized detector response')
        plt.title(f'{self.material.capitalize()} measured vs inferred-PSD CLARITAS response')
        plt.xticks(np.arange(0,180,10)); plt.grid(True,alpha=0.25); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(self.output_dir/f"{prefix}detector_fit.png",dpi=200); plt.close()


def main():
    ap=argparse.ArgumentParser(description='Infer smooth effective PSDs using the V23.2 optical forward model')
    ap.add_argument('--material',choices=['kaolin','loess','all'],default='all')
    ap.add_argument('--measurements',default='measured_detector_responses.csv')
    ap.add_argument('--config',default='psd_inference_config.json')
    ap.add_argument('--output-dir',default='psd_inference_results')
    ap.add_argument('--quick',action='store_true',help='Very low ray count / iteration smoke test')
    ap.add_argument('--baseline-only',action='store_true',help='Run source PSDs only; do not optimise')
    args=ap.parse_args()
    cfg=load_config(args.config); df=load_measurements(args.measurements)
    materials=['kaolin','loess'] if args.material=='all' else [args.material]
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    for mat in materials:
        inf=PSDInference(mat,df,cfg,root/mat,quick=args.quick)
        if args.baseline_only:
            score=inf.baseline_only(); print(f'{mat} baseline objective: {score:.7g}')
        else:
            inf.optimise()

if __name__=='__main__':
    main()
