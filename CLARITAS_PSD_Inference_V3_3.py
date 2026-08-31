#!/usr/bin/env python3
"""CLARITAS effective-PSD inference V3.3 using V24.3 exact TARDIIS detector statistics.

This is an inverse-model wrapper around the CLARITAS V24.3 3-D geometric / Snell / Fresnel particle physics plus the
physical TARDIIS cylindrical water/acrylic/air cell and detector collimators.
The shared forward implementation is ``claritas_tardiis_core_v24.py``.

V3.3 retains the broader suspension-state parameterisation introduced in V2.  Instead of a
single monotonic-looking concentration trend, each measured concentration gets
its own seven-knot smooth size-dependent PSD modification function.  The three
functions are *cross-regularised* so that they are encouraged, but not forced,
to evolve smoothly with concentration.  Each condition also gets a bounded
local/effective concentration scale factor.

Nothing in the optical forward kernel is fitted: refractive indices, Fresnel
physics, detector geometry, source geometry, sample geometry and geometric
cross sections are fixed.  The inferred quantities are suspension-state
quantities only. For the present experimental series these are interpreted as
steady/ensemble properties of a continuously magnetically stirred suspension;
there is no post-stir settling clock in V3.3:

    1. effective PSD shape at the optical sampling volume;
    2. effective local solids concentration relative to the nominal g/L.

The PSD modification is represented in log diameter with a shape-preserving
cubic PCHIP interpolator.  Non-monotonic, peaked, U-shaped and band-selective
changes are therefore allowed while avoiding spline ringing between knots.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from claritas_tardiis_core_v24_3 import TardiisForwardModel as ClaritasForwardModel, get_material_psd


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_measurements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"material", "concentration_g_per_L", "detector_deg", "measured_normalized"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Measurement CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["material"] = df["material"].astype(str).str.lower()
    for (mat, conc), g in df.groupby(["material", "concentration_g_per_L"]):
        if len(g) != 18:
            raise ValueError(f"{mat} {conc:g} g/L requires 18 detector rows; found {len(g)}")
        angles = np.sort(g.detector_deg.to_numpy(float))
        if not np.array_equal(angles, np.arange(0.0, 180.0, 10.0)):
            raise ValueError(f"{mat} {conc:g} g/L detector angles must be 0..170 in 10-degree steps")
    return df


def detector_metrics(measured: np.ndarray, modelled: np.ndarray) -> Tuple[float, float, float]:
    diff = modelled - measured
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    if np.std(measured) > 0 and np.std(modelled) > 0:
        corr = float(np.corrcoef(measured, modelled)[0, 1])
    else:
        corr = float("nan")
    return rmse, mae, corr


def centre_controls(ctrl: np.ndarray) -> np.ndarray:
    """Remove the unidentifiable uniform PSD multiplier."""
    c = np.asarray(ctrl, dtype=np.float64)
    return c - np.mean(c)


class PSDInferenceV33:
    def __init__(self, material: str, measurements: pd.DataFrame, config: dict,
                 output_dir: Path, quick: bool = False):
        self.material = material.lower()
        self.df = measurements[measurements.material == self.material].copy()
        if self.df.empty:
            raise ValueError(f"No measurements found for {self.material}")
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diameters, self.base_weights = get_material_psd(self.material)
        self.logd = np.log(self.diameters)
        self.concentrations = np.sort(self.df.concentration_g_per_L.unique().astype(float))
        if self.concentrations.size < 2:
            raise ValueError("V3.3 expects at least two measured concentrations")
        self.n_conc = self.concentrations.size
        self.n_control = int(config["n_control_points"])
        if self.n_control < 4:
            raise ValueError("n_control_points should be at least 4")
        self.ctrl_x = np.linspace(self.logd.min(), self.logd.max(), self.n_control)
        self.ctrl_diameters_m = np.exp(self.ctrl_x)

        self.max_psd_factor = float(config["max_psd_factor"])
        self.log_psd_bound = math.log(self.max_psd_factor)
        c_lo, c_hi = map(float, config["effective_concentration_scale_bounds"])
        if not (0 < c_lo < 1 < c_hi):
            raise ValueError("effective_concentration_scale_bounds must straddle 1.0")
        self.log_cscale_bounds = (math.log(c_lo), math.log(c_hi))

        self.seed = int(config["seed"])
        self.n_rays_coarse = int(config["n_rays_coarse"])
        self.n_rays_final = int(config["n_rays_final"])
        self.opt_cfg = config["optimizer"].copy()
        if quick:
            q = config["quick"]
            self.n_rays_coarse = int(q["n_rays_coarse"])
            self.n_rays_final = int(q["n_rays_final"])
            self.opt_cfg["local_maxfev"] = int(q["local_maxfev"])
            self.opt_cfg["joint_maxfev"] = int(q["joint_maxfev"])

        fm = config["forward_model"]
        self.forward = ClaritasForwardModel(**fm)

        self.n_psd_params = self.n_conc * self.n_control
        self.n_params = self.n_psd_params + self.n_conc
        self.eval_count = 0
        self.best_score = float("inf")
        self.best_parameters = np.zeros(self.n_params, dtype=np.float64)
        self.history: List[Dict[str, float]] = []
        self.cache: Dict[Tuple[int, Tuple[float, ...]], object] = {}

    # ---------------- parameter mapping ----------------
    def unpack(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = np.asarray(p, dtype=np.float64)
        if p.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {p.size}")
        controls = p[:self.n_psd_params].reshape(self.n_conc, self.n_control)
        log_cscale = p[self.n_psd_params:]
        return controls, log_cscale

    def condition_psd(self, concentration_index: int, p: np.ndarray):
        controls, log_cscale = self.unpack(p)
        ctrl = centre_controls(controls[concentration_index])
        # PCHIP allows non-monotonic knot patterns without cubic ringing.
        interpolator = PchipInterpolator(self.ctrl_x, ctrl, extrapolate=True)
        log_factor = np.asarray(interpolator(self.logd), dtype=np.float64)
        # The bound applies to the actual bin multiplier, not merely knot values.
        log_factor = np.clip(log_factor, -self.log_psd_bound, self.log_psd_bound)
        factor = np.exp(log_factor)
        effective = self.base_weights * factor
        effective /= effective.sum()
        concentration_scale = float(np.exp(np.clip(
            log_cscale[concentration_index], *self.log_cscale_bounds)))
        nominal_c = float(self.concentrations[concentration_index])
        effective_c = nominal_c * concentration_scale
        return effective, factor, ctrl, concentration_scale, effective_c

    def measured_vector(self, concentration: float):
        d = self.df[np.isclose(self.df.concentration_g_per_L, concentration)].sort_values("detector_deg")
        return d.detector_deg.to_numpy(float), d.measured_normalized.to_numpy(float)

    # ---------------- objective ----------------
    def response_loss(self, measured: np.ndarray, modelled: np.ndarray) -> Dict[str, float]:
        oc = self.config["objective"]
        floor = float(oc["log_floor"])
        lin = modelled - measured
        log_res = np.log(modelled + floor) - np.log(measured + floor)
        grad_res = np.diff(modelled) - np.diff(measured)

        high_deg = np.asarray(oc.get("high_angle_degrees", [140, 150, 160, 170]), dtype=float)
        detector_deg = np.arange(0.0, 180.0, 10.0)
        idx = [int(np.argmin(np.abs(detector_deg - x))) for x in high_deg]
        # Adjacent log ratios measure *shape* of the high-angle tail independently
        # of its overall amplitude.  This distinguishes a sharp 170-degree peak
        # from an overly broad 140-170 degree hump.
        mlog = np.log(measured[idx] + floor)
        plog = np.log(modelled[idx] + floor)
        high_shape = np.diff(plog) - np.diff(mlog)

        return {
            "linear_mse": float(np.mean(lin * lin)),
            "log_mse": float(np.mean(log_res * log_res)),
            "gradient_mse": float(np.mean(grad_res * grad_res)),
            "high_angle_shape_mse": float(np.mean(high_shape * high_shape)),
        }

    def local_regularisation(self, ctrl: np.ndarray, log_scale: float) -> Dict[str, float]:
        reg = self.config["regularization"]
        c = centre_controls(ctrl)
        dep = float(np.mean(c * c))
        rough = float(np.mean(np.diff(c, n=2) ** 2)) if c.size >= 3 else 0.0
        scale = float(log_scale * log_scale)
        return {
            "departure": dep * float(reg["departure_strength"]),
            "size_smoothness": rough * float(reg["size_smoothness_strength"]),
            "concentration_scale": scale * float(reg["concentration_scale_strength"]),
        }

    def global_regularisation(self, p: np.ndarray) -> Dict[str, float]:
        reg = self.config["regularization"]
        controls, log_scale = self.unpack(p)
        centered = np.vstack([centre_controls(x) for x in controls])

        departure = float(np.mean(centered * centered)) * float(reg["departure_strength"])
        size_smooth = 0.0
        if self.n_control >= 3:
            size_smooth = float(np.mean(np.diff(centered, n=2, axis=1) ** 2))
        size_smooth *= float(reg["size_smoothness_strength"])

        # Penalise concentration-to-concentration changes only after scaling by
        # the actual separation in log concentration.  This encourages a coherent
        # progression but does NOT impose monotonicity or a common functional form.
        logc = np.log(self.concentrations)
        cross_terms = []
        for j in range(self.n_conc - 1):
            dc = max(float(logc[j + 1] - logc[j]), 1.0e-12)
            cross_terms.append(np.mean(((centered[j + 1] - centered[j]) / dc) ** 2))
        cross = float(np.mean(cross_terms)) if cross_terms else 0.0
        cross *= float(reg["cross_concentration_strength"])

        scale_reg = float(np.mean(log_scale * log_scale)) * float(reg["concentration_scale_strength"])
        scale_smooth = 0.0
        if self.n_conc >= 3:
            # Curvature of log effective concentration factor vs log nominal concentration.
            slopes = np.diff(log_scale) / np.diff(logc)
            scale_smooth = float(np.mean(np.diff(slopes) ** 2))
        scale_smooth *= float(reg["concentration_scale_smoothness_strength"])

        return {
            "departure": departure,
            "size_smoothness": size_smooth,
            "cross_concentration": cross,
            "concentration_scale": scale_reg,
            "concentration_scale_smoothness": scale_smooth,
        }

    def weighted_response_total(self, pieces: Dict[str, float]) -> float:
        oc = self.config["objective"]
        return (
            float(oc["linear_weight"]) * pieces["linear_mse"] +
            float(oc["log_weight"]) * pieces["log_mse"] +
            float(oc["gradient_weight"]) * pieces["gradient_mse"] +
            float(oc["high_angle_shape_weight"]) * pieces["high_angle_shape_mse"]
        )

    def simulate_condition(self, j: int, p: np.ndarray, n_rays: int):
        c = float(self.concentrations[j])
        w, factor, ctrl, cscale, effective_c = self.condition_psd(j, p)
        seed = self.seed + j * 10007 + (0 if self.material == "kaolin" else 500003)
        res = self.forward.simulate(
            self.material, effective_c, weights=w, n_rays=n_rays, seed=seed)
        angles, measured = self.measured_vector(c)
        if res.normalized_response.size != measured.size:
            raise RuntimeError("Detector channel mismatch between forward model and measurements")
        pieces = self.response_loss(measured, res.normalized_response)
        return {
            "index": j, "nominal_concentration": c,
            "effective_concentration": effective_c,
            "concentration_scale": cscale,
            "weights": w, "factor": factor, "controls": ctrl,
            "result": res, "angles": angles, "measured": measured,
            "response_loss": pieces,
        }

    def evaluate(self, p: np.ndarray, n_rays: int, record: bool = True):
        p = np.asarray(p, dtype=np.float64)
        key = (int(n_rays), tuple(np.round(p, 8)))
        if key in self.cache:
            return self.cache[key]

        cond = [self.simulate_condition(j, p, n_rays) for j in range(self.n_conc)]
        response_total = float(np.mean([
            self.weighted_response_total(x["response_loss"]) for x in cond
        ]))
        reg = self.global_regularisation(p)
        total = response_total + float(sum(reg.values()))
        payload = (total, cond, response_total, reg)
        self.cache[key] = payload

        if record:
            self.eval_count += 1
            if total < self.best_score:
                self.best_score = total
                self.best_parameters = p.copy()
                mark = " *best*"
                self.write_checkpoint(p, total)
            else:
                mark = ""
            row = {"evaluation": self.eval_count, "objective": total,
                   "response_objective": response_total, "n_rays": int(n_rays),
                   "best_so_far": self.best_score}
            row.update({f"reg_{k}": v for k, v in reg.items()})
            self.history.append(row)
            if self.eval_count == 1 or self.eval_count % 10 == 0 or mark:
                print(f"[{self.material}] eval {self.eval_count:5d}: objective={total:.8g}{mark}")
        return payload

    def write_checkpoint(self, p: np.ndarray, score: float):
        payload = {
            "version": "3.3",
            "material": self.material,
            "best_objective_so_far": float(score),
            "parameters": np.asarray(p, float).tolist(),
            "parameter_count": int(self.n_params),
        }
        (self.output_dir / "checkpoint_best_parameters.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------- staged optimisation ----------------
    def local_condition_objective(self, local_p: np.ndarray, j: int, p_template: np.ndarray) -> float:
        """Fit one condition's seven controls + local concentration factor.

        This stage is a computationally efficient seed generator.  It does not
        include cross-concentration regularisation; that enters in the joint stage.
        """
        p = np.asarray(p_template, float).copy()
        start = j * self.n_control
        p[start:start + self.n_control] = local_p[:self.n_control]
        p[self.n_psd_params + j] = local_p[self.n_control]
        out = self.simulate_condition(j, p, self.n_rays_coarse)
        response = self.weighted_response_total(out["response_loss"])
        r = self.local_regularisation(local_p[:self.n_control], local_p[self.n_control])
        return response + float(sum(r.values()))

    def optimise_local_seeds(self, p0: np.ndarray) -> np.ndarray:
        p = np.asarray(p0, float).copy()
        method = self.opt_cfg.get("method", "Powell")
        xtol = float(self.opt_cfg.get("xtol", 0.025))
        ftol = float(self.opt_cfg.get("ftol", 1e-4))
        maxfev = int(self.opt_cfg.get("local_maxfev", 900))
        local_bounds = [(-self.log_psd_bound, self.log_psd_bound)] * self.n_control
        local_bounds.append(self.log_cscale_bounds)

        print("\nStage 1: concentration-specific seed fits")
        for j, c in enumerate(self.concentrations):
            start = j * self.n_control
            x0 = np.concatenate([
                p[start:start + self.n_control],
                [p[self.n_psd_params + j]],
            ])
            print(f"  {c:g} g/L: {self.n_control} PSD controls + 1 local concentration factor")
            result = minimize(
                lambda x: self.local_condition_objective(x, j, p),
                x0, method=method, bounds=local_bounds,
                options={"maxfev": maxfev, "xtol": xtol, "ftol": ftol, "disp": False})
            p[start:start + self.n_control] = result.x[:self.n_control]
            p[self.n_psd_params + j] = result.x[self.n_control]
            w, f, ctrl, cscale, ec = self.condition_psd(j, p)
            print(f"    local objective={result.fun:.7g}; C_eff={ec:.5g} g/L ({cscale:.3f}x nominal); nfev={result.nfev}")
        return p

    def joint_objective(self, p: np.ndarray) -> float:
        return self.evaluate(p, self.n_rays_coarse, record=True)[0]

    def parameter_bounds(self):
        return ([(-self.log_psd_bound, self.log_psd_bound)] * self.n_psd_params +
                [self.log_cscale_bounds] * self.n_conc)

    def load_initial_parameters(self, path: str | None) -> np.ndarray:
        if not path:
            return np.zeros(self.n_params, dtype=np.float64)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        p = np.asarray(payload.get("parameters", []), dtype=np.float64)
        if p.size != self.n_params:
            raise ValueError(f"Initial parameter file has {p.size} parameters; V3.3 expects {self.n_params}")
        print(f"Loaded initial parameters from {path}")
        return p

    def load_v1_warm_start(self, path: str) -> np.ndarray:
        """Approximate V1 effective PSDs with the V3.3 seven-knot representation."""
        df = pd.read_csv(path)
        c_col = "concentration_g_per_L" if "concentration_g_per_L" in df.columns else "nominal_concentration_g_per_L"
        required = {c_col, "diameter_m", "effective_weight"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"V1 warm-start CSV missing columns: {sorted(missing)}")
        p = np.zeros(self.n_params, dtype=np.float64)
        for j, c in enumerate(self.concentrations):
            g = df[np.isclose(df[c_col].to_numpy(float), c)].sort_values("diameter_m")
            if g.empty:
                raise ValueError(f"V1 warm-start CSV has no rows for {c:g} g/L")
            got_d = g["diameter_m"].to_numpy(float)
            got_w = g["effective_weight"].to_numpy(float)
            if got_d.size != self.diameters.size or not np.allclose(got_d, self.diameters, rtol=1e-7, atol=0):
                raise ValueError(f"V1 warm-start diameters do not match built-in {self.material} PSD bins")
            got_w = got_w / got_w.sum()
            log_ratio = np.log(np.maximum(got_w, 1e-300) / np.maximum(self.base_weights, 1e-300))
            # Sample the V1 log-ratio at the V3 knot locations, then remove the
            # uniform component because PSD renormalisation makes it unidentifiable.
            ctrl = np.interp(self.ctrl_x, self.logd, log_ratio)
            ctrl = np.clip(centre_controls(ctrl), -self.log_psd_bound, self.log_psd_bound)
            start = j * self.n_control
            p[start:start + self.n_control] = ctrl
        print(f"Warm-started V3.3 PSD controls from V1 effective PSDs: {path}")
        return p

    def optimise(self, initial_parameters: str | None = None, skip_local: bool = False,
                 v1_warm_start: str | None = None):
        print(f"\n=== {self.material.upper()} PSD inference V3.3 / V24.3 exact TARDIIS apparatus ===")
        print(f"Nominal concentrations: {self.concentrations.tolist()} g/L")
        print(f"PSD spline knots per concentration: {self.n_control}")
        print(f"Independent concentration-specific PSD functions: {self.n_conc}")
        print(f"Effective local concentration factors: {self.n_conc}")
        print(f"Total joint parameters: {self.n_params}")
        print(f"PSD bin multiplier bound: 1/{self.max_psd_factor:g} .. {self.max_psd_factor:g}")
        print(f"Local concentration factor bound: {math.exp(self.log_cscale_bounds[0]):g} .. {math.exp(self.log_cscale_bounds[1]):g}")
        print(f"Coarse rays/evaluation: {self.n_rays_coarse:,}")
        print("V24.3 TARDIIS apparatus + particle optical physics is fixed. Detector scoring is the hard physical aperture with unbiased left/right symmetry averaging; no KDE/extrapolation. Only suspension-state PSD and local concentration are inferred.\n")

        if initial_parameters and v1_warm_start:
            raise ValueError("Use either --initial-parameters or --warm-start-v1, not both")
        if v1_warm_start:
            p0 = self.load_v1_warm_start(v1_warm_start)
        else:
            p0 = self.load_initial_parameters(initial_parameters)
        if not skip_local:
            p0 = self.optimise_local_seeds(p0)

        # Evaluate the seed once through the complete regularised objective.
        seed_score = self.evaluate(p0, self.n_rays_coarse, record=True)[0]
        print(f"\nStage 2: joint cross-concentration refinement; seed objective={seed_score:.7g}")
        result = minimize(
            self.joint_objective, p0,
            method=self.opt_cfg.get("method", "Powell"),
            bounds=self.parameter_bounds(),
            options={
                "maxfev": int(self.opt_cfg.get("joint_maxfev", 2200)),
                "xtol": float(self.opt_cfg.get("xtol", 0.025)),
                "ftol": float(self.opt_cfg.get("ftol", 1e-4)),
                "disp": True,
            })

        # Use the best actually observed point, not blindly scipy's final trial.
        p_best = self.best_parameters.copy() if np.isfinite(self.best_score) else np.asarray(result.x, float)
        final_score, final_conditions, final_response, final_reg = self.evaluate(
            p_best, self.n_rays_final, record=False)
        self.write_outputs(result, p_best, final_score, final_conditions, final_response, final_reg)
        return result

    def baseline_only(self):
        p = np.zeros(self.n_params, dtype=np.float64)
        score, cond, response, reg = self.evaluate(p, self.n_rays_final, record=False)
        self.write_outputs(None, p, score, cond, response, reg, prefix="baseline_check_")
        return score

    # ---------------- outputs ----------------
    def write_outputs(self, opt_result, p_best: np.ndarray, final_score: float,
                      conditions: list, response_total: float, reg: Dict[str, float], prefix: str = ""):
        psd_rows = []
        detector_rows = []
        summary_rows = []
        ctrl_rows = []
        concentration_rows = []
        objective_rows = []

        controls, log_scales = self.unpack(p_best)

        for x in conditions:
            c = float(x["nominal_concentration"])
            ec = float(x["effective_concentration"])
            cscale = float(x["concentration_scale"])
            w = x["weights"]
            factor = x["factor"]
            res = x["result"]
            measured = x["measured"]
            rmse, mae, corr = detector_metrics(measured, res.normalized_response)

            for d, bw, ew, fac in zip(self.diameters, self.base_weights, w, factor):
                psd_rows.append({
                    "material": self.material,
                    "nominal_concentration_g_per_L": c,
                    "effective_local_concentration_g_per_L": ec,
                    "concentration_scale": cscale,
                    "diameter_m": d,
                    "diameter_um": d * 1e6,
                    "source_weight": bw,
                    "effective_weight": ew,
                    "effective_to_source_ratio": ew / bw if bw > 0 else np.nan,
                    "raw_multiplier_before_renormalization": fac,
                })

            for jj, (a, mv, mod) in enumerate(zip(res.detector_angles_deg, measured, res.normalized_response)):
                detector_rows.append({
                    "material": self.material,
                    "nominal_concentration_g_per_L": c,
                    "effective_local_concentration_g_per_L": ec,
                    "detector_deg": int(a),
                    "measured_normalized": mv,
                    "modelled_normalized": mod,
                    "modelled_normalized_native_exact": float(res.normalized_native_exact_response[jj]),
                    "modelled_normalized_mirror_exact": float(res.normalized_mirror_exact_response[jj]),
                    "symmetry_exact_physical_equivalent_score": float(res.symmetry_exact_scores[jj]),
                    "native_exact_hits": int(res.native_exact_hits[jj]),
                    "mirror_exact_hits": int(res.mirror_exact_hits[jj]),
                    "modelled_normalized_jackknife_se": float(res.normalized_response_jackknife_se[jj]),
                    "residual": mod - mv,
                })

            summary_rows.append({
                "material": self.material,
                "nominal_concentration_g_per_L": c,
                "effective_local_concentration_g_per_L": ec,
                "concentration_scale": cscale,
                "rmse": rmse,
                "mae": mae,
                "pearson_r": corr,
                "mu_geom_per_m": res.mu_geom_per_m,
                "optical_depth_diameter": res.optical_depth_diameter,
                "mean_interactions": res.mean_interactions,
                "median_interactions": res.median_interactions,
                "ballistic_fraction": res.ballistic_fraction,
                "entry_reflection_fraction": res.entry_reflection_fraction,
                "internal_reflection_fraction": res.internal_reflection_fraction,
                "mean_cell_inner_reflections": res.mean_cell_inner_reflections,
                "mean_cell_outer_reflections": res.mean_cell_outer_reflections,
                "cell_tir_fraction": res.cell_tir_fraction,
                "entered_water_fraction": res.entered_water_fraction,
                "air_exit_fraction": res.air_exit_fraction,
                "axial_loss_fraction": res.axial_loss_fraction,
                "detector_detection_fraction": res.detector_detection_fraction,
                "native_detector_detection_fraction": res.native_detector_detection_fraction,
                "mirror_detector_detection_fraction": res.mirror_detector_detection_fraction,
                "symmetry_variance_reduction": res.symmetry_variance_reduction,
                "actual_rays": res.n_rays,
            })

            j = int(x["index"])
            for k, (d_ctrl, val) in enumerate(zip(self.ctrl_diameters_m, centre_controls(controls[j]))):
                ctrl_rows.append({
                    "material": self.material,
                    "nominal_concentration_g_per_L": c,
                    "control_index": k,
                    "control_diameter_um": d_ctrl * 1e6,
                    "log_psd_multiplier_control": val,
                    "psd_multiplier_control": math.exp(np.clip(val, -self.log_psd_bound, self.log_psd_bound)),
                })
            concentration_rows.append({
                "material": self.material,
                "nominal_concentration_g_per_L": c,
                "effective_local_concentration_g_per_L": ec,
                "concentration_scale": cscale,
                "log_concentration_scale": log_scales[j],
            })
            rloss = x["response_loss"]
            objective_rows.append({
                "material": self.material,
                "nominal_concentration_g_per_L": c,
                **rloss,
                "weighted_response_objective": self.weighted_response_total(rloss),
            })

        pd.DataFrame(psd_rows).to_csv(self.output_dir / f"{prefix}effective_psds.csv", index=False)
        pd.DataFrame(detector_rows).to_csv(self.output_dir / f"{prefix}detector_fit.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(self.output_dir / f"{prefix}fit_summary.csv", index=False)
        pd.DataFrame(ctrl_rows).to_csv(self.output_dir / f"{prefix}psd_control_points.csv", index=False)
        pd.DataFrame(concentration_rows).to_csv(self.output_dir / f"{prefix}effective_concentrations.csv", index=False)
        pd.DataFrame(objective_rows).to_csv(self.output_dir / f"{prefix}objective_by_condition.csv", index=False)
        if self.history:
            pd.DataFrame(self.history).to_csv(self.output_dir / "evaluation_history.csv", index=False)

        payload = {
            "version": "3.3",
            "material": self.material,
            "final_objective": float(final_score),
            "final_response_objective": float(response_total),
            "regularisation": {k: float(v) for k, v in reg.items()},
            "n_control_points_per_concentration": self.n_control,
            "parameters": np.asarray(p_best, float).tolist(),
            "parameter_count": int(self.n_params),
            "parameterization": "independent concentration-specific PCHIP log-PSD multipliers + per-condition local concentration scale, cross-regularised",
            "nominal_concentration_fixed": False,
            "optical_physics_fixed": True,
            "detector_scoring": "V24.3 exact hard two-plane apertures with unbiased x-reflection symmetry average; native exact hits retained",
            "apparatus_physics_fixed": True,
            "stirrer_interpretation": "continuous_on_no_settling_clock",
            "mass_fraction_psd_renormalized": True,
            "effective_concentration_scale_bounds": [math.exp(self.log_cscale_bounds[0]), math.exp(self.log_cscale_bounds[1])],
            "max_psd_factor": self.max_psd_factor,
        }
        if opt_result is not None:
            payload.update({
                "optimizer_success": bool(opt_result.success),
                "optimizer_message": str(opt_result.message),
                "optimizer_fun_last_coarse": float(opt_result.fun),
                "optimizer_nfev": int(opt_result.nfev),
            })
        (self.output_dir / f"{prefix}best_fit_parameters.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

        # PSD ratio plot
        plt.figure(figsize=(8.5, 5.5))
        for x in conditions:
            plt.plot(self.diameters * 1e6, x["weights"] / self.base_weights,
                     marker="o", markersize=2, label=f"{x['nominal_concentration']:g} g/L")
        plt.xscale("log"); plt.yscale("log")
        plt.axhline(1.0, linewidth=1)
        plt.xlabel("Particle diameter (µm)")
        plt.ylabel("Effective/source PSD weight ratio")
        plt.title(f"{self.material.capitalize()} V3.3 inferred optical PSD modification")
        plt.grid(True, which="both", alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}psd_ratio.png", dpi=200); plt.close()

        # Absolute PSD plot
        plt.figure(figsize=(8.5, 5.5))
        plt.plot(self.diameters * 1e6, self.base_weights, linewidth=2, label="Source PSD")
        for x in conditions:
            plt.plot(self.diameters * 1e6, x["weights"],
                     label=f"Effective {x['nominal_concentration']:g} g/L")
        plt.xscale("log")
        plt.xlabel("Particle diameter (µm)"); plt.ylabel("Mass-fraction weight")
        plt.title(f"{self.material.capitalize()} source and V3.3 effective PSDs")
        plt.grid(True, which="both", alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}effective_psds.png", dpi=200); plt.close()

        # Detector overlay plot
        plt.figure(figsize=(9.5, 6.2))
        for x in conditions:
            c = x["nominal_concentration"]
            plt.plot(x["result"].detector_angles_deg, x["measured"], "o-", label=f"Measured {c:g} g/L")
            plt.plot(x["result"].detector_angles_deg, x["result"].normalized_response, "--",
                     label=f"V3.3 fit {c:g} g/L")
        plt.xlabel("Detector angle (deg)"); plt.ylabel("Normalized detector response")
        plt.title(f"{self.material.capitalize()} measured vs V3.3 exact TARDIIS-apparatus inferred-suspension response")
        plt.xticks(np.arange(0, 180, 10)); plt.grid(True, alpha=0.25); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}detector_fit.png", dpi=200); plt.close()

        # Residual plot
        plt.figure(figsize=(9.5, 5.5))
        for x in conditions:
            c = x["nominal_concentration"]
            residual = x["result"].normalized_response - x["measured"]
            plt.plot(x["result"].detector_angles_deg, residual, "o-", label=f"{c:g} g/L")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Detector angle (deg)"); plt.ylabel("Model - measured")
        plt.title(f"{self.material.capitalize()} V3.3 detector residuals")
        plt.xticks(np.arange(0, 180, 10)); plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}detector_residuals.png", dpi=200); plt.close()

        # Effective concentration mapping
        plt.figure(figsize=(7.5, 5.0))
        nominal = np.array([x["nominal_concentration"] for x in conditions], dtype=float)
        effective = np.array([x["effective_concentration"] for x in conditions], dtype=float)
        lo = min(nominal.min(), effective.min()); hi = max(nominal.max(), effective.max())
        plt.plot([lo, hi], [lo, hi], "--", label="C_eff = C_nominal")
        plt.plot(nominal, effective, "o-", label="Inferred")
        plt.xlabel("Nominal concentration (g/L)"); plt.ylabel("Effective local concentration (g/L)")
        plt.title(f"{self.material.capitalize()} inferred local optical-volume concentration")
        plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(self.output_dir / f"{prefix}effective_concentration.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="CLARITAS PSD Inference V3.3: V24.3 exact TARDIIS apparatus + non-monotonic effective PSD/local concentration inference")
    ap.add_argument("--material", choices=["kaolin", "loess", "all"], default="all")
    ap.add_argument("--measurements", default="measured_detector_responses_v3_3.csv")
    ap.add_argument("--config", default="psd_inference_v3_config.json")
    ap.add_argument("--output-dir", default="psd_inference_v3_results")
    ap.add_argument("--quick", action="store_true", help="250k coarse / 1M final capped-evaluation exploratory fit")
    ap.add_argument("--baseline-only", action="store_true", help="Run unchanged source PSD and nominal concentration only")
    ap.add_argument("--initial-parameters", help="Compatible V2/V3 best_fit_parameters.json or checkpoint to resume/refine")
    ap.add_argument("--warm-start-v1", help="V1 effective_psds.csv used to initialise the seven-knot V3.3 PSDs")
    ap.add_argument("--warm-start-v2", help="V2 best_fit_parameters.json/checkpoint; parameterization is compatible with V3")
    ap.add_argument("--skip-local", action="store_true", help="Skip concentration-specific seed fits and start joint fit directly")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = load_measurements(args.measurements)
    materials = ["kaolin", "loess"] if args.material == "all" else [args.material]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for mat in materials:
        inf = PSDInferenceV33(mat, df, cfg, root / mat, quick=args.quick)
        if args.baseline_only:
            score = inf.baseline_only()
            print(f"{mat} V3.3 baseline objective: {score:.8g}")
        else:
            init = args.initial_parameters or args.warm_start_v2
            if args.initial_parameters and args.warm_start_v2:
                raise ValueError("Use either --initial-parameters or --warm-start-v2, not both")
            inf.optimise(initial_parameters=init, skip_local=args.skip_local,
                         v1_warm_start=args.warm_start_v1)


if __name__ == "__main__":
    main()
