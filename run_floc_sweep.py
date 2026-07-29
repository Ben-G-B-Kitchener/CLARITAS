#!/usr/bin/env python3
"""Floc parameter sweep: tests 3 floc configurations across all 6
(material, concentration) calibration permutations.

Each config modifies CLARITAS_103.py's floc parameters, runs it for
all 6 permutations, and collects calibration comparison CSVs."""

import subprocess, sys, os, time
import numpy as np, pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLARITAS_PATH = os.path.join(THIS_DIR, "CLARITAS_103.py")
SWEEP_ROOT = os.path.join(THIS_DIR, "floc_sweep")
os.makedirs(SWEEP_ROOT, exist_ok=True)

PERMUTATIONS = [
    ("loess", 0.5), ("loess", 2.0), ("loess", 4.0),
    ("kaolin", 0.5), ("kaolin", 2.0), ("kaolin", 4.0),
]

# =========================================================================
# Floc configurations to test
# =========================================================================
FLOC_CONFIGS = {
    "A_smaller_flocs": {
        "label": "A: Smaller flocs (5-250µm, Df=2.0, eff=0.85)",
        "FLOC_POOL_EFFECTIVE_DIAMETER_M": np.array([
            5.0e-6, 8.0e-6, 12.0e-6, 20.0e-6, 30.0e-6, 40.0e-6,
            60.0e-6, 80.0e-6, 100.0e-6, 150.0e-6, 200.0e-6, 250.0e-6
        ], dtype=np.float64),
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "FLOC_SCATTER_EFFICIENCY": 0.85,
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
    },
    "B_fluffy_flocs": {
        "label": "B: Fluffy flocs (Df=1.3, eff=0.85)",
        "FLOC_POOL_EFFECTIVE_DIAMETER_M": np.array([
            40.0e-6, 50.0e-6, 60.0e-6, 70.0e-6, 80.0e-6, 90.0e-6,
            100.0e-6, 110.0e-6, 120.0e-6, 150.0e-6, 200.0e-6, 250.0e-6
        ], dtype=np.float64),
        "FLOC_FRACTAL_DIMENSION": 1.3,
        "FLOC_SCATTER_EFFICIENCY": 0.85,
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
    },
    "C_low_efficiency": {
        "label": "C: Low scatter efficiency (Df=2.0, eff=0.3)",
        "FLOC_POOL_EFFECTIVE_DIAMETER_M": np.array([
            40.0e-6, 50.0e-6, 60.0e-6, 70.0e-6, 80.0e-6, 90.0e-6,
            100.0e-6, 110.0e-6, 120.0e-6, 150.0e-6, 200.0e-6, 250.0e-6
        ], dtype=np.float64),
        "FLOC_FRACTAL_DIMENSION": 2.0,
        "FLOC_SCATTER_EFFICIENCY": 0.3,
        "FLOC_COLLISION_LENGTH_M": 250.0e-6,
    },
}

# Read the base CLARITAS_103.py once
with open(CLARITAS_PATH, "r") as f:
    base_src = f.read()

def patch_and_run(config_name, config_params, mat, conc):
    """Patch CLARITAS_103.py source with floc params + material + concentration,
    then run it."""
    label = f"{config_name}_{mat}_{conc}gpl"
    outdir = os.path.join(SWEEP_ROOT, label)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  [{config_name}] {mat} @ {conc} g/L  ->  {outdir}")
    print(f"{'='*60}")

    src = base_src

    # --- Patch floc parameters ---
    # Patch FLOC_POOL_EFFECTIVE_DIAMETER_M
    # Find the array assignment and replace it
    diam_repr = repr(list(config_params["FLOC_POOL_EFFECTIVE_DIAMETER_M"]))
    # Replace the line that starts with "FLOC_POOL_EFFECTIVE_DIAMETER_M = np.array(["
    import re
    # Match the entire array definition block
    src = re.sub(
        r'FLOC_POOL_EFFECTIVE_DIAMETER_M = np\.array\(\[[^\]]*\][\s\S]*?dtype=np\.float64\)',
        f'FLOC_POOL_EFFECTIVE_DIAMETER_M = np.array({diam_repr}, dtype=np.float64)',
        src
    )

    # Patch FLOC_FRACTAL_DIMENSION
    src = re.sub(
        r'FLOC_FRACTAL_DIMENSION = [0-9.]+',
        f'FLOC_FRACTAL_DIMENSION = {config_params["FLOC_FRACTAL_DIMENSION"]}',
        src
    )

    # Patch FLOC_SCATTER_EFFICIENCY
    src = re.sub(
        r'FLOC_SCATTER_EFFICIENCY = [0-9.]+',
        f'FLOC_SCATTER_EFFICIENCY = {config_params["FLOC_SCATTER_EFFICIENCY"]}',
        src
    )

    # Patch FLOC_COLLISION_LENGTH_M
    src = re.sub(
        r'FLOC_COLLISION_LENGTH_M = [0-9.e\-+]+',
        f'FLOC_COLLISION_LENGTH_M = {config_params["FLOC_COLLISION_LENGTH_M"]}',
        src
    )

    # --- Patch material selection ---
    src = src.replace(
        "particle_diameter_m = loess_diameter.copy()",
        f"particle_diameter_m = {mat}_diameter.copy()"
    )
    src = src.replace(
        "particle_weights = loess_weights.copy()",
        f"particle_weights = {mat}_weights.copy()"
    )

    # --- Patch concentration ---
    src = src.replace(
        "mass_concentration_g_per_L = 0.5",
        f"mass_concentration_g_per_L = {conc}"
    )

    # --- Patch OUTDIR ---
    src = src.replace('OUTDIR = "."', f'OUTDIR = {outdir!r}')
    src = src.replace("OUTDIR = '.'", f"OUTDIR = {outdir!r}")

    # Write patched script
    patched_path = os.path.join(outdir, "_cl103_patched.py")
    with open(patched_path, "w") as f:
        f.write(src)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, patched_path],
        cwd=outdir,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    elapsed = time.perf_counter() - t0

    # Print tail of output
    if result.stdout:
        tail = result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout
        print(tail)
    if result.stderr:
        tail_err = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        if "Traceback" in tail_err or "Error" in tail_err:
            print("STDERR:", tail_err, file=sys.stderr)

    comp_csv = os.path.join(outdir, "calibration_comparison_622nm.csv")
    if result.returncode != 0:
        print(f"  ❌ FAILED (rc={result.returncode}) for {label}")
        return None
    if not os.path.exists(comp_csv):
        print(f"  ❌ No comparison CSV found for {label}")
        return None

    print(f"  ✅ {label} completed in {elapsed:.1f}s")
    return comp_csv


# ==========================================================================
# RUN ALL COMBINATIONS
# ==========================================================================
all_results = {}
for config_name, config_params in FLOC_CONFIGS.items():
    print(f"\n{'#'*70}")
    print(f"  FLOC CONFIG: {config_params['label']}")
    print(f"{'#'*70}")
    results = {}
    for mat, conc in PERMUTATIONS:
        csv_path = patch_and_run(config_name, config_params, mat, conc)
        if csv_path:
            results[(mat, conc)] = csv_path
    all_results[config_name] = results
    n_ok = len(results)
    print(f"\n  {config_name}: {n_ok}/{len(PERMUTATIONS)} successful")

# ==========================================================================
# COMPARISON ANALYSIS
# ==========================================================================
print(f"\n{'='*80}")
print("  FLOC SWEEP COMPARISON")
print(f"{'='*80}")

CALIB_SUSPECT_ANGLES = [150, 160]

# Build summary
summary_rows = []
for config_name, results in all_results.items():
    for (mat, conc), csv_path in sorted(results.items()):
        df = pd.read_csv(csv_path)
        valid = df[(~df["detector_angle_deg"].isin(CALIB_SUSPECT_ANGLES)) &
                   (df["ratio_sim_over_calib"].notna()) &
                   (df["ratio_sim_over_calib"] > 0)]
        if len(valid) == 0:
            continue
        log_mean = np.mean(np.log(valid["ratio_sim_over_calib"]))
        geo_mean = np.exp(log_mean)
        summary_rows.append({
            "config": config_name,
            "material": mat,
            "concentration": conc,
            "geo_mean_ratio": geo_mean,
            "log_mean": log_mean,
        })

summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(SWEEP_ROOT, "floc_sweep_summary.csv")
summary.to_csv(summary_csv, index=False)
print(f"\n✅ Saved {summary_csv}")

# Print summary table
print(f"\n{'='*80}")
print("  GEO_MEAN RATIOS (excl. 150°/160°) — lower = closer to measured")
print(f"{'='*80}")
print(f"\n{'Config':<25s} {'Material':>8s} {'Conc':>5s}  {'geo_mean':>8s}")
print(f"{'-'*55}")

for config_name in FLOC_CONFIGS.keys():
    for row in summary_rows:
        if row["config"] == config_name:
            print(f"{config_name:<25s} {row['material']:>8s} {row['concentration']:5.1f}  {row['geo_mean_ratio']:8.3f}")

# Average per config
print(f"\n{'='*80}")
print("  AVERAGE GEO_MEAN RATIO PER CONFIG")
print(f"{'='*80}")
for config_name in FLOC_CONFIGS.keys():
    sub = summary[summary["config"] == config_name]
    if len(sub) > 0:
        avg = np.exp(np.mean(np.log(sub["geo_mean_ratio"])))
        print(f"  {config_name}: avg_geo_mean = {avg:.4f}  (n={len(sub)})")

print(f"\n{'='*80}")
print("  FLOC SWEEP COMPLETE")
print(f"{'='*80}")
print(f"  All outputs in: {SWEEP_ROOT}")