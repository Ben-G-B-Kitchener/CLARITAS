#!/usr/bin/env python3
"""Batch sweep: runs CLARITAS_100.py for all 6 (material, concentration)
calibration permutations, then generates master comparison analysis.

Approach: reads CLARITAS_100.py, patches the three config lines for each
permutation, writes a temp patched script, and runs it."""

import subprocess, sys, os, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PERMUTATIONS = [
    ("loess", 0.5), ("loess", 2.0), ("loess", 4.0),
    ("kaolin", 0.5), ("kaolin", 2.0), ("kaolin", 4.0),
]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLARITAS_PATH = os.path.join(THIS_DIR, "CLARITAS_18_17-06-2026.py")
SWEEP_ROOT = os.path.join(THIS_DIR, "calib_sweep")
os.makedirs(SWEEP_ROOT, exist_ok=True)

# Read the original CLARITAS_100.py once
with open(CLARITAS_PATH, "r") as f:
    base_src = f.read()

def run_one(mat, conc):
    label = f"{mat}_{conc}gpl"
    outdir = os.path.join(SWEEP_ROOT, label)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SWEEP: {mat} @ {conc} g/L  ->  {outdir}")
    print(f"{'='*60}")

    src = base_src

    # Patch material selection
    src = src.replace(
        "particle_diameter_m = loess_diameter",
        f"particle_diameter_m = {mat}_diameter"
    )
    src = src.replace(
        "particle_weights = loess_weights",
        f"particle_weights = {mat}_weights"
    )

    # Patch concentration
    src = src.replace(
        "mass_concentration_g_per_L = 0.5",
        f"mass_concentration_g_per_L = {conc}"
    )

    # Patch OUTDIR (try both quote styles)
    src = src.replace('OUTDIR = "."', f'OUTDIR = {outdir!r}')
    src = src.replace("OUTDIR = '.'", f"OUTDIR = {outdir!r}")

    # Write patched script
    patched_path = os.path.join(outdir, "_cl100_patched.py")
    with open(patched_path, "w") as f:
        f.write(src)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, patched_path],
        cwd=outdir,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    elapsed = time.perf_counter() - t0

    # Print tail of output
    if result.stdout:
        tail = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        print(tail)
    if result.stderr:
        tail_err = result.stderr[-800:] if len(result.stderr) > 800 else result.stderr
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
# RUN ALL PERMUTATIONS
# ==========================================================================
results = {}
for mat, conc in PERMUTATIONS:
    csv_path = run_one(mat, conc)
    if csv_path:
        results[(mat, conc)] = csv_path

if not results:
    print("No successful runs. Aborting.")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"  COLLECTING RESULTS: {len(results)}/{len(PERMUTATIONS)} successful")
print(f"{'='*60}")

# ==========================================================================
# MASTER ANALYSIS
# ==========================================================================

all_data = []
for (mat, conc), csv_path in sorted(results.items()):
    df = pd.read_csv(csv_path)
    df["material"] = mat
    df["concentration_gpl"] = conc
    all_data.append(df)

master = pd.concat(all_data, ignore_index=True)
master_csv = os.path.join(SWEEP_ROOT, "master_calibration_comparison.csv")
master.to_csv(master_csv, index=False)
print(f"\n✅ Saved {master_csv}")

CALIB_SUSPECT_ANGLES = [150, 160]

# ==========================================================================
# PLOT 1: Ratio vs angle, grouped by material, coloured by concentration
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, mat in zip(axes, ["loess", "kaolin"]):
    sub = master[master["material"] == mat].copy()
    if sub.empty:
        continue
    for conc_val in sorted(sub["concentration_gpl"].unique()):
        sc = sub[sub["concentration_gpl"] == conc_val].sort_values("detector_angle_deg")
        ax.plot(sc["detector_angle_deg"], sc["ratio_sim_over_calib"],
                marker="o", label=f"{conc_val} g/L", linewidth=1.5)
    ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5, label="Perfect match")
    for sa in CALIB_SUSPECT_ANGLES:
        ax.axvline(x=sa, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Detector angle (deg)")
    ax.set_ylabel("Ratio sim / measured")
    ax.set_title(f"Sim-to-Calibration Ratio — {mat.capitalize()}")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("Master Calibration Comparison — CLARITAS_100", fontsize=14, fontweight="bold")
plt.tight_layout()
master_png = os.path.join(SWEEP_ROOT, "master_calibration_ratio.png")
plt.savefig(master_png, dpi=200)
plt.close()
print(f"✅ Saved {master_png}")

# ==========================================================================
# PLOT 2: Normalised response overlay — all 6 conditions (log scale)
# ==========================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes_list = axes.flatten()

CALIBRATION_DATA = {
    "kaolin": {
        0.5: np.array([0.044966297, 0.040141335, 0.040635691, 0.033261336,
                       0.031612225, 0.031139978, 0.032483981, 0.031326215,
                       0.032538061, 0.035728024, 0.035407573, 0.034166179,
                       0.030915492, 0.029964220, 0.030456849, 0.031725835,
                       0.083952330, 0.087742965]),
        2.0: np.array([0.029303169, 0.026980024, 0.027588378, 0.025777957,
                       0.025938714, 0.026363886, 0.027901944, 0.027897973,
                       0.029080775, 0.033741002, 0.035486995, 0.034008434,
                       0.031570375, 0.031749041, 0.035415781, 0.043839834,
                       0.108949574, 0.126926344]),
        4.0: np.array([0.029225296, 0.026847501, 0.027449791, 0.025541964,
                       0.025585957, 0.025953808, 0.027367672, 0.027328617,
                       0.027985542, 0.030501575, 0.031240254, 0.031768086,
                       0.030057267, 0.030840993, 0.034682898, 0.044086378,
                       0.109560000, 0.138534455]),
    },
    "loess": {
        0.5: np.array([2.440330303, 0.626519136, 0.113698796, 0.043190244,
                       0.030804260, 0.028475998, 0.029407479, 0.028713983,
                       0.029475580, 0.030494678, 0.031335233, 0.033561334,
                       0.034642588, 0.035014151, 0.032513842, 0.034008468,
                       0.079254431, 0.083919911]),
        2.0: np.array([0.113367757, 0.071473857, 0.048613776, 0.030320616,
                       0.026228092, 0.025123021, 0.026763737, 0.026019845,
                       0.027114215, 0.027957246, 0.028971321, 0.031688533,
                       0.030873268, 0.030837965, 0.029129781, 0.028983729,
                       0.064161392, 0.066052410]),
        4.0: np.array([0.028640327, 0.025272048, 0.026527090, 0.023043773,
                       0.022833552, 0.022951635, 0.024207695, 0.023788577,
                       0.025601882, 0.026688448, 0.027272578, 0.029114463,
                       0.026275656, 0.027099212, 0.027270948, 0.027722322,
                       0.060134199, 0.063353778]),
    },
}
CALIBRATION_ANGLES_DEG = np.arange(0, 180, 10)

plot_order = [
    ("loess", 0.5), ("loess", 2.0), ("loess", 4.0),
    ("kaolin", 0.5), ("kaolin", 2.0), ("kaolin", 4.0),
]

for idx, (mat_val, conc_val) in enumerate(plot_order):
    ax = axes_list[idx]
    sub = master[(master["material"] == mat_val) &
                 (master["concentration_gpl"] == conc_val)]
    if sub.empty:
        ax.set_title(f"{mat_val} {conc_val} g/L \u2014 NO DATA")
        continue
    sub = sub.sort_values("detector_angle_deg")
    calib_raw = CALIBRATION_DATA[mat_val][conc_val]
    calib_norm = calib_raw / np.sum(calib_raw)
    ax.semilogy(sub["detector_angle_deg"], sub["sim_normalised"],
                marker="o", label="Sim", color="C0", linewidth=2)
    ax.semilogy(sub["detector_angle_deg"], calib_norm,
                marker="s", label="Measured", color="C1", linewidth=1.5)
    for sa in CALIB_SUSPECT_ANGLES:
        ax.axvline(x=sa, color="red", linestyle=":", alpha=0.5)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Norm. response (log)")
    ax.set_title(f"{mat_val.capitalize()} {conc_val} g/L")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)

plt.suptitle("Simulated vs Measured Angular Response \u2014 All Calibration Permutations",
             fontsize=14)
plt.tight_layout()
overlay_png = os.path.join(SWEEP_ROOT, "master_overlay_all.png")
plt.savefig(overlay_png, dpi=200)
plt.close()
print(f"\u2705 Saved {overlay_png}")

# ==========================================================================
# SUMMARY TABLE
# ==========================================================================
print(f"\n{'='*80}")
print("  MASTER SUMMARY: Sim/Calib ratio by angle")
print(f"{'='*80}")

for mat_val in ["loess", "kaolin"]:
    sub = master[master["material"] == mat_val]
    if sub.empty:
        continue
    print(f"\n--- {mat_val.upper()} ---")
    try:
        piv = sub.pivot_table(
            values="ratio_sim_over_calib",
            index="detector_angle_deg",
            columns="concentration_gpl",
            aggfunc="first"
        )
        for angle_row in piv.itertuples():
            ang = angle_row.Index
            cols = sorted(piv.columns)
            vals = "  ".join(
                f"{c:3.1f}g/L: {getattr(angle_row, str(c), np.nan):.3f}"
                for c in cols
            )
            flag = " \u26a0" if ang in CALIB_SUSPECT_ANGLES else ""
            print(f"  {ang:6.0f}\u00b0  {vals}{flag}")
    except Exception as e:
        print(f"  (pivot error: {e})")

# Geometric mean ratio per condition (excluding suspect angles)
print(f"\n{'='*80}")
print("  GEOMETRIC MEAN RATIOS (excl. 150\u00b0/160\u00b0)")
print(f"{'='*80}")
for mat_val in ["loess", "kaolin"]:
    sub = master[(master["material"] == mat_val) &
                 (~master["detector_angle_deg"].isin(CALIB_SUSPECT_ANGLES))]
    if sub.empty:
        continue
    for conc_val in sorted(sub["concentration_gpl"].unique()):
        sc = sub[sub["concentration_gpl"] == conc_val]
        valid = sc[sc["ratio_sim_over_calib"].notna() &
                    (sc["ratio_sim_over_calib"] > 0)]
        if len(valid) == 0:
            continue
        log_mean = np.mean(np.log(valid["ratio_sim_over_calib"]))
        geo_mean = np.exp(log_mean)
        print(f"  {mat_val:>8s} {conc_val:3.1f} g/L: geo_mean = {geo_mean:.4f}  "
              f"(log_mean = {log_mean:+.4f})")

# ==========================================================================
# DIAGNOSTIC: Ratio breakdown by angle region
# ==========================================================================
print(f"\n{'='*80}")
print("  ANGLE-REGION BREAKDOWN (geo_mean ratio)")
print(f"{'='*80}")

regions = {
    "0\u00b0\u201320\u00b0 (forward peak)": [0, 10, 20],
    "30\u00b0\u2013140\u00b0 (mid-range)": list(range(30, 150, 10)),
    "150\u00b0\u2013170\u00b0 (backscatter)": [150, 160, 170],
}

for region_name, region_angles in regions.items():
    print(f"\n  {region_name}:")
    for mat_val in ["loess", "kaolin"]:
        for conc_val in sorted(master["concentration_gpl"].unique()):
            sc = master[(master["material"] == mat_val) &
                        (master["concentration_gpl"] == conc_val) &
                        (master["detector_angle_deg"].isin(region_angles))]
            valid = sc[sc["ratio_sim_over_calib"].notna() &
                        (sc["ratio_sim_over_calib"] > 0)]
            if len(valid) < 2:
                print(f"    {mat_val:>8s} {conc_val:3.1f} g/L: insufficient data")
                continue
            log_mean = np.mean(np.log(valid["ratio_sim_over_calib"]))
            geo_mean = np.exp(log_mean)
            print(f"    {mat_val:>8s} {conc_val:3.1f} g/L: geo_mean = {geo_mean:.4f}"
                  f" (log_mean = {log_mean:+.4f})")

print(f"\n{'='*80}")
print("  SWEEP COMPLETE")
print(f"{'='*80}")
print(f"  All outputs in: {SWEEP_ROOT}")