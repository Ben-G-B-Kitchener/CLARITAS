#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
RUNNER=${RUNNER:-CLARITAS_24_16_06-09-2026_cuda_vortex_sediment_drift_diffusion.py}
CONFIG=${CONFIG:-claritas_v24_16_config.json}
ROOT=${ROOT:-../claritas_v24_16_sediment_transport_sweep}
TARGET_SCORE=${TARGET_SCORE:-120}
MIN_RAYS=${MIN_RAYS:-250000}
MAX_RAYS=${MAX_RAYS:-5000000}
EDDY_DIFFUSIVITY_VALUES=${EDDY_DIFFUSIVITY_VALUES:-"0.00025 0.0005 0.001 0.002"}
VORTEX_CORE_RADIUS_MM=${VORTEX_CORE_RADIUS_MM:-20}
VORTEX_DELTA_H_MM=${VORTEX_DELTA_H_MM:-50}
SLOPE=${SLOPE:-25}
HEIGHT_NM=${HEIGHT_NM:-100}

mkdir -p "$ROOT"

audit_complete() {
  local out=$1 model=$2 dt=$3
  [[ -f "$out/diagnostics.json" && -f "$out/sediment_transport_diagnostics.json" ]] || return 1
  "$PYTHON" - "$out" "$model" "$dt" "$TARGET_SCORE" "$VORTEX_CORE_RADIUS_MM" "$VORTEX_DELTA_H_MM" <<'PY' >/dev/null
from pathlib import Path
import json,sys,math
out=Path(sys.argv[1]);model=sys.argv[2];dt=float(sys.argv[3]);target=float(sys.argv[4]);a=float(sys.argv[5]);dh=float(sys.argv[6])
d=json.loads((out/'diagnostics.json').read_text());s=json.loads((out/'sediment_transport_diagnostics.json').read_text())
score=float(sum(d.get('hardware_symmetry_scores',[])));stop=str(d.get('stop_reason',''))
ok=(s.get('sediment_transport_model')==model and math.isclose(float(s.get('eddy_diffusivity_m2_s',-1)),dt,rel_tol=0,abs_tol=1e-15)
    and math.isclose(float(d.get('vortex_core_radius_mm',-1)),a,abs_tol=1e-9)
    and math.isclose(float(d.get('vortex_wall_center_height_difference_mm',-1)),dh,abs_tol=1e-9)
    and score>=target and stop.startswith('target_detector_score'))
raise SystemExit(0 if ok else 1)
PY
}

run_case() {
  local model=$1 dt=$2 conc=$3 out=$4
  if audit_complete "$out" "$model" "$dt"; then
    echo "SKIP complete: $out"
    return
  fi
  echo "RUN model=$model D_t=$dt concentration=$conc -> $out"
  "$PYTHON" "$RUNNER" \
    --material loess --concentration "$conc" --config "$CONFIG" \
    --source-angular-model beta_radiance_3d --alpha1 0.45 --alpha2 5 --source-launch-radius-mm 65.5 \
    --real-index-only --event-model hybrid_mie_excess --wave-scale 1.0 \
    --surface-rms-slope-deg "$SLOPE" --wave-rms-height-nm "$HEIGHT_NM" \
    --free-surface-model localized_scully_vortex --vortex-core-radius-mm "$VORTEX_CORE_RADIUS_MM" --vortex-delta-h-mm "$VORTEX_DELTA_H_MM" \
    --sediment-transport-model "$model" --eddy-diffusivity-m2-s "$dt" \
    --heatmap-size 0 \
    --target-detector-score "$TARGET_SCORE" --min-rays "$MIN_RAYS" --max-rays "$MAX_RAYS" \
    --output-dir "$out"
}

# Exact V24.15 uniform-suspension control at the frozen a=20 mm, D=50 mm geometry.
for C in 0.5 2 4; do
  CTAG=$(echo "$C" | sed 's/\./p/g')
  run_case uniform 0.0005 "$C" "$ROOT/uniform_control/loess_${CTAG}gL"
done

for DT in $EDDY_DIFFUSIVITY_VALUES; do
  DTTAG=$(printf '%g' "$DT" | sed 's/\./p/g; s/-/m/g')
  for C in 0.5 2 4; do
    CTAG=$(echo "$C" | sed 's/\./p/g')
    run_case stokes_drift_diffusion "$DT" "$C" "$ROOT/Dt_${DTTAG}/loess_${CTAG}gL"
  done
done

"$PYTHON" summarize_v24_16_sediment_transport_sweep.py "$ROOT"

NDT=$(printf '%s\n' "$EDDY_DIFFUSIVITY_VALUES" | wc -w)
EXPECTED=$(( (NDT + 1) * 3 ))
"$PYTHON" - "$ROOT" "$TARGET_SCORE" "$EXPECTED" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);target=float(sys.argv[2]);expected=int(sys.argv[3])
files=sorted(root.rglob('diagnostics.json'));bad=[]
for p in files:
    d=json.loads(p.read_text());score=float(sum(d.get('hardware_symmetry_scores',[])));stop=str(d.get('stop_reason',''))
    if score<target or not stop.startswith('target_detector_score'):bad.append((str(p.parent),score,stop))
print(f'completion audit: {len(files)} result cases found; expected {expected}')
if len(files)!=expected:raise SystemExit(f'Expected {expected} result cases but found {len(files)}')
if bad:
    for row in bad:print('INCOMPLETE',*row)
    raise SystemExit(f'{len(bad)} case(s) did not reach target {target:g}')
print(f'PASS: every result case reached detector score >= {target:g}')
PY
