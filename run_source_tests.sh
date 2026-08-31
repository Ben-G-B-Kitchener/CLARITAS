#!/usr/bin/env bash
set -euo pipefail

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py \
  --material loess --concentration 0.5 --n-rays 1000000 \
  --config source_a2_25.json --output-dir source_test/loess_05_a2_25

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py \
  --material loess --concentration 0.5 --n-rays 1000000 \
  --config source_a2_50.json --output-dir source_test/loess_05_a2_50

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py \
  --material loess --concentration 0.5 --n-rays 1000000 \
  --config source_a2_100.json --output-dir source_test/loess_05_a2_100

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py \
  --material loess --concentration 0.5 --n-rays 1000000 \
  --config source_a2_200.json --output-dir source_test/loess_05_a2_200

echo "All four source-divergence tests completed."
