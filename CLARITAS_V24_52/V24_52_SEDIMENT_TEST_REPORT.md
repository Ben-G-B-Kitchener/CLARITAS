# CLARITAS V24.52 sediment test report — packaged candidate

**Build-environment status: NOT RUN (no compatible CuPy/CUDA runtime).**

This file records the packaged-candidate status only. `run_v24_52_campaign.py` creates the real runtime report at:

`results/V24_52_SEDIMENT_TEST_REPORT.md`

The runtime report covers all six canonical cases (loess and kaolin at 0.5, 2.0 and 4.0 g/L), detector-channel outputs including H170, ray count, seed, elapsed time, production failure/watchdog/nonfinite counts, trusted-detector comparison metrics and the output directory for each case.

Diagnostic execution and production authorization are intentionally separate. A sediment case can be marked **DIAGNOSTIC / NOT PRODUCTION-AUTHORIZED** without falsely authorizing the scientific campaign.
