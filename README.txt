CLARITAS V24.3 source-divergence diagnostic pack

Purpose
-------
Run four otherwise identical loess 0.5 g/L V24.3 tests while varying only the
inherited Beta source angular-distribution parameter alpha2.

Files
-----
source_a2_25.json
source_a2_50.json
source_a2_100.json   (current V24.3 default)
source_a2_200.json
run_source_tests.bat (Windows Command Prompt)
run_source_tests.sh  (Git Bash / Linux shell)

Usage
-----
Place this folder beside:
  CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py
  claritas_tardiis_core_v24_3.py

Then run either:
  run_source_tests.bat
or:
  bash run_source_tests.sh

Each test launches 1,000,000 rays and writes to a separate subdirectory under
source_test/.

Please return the four detector_response_normalized.csv files from:
  source_test/loess_05_a2_25/
  source_test/loess_05_a2_50/
  source_test/loess_05_a2_100/
  source_test/loess_05_a2_200/

No particle physics, cell geometry, detector aperture, concentration, PSD, or
random-seed setting is changed by these four config files. Only alpha2 changes.
