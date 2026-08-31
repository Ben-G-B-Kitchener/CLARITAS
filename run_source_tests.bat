@echo off
setlocal

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material loess --concentration 0.5 --n-rays 1000000 --config source_a2_25.json --output-dir source_test\loess_05_a2_25
if errorlevel 1 exit /b %errorlevel%

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material loess --concentration 0.5 --n-rays 1000000 --config source_a2_50.json --output-dir source_test\loess_05_a2_50
if errorlevel 1 exit /b %errorlevel%

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material loess --concentration 0.5 --n-rays 1000000 --config source_a2_100.json --output-dir source_test\loess_05_a2_100
if errorlevel 1 exit /b %errorlevel%

python CLARITAS_24_3_31-08-2026_TARDIIS_exact_streaming.py --material loess --concentration 0.5 --n-rays 1000000 --config source_a2_200.json --output-dir source_test\loess_05_a2_200
if errorlevel 1 exit /b %errorlevel%

echo.
echo All four source-divergence tests completed.
endlocal
