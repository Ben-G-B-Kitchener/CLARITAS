# CLARITAS 3-D reference model

This directory contains a CPU-only, primary-particle reference implementation.
It is independent of the legacy monolithic scripts and contains no floc model.

Quick start:

```bash
python3 -m claritas_reference.validate --output claritas_reference/validation_outputs
python3 -m claritas_reference.run_benchmarks --rays 100000 --output claritas_reference/benchmark_outputs
```

The model requires only the packages already used by CLARITAS: NumPy, pandas,
and miepython.

