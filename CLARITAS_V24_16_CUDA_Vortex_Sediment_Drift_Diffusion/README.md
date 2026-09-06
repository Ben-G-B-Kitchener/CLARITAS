# CLARITAS V24.16 — CUDA Vortex Sediment Drift–Diffusion

V24.16 follows the V24.15 localized-free-surface sweep. V24.15 showed that a localized vortex funnel behaves much more plausibly than the V24.14 full-cell paraboloid, but free-surface geometry alone does not reproduce the concentration-dependent TARDIIS angular response. V24.16 therefore freezes the best non-destructive V24.15 localized surface case and changes **only the spatial sediment field**.

The release remains process based. It does not add a material-specific, concentration-specific, PSD-specific, detector-specific or angular scattering multiplier. Instead, the continuous stir is represented as a size-resolved particle drift–diffusion problem, and the existing optical event laws act on the resulting local number density.

## Frozen optical/apparatus baseline

- DGB-calibrated `beta_radiance_3d` source, alpha1=0.45, alpha2=5, launch radius 65.5 mm.
- Exact reconstructed TARDIIS hard-bore detector scorer with x-reflection variance reduction.
- Mean water height 142 mm; sensor/beam plane 93 mm above the bottom; 3-mm acrylic bottom disc.
- Nominal material PSD and requested bulk mass concentration; no aggregation or fragmentation.
- Reference particle morphology: RMS surface slope 25°, q=1.0, wave RMS height 100 nm, isotropic orientation.
- Real-index-only particle-profile comparison path and `hybrid_mie_excess` event model retained for the supplied sweep.
- V24.15 localized Scully free surface fixed by default at core radius `a=20 mm` and wall-to-centre depth `D=50 mm`.
- Top-transmitted rays remain headspace exits; air-headspace re-entry is not introduced here.

The V24.15 surface is

```text
v_theta(r) = Omega_c r / [1 + (r/a)^2]
h(r)       = Hinf r^2 / (a^2 + r^2)
Hinf       = D (a^2 + R^2) / R^2
hbar       = Hinf [1 - (a^2/R^2) ln(1 + R^2/a^2)]
z_s(r)     = Zmean - hbar + h(r)
```

so the measured mean water volume is preserved. The inferred `Omega_c` is a geometry-consistent velocity parameter, not a claim about measured magnetic-stirrer RPM.

## V24.16 particle-transport closure

For PSD bin `i`, diameter `d_i`, particle density `rho_p`, water density `rho_w` and dynamic viscosity `mu`, the Stokes response time is

```text
tau_i = rho_p d_i^2 / (18 mu).
```

The outward inertial slip induced by the swirling carrier flow is approximated by

```text
w_r,i(r) = tau_i v_theta(r)^2 / r.
```

The downward Stokes settling velocity is

```text
w_s,i = (rho_p-rho_w) g d_i^2 / (18 mu).
```

Unresolved turbulent stirring and recirculating mixing are represented by a scalar effective eddy diffusivity `D_t`. This is a **particle-transport parameter**, not an optical correction. The first V24.16 closure assumes the same `D_t` in the radial and vertical directions.

For steady zero net particle flux,

```text
J_r = n_i w_r,i - D_t dn_i/dr = 0
J_z = -n_i w_s,i - D_t dn_i/dz = 0.
```

Using the frozen Scully velocity profile gives

```text
n_i(r,z) / nbar_i = exp[A_i f(r) - B_i (z-zbottom)] / N_i
f(r)              = r^2/(a^2+r^2)
A_i               = tau_i Omega_c^2 a^2/(2 D_t)
B_i               = w_s,i/D_t.
```

For numerical stability and an exact Woodcock majorant, the CUDA implementation evaluates the equivalent peak-scaled form

```text
S_i(r,z) = exp[A_i (f(r)-f(R)) - B_i (z-zbottom)] / Nscaled_i.
```

The unnormalised peak is therefore 1 at the bottom outer wall. `Nscaled_i` is the vessel-volume mean of the peak-scaled exponential. Each PSD bin is independently normalised over the **actual curved V24.15 water volume**, so

```text
< S_i >_vessel = 1
```

for every size bin. Consequently the integrated number and mass in every PSD bin are conserved, and the requested bulk concentration and bulk PSD are unchanged. Only their spatial distribution is altered.

## CUDA optical transport through a non-uniform suspension

A spatially varying extinction/event coefficient cannot use the historical single exponential free path directly. V24.16 therefore adds CUDA Woodcock/delta tracking.

For each PSD bin the exact maximum local density scale is

```text
S_i,max = 1 / Nscaled_i,
```

at the bottom outer wall. The global majorant is constructed from the sum of each optical event-bin coefficient times its own `S_i,max`. A candidate free path is sampled from this constant majorant. At the candidate location CUDA evaluates all local size-bin scales, forms the local geometric/wave event coefficients, accepts the candidate with

```text
P_accept = mu_local / mu_majorant,
```

and, on acceptance, samples the interacting particle-size bin from the **local** event-rate distribution. Rejected candidates are null collisions and the ray continues unchanged.

The `uniform` transport branch remains explicit. It bypasses Woodcock acceptance and uses the historical V24.15 free-path/event-bin path so it provides a clean regression control.

## Default V24.16 screen

The free-surface geometry is fixed at the V24.15 non-destructive candidate

```text
a = 20 mm
D = 50 mm
```

and the supplied screen varies only the transport mixing strength:

```text
D_t = 2.5e-4, 5e-4, 1e-3, 2e-3 m^2/s
concentration = 0.5, 2, 4 g/L
```

plus a uniform-suspension control at each concentration. This gives 15 cases and isolates sediment redistribution from the previously tested surface geometry. `v24_16_sediment_transport_grid.csv` records the host-reference transport-strength diagnostics for the four supplied `D_t` values.

Run the checks and sweep on the CUDA workstation with:

```bash
rm -rf ~/.cupy/kernel_cache
python verify_v24_16_cuda_sediment_transport.py
bash run_v24_16_sediment_transport_sweep.sh
```

The default adaptive screen uses detector target score 120, minimum 250k rays and maximum 5M rays per case. For higher-statistics confirmation, for example:

```bash
TARGET_SCORE=300 MIN_RAYS=500000 MAX_RAYS=10000000 \
  bash run_v24_16_sediment_transport_sweep.sh
```

The sweep writes `v24_16_case_summary.csv` and `v24_16_joint_ranking.csv` together with per-case detector curves, transport diagnostics, convergence files and PSD-bin transport tables.

## New diagnostics

Each run writes `sediment_transport_diagnostics.json` and `sediment_transport_by_size.csv`. The size-resolved diagnostics include `tau_i`, settling velocity, radial coefficient `A_i`, vertical coefficient `B_i`, volume normalisation, maximum local density scale, and selected local scales at the sensor plane and near the bottom. The run diagnostics also report geometric, wave and total Woodcock majorants.

The ranking retains complete-curve RMSE/correlation and the previous physically useful angular diagnostics, including H170, H170/H160, the 0–30°, 40–130° and 140–170° sectors, detector score and uncertainty. A large H170/H160 value alone is not treated as evidence of success.

## Scope boundary

V24.16 is deliberately the first transport closure, not CFD. It assumes Stokes particle slip and a scalar effective eddy diffusivity. It does not yet include finite-Reynolds-number drag, anisotropic/tensor turbulent diffusivity, an explicit meridional carrier-flow field, particle-wall resuspension/deposition kinetics, particle-particle interactions, aggregation/fragmentation, air-core entrainment, or feedback of sediment loading on the carrier flow.

These limitations are intentional: V24.16 asks a clean question — **can physically generated, size-dependent sediment redistribution from the observed stirred vortex account for the remaining TARDIIS angular discrepancy while conserving sediment mass?**
