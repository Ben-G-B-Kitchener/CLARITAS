# CLARITAS 77 transfer audit

## Scope and interpretation

This audit compares:

- `CLARITAS_76.py`, the previous production CUDA implementation;
- `CLARITAS_77.py`, the current production CUDA implementation;
- `claritas_reference/transport.py`;
- `claritas_reference/geometry.py`;
- `claritas_reference/physics.py`.

The two post-processing modules now called by CLARITAS 77 are also considered
where they implement requested accounting:

- `claritas_production_diagnostics.py`;
- `claritas_measured_comparison.py`.

The audit is based on the source state inspected on 27 July 2026. CUDA could not
be executed in this environment, so statements about the CUDA kernel are static
code findings, not claims of runtime validation. No Python source file was
modified while preparing this document.

The requested classification is used as follows:

1. **Physics improvement**
2. **Numerical improvement**
3. **Software engineering improvement**
4. **Validation/diagnostic improvement**
5. **Cosmetic only**

Transfer states used below are:

- **Fully transferred**: the reference algorithm is present in the production
  path, subject to CUDA runtime validation.
- **Partially transferred / deferred**: a useful part is present, but reference
  equivalence or the recommended production implementation is incomplete.
- **Retained production behaviour**: deliberately not copied from the reference.
- **Floc-model caveat**: the change touches physics absent from the validated
  primary-only reference and therefore has no reference-validation claim.

## Executive finding

CLARITAS 77 is a substantive production evolution, not another parallel
prototype. The most important reference algorithms have been transferred:

- physical solid-angle phase sampling, including the `sin(theta)` Jacobian;
- trapezoidal phase-CDF construction;
- interpolated inverse-CDF polar-angle sampling;
- full 3-D direction vectors, uniform azimuth, and normalized local-basis
  rotations;
- spherical outer-domain intersections and zero-extinction ballistic exit;
- distinct absorption and truncation outcomes;
- spherical polar detector angles, exclusive nearest-detector assignment, and
  removal of ballistic rear-detector suppression;
- deterministic host seeding and global-ray-index device stream assignment;
- an explicit centred/collimated, floc-disabled CPU-reference parity mode;
- complete per-run diagnostic and measured-shape output.

It is not yet a bit-for-bit CUDA counterpart of the CPU reference. The principal
deferred items are Philox or another high-quality counter-based GPU generator,
double-precision transport state, and CUDA execution of the validation gates.

The current floc path also needs a careful label. CLARITAS 77 converts the
internal floc domain from a 2-D circle to a 3-D sphere. That makes the code
geometrically compatible with 3-D bulk directions, but it is a real floc-model
physics change. The validated CPU reference contains no flocs and cannot
validate that decision.

## Algorithms already shared by CLARITAS 76 and the reference

These were retained rather than transferred anew:

| Algorithm | Production evidence | Reference evidence | Decision | Numerical effect | Physics effect |
|---|---|---|---|---|---|
| Mass-fraction number density, `n_i = C w_i / m_i`, in the currently selected mode | CLARITAS 76:132-158, 652-676; CLARITAS 77:132-158, 655-679 | `physics.py:63-68` | **Retained production behaviour.** Keep the explicit `PSD_WEIGHT_MODE`; only mass-fraction mode has reference parity. | None in the current mode | Changes the material population if switched |
| Complex-index primary Mie efficiencies | CLARITAS 76:750-773; CLARITAS 77:753-775 | `physics.py:70-93` | **Retained production behaviour.** | None apart from the clamp audited below | No |
| Bulk `mu_s`, `mu_a`, and `mu_t` as sums of `n_i sigma_i` | CLARITAS 76:784-790; CLARITAS 77:786-792 | `physics.py:95-100` | **Retained production behaviour.** | No | No |
| Extinction-object probability `P_i = n_i sigma_t,i / mu_t` | CLARITAS 76:809-830; CLARITAS 77:811-841 | `physics.py:102-106`, `transport.py:65-76,130` | **Retained, with validation added in CLARITAS 77.** | No for valid inputs | No |
| Per-selected-bin single-scattering albedo | CLARITAS 76:2671-2679; CLARITAS 77:2815-2826 | `physics.py:107-112`, `transport.py:131-134` | **Retained production behaviour.** | No, except the endpoint-comparison caveat below | No |
| Exponential event distance `-log(U)/mu_t` | CLARITAS 76:2622-2624; CLARITAS 77:2761-2764 | `transport.py:112` | **Retained production behaviour.** | RNG precision differs | No |

## Transfer matrix: optical probabilities and angular sampling

| ID | Algorithmic difference and evidence | Category | Transfer decision and reason | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| P1 | CLARITAS 76 built polar CDFs from raw `I(theta)` through `ANGULAR_CDF_POWER=0` (`CLARITAS_76.py:1009-1011,1107-1143`). CLARITAS 77 deletes that tuning power and hard-codes `I(theta) sin(theta)` (`CLARITAS_77.py:1060,1131-1162`), matching `physics.py:39-52,90-93`. | 1 | **Fully transferred.** The solid-angle Jacobian is physical measure, not an empirical angular correction. Retain. | Large, particularly for forward-peaked phases and repeated scattering | Yes; corrects planar probability to 3-D solid-angle probability |
| P2 | CLARITAS 76 used a pointwise cumulative sum. CLARITAS 77 uses trapezoidal increments with `cdf[0]=0` and forces the endpoint to one (`CLARITAS_77.py:1145-1157`), matching `physics.py:41-47`. | 2 | **Fully transferred.** Retain. | Small but systematic | No |
| P3 | CLARITAS 76 snapped a sampled phase angle to a table index (`CLARITAS_76.py:2492-2512,2874-2882`). CLARITAS 77 interpolates between adjacent CDF/theta entries (`CLARITAS_77.py:2566-2584,2927-2935,2992-3000,3023-3038`), matching the intent of `transport.py:48-62`. | 2 | **Fully transferred at algorithm level.** Retain; the remaining float precision is separately deferred. | Small at the 0.001-degree table spacing, but removes grid atoms | No |
| P4 | CLARITAS 76 silently retained an invalid old `cdf=np.cumsum(weights); cdf/=sum(cdf)` that did not end at one (`CLARITAS_76.py:545-546`). CLARITAS 77 removes it. | 3 | **Fully transferred cleanup.** The real extinction CDF remains the only selectable table. | None now; removes a future misuse hazard | No |
| P5 | CLARITAS 77 validates event-CDF finiteness, monotonicity, endpoint, and albedo bounds (`CLARITAS_77.py:826-854`), whereas CLARITAS 76 did not. A valid placeholder event CDF is constructed when `mu_t==0` (`832-841`), matching the reference’s bypass semantics (`physics.py:102-106`, `transport.py:100-107`). | 3 | **Fully transferred.** Retain. The zero-extinction placeholder is never sampled; it only keeps the device interface valid. | Invalid configurations now fail early; zero concentration can reach ballistic transport | No |
| P6 | Primary `sigma_s` is now unconditionally clamped into `[0,sigma_t]` and `sigma_a=sigma_t-sigma_s` (`CLARITAS_77.py:762-767`), matching `physics.py:84-88`; CLARITAS 76 only clamped when `sigma_t>0` (`759-765`). | 2 | **Fully transferred.** Retain. | Only affects pathological round-off cases | No |
| P7 | CLARITAS 76 silently sanitized invalid/all-zero phase profiles. CLARITAS 77 now requires a finite, nonnegative, nonzero 1-D intensity, a finite strictly increasing theta grid, and finite monotone device CDF tables with exact endpoints (`CLARITAS_77.py`, `_normalise_profile_for_cdf`, `build_angular_pdf_and_cdf`, and `validate_transport_tables`). | 3 | **Fully transferred and strengthened.** Invalid transport probability tables fail before CUDA allocation or launch. | Only malformed phase/configuration inputs | No; it changes failure handling |
| P8 | Phase and internal-domain caches now identify the physical measure and use new version/hash inputs (`CLARITAS_77.py:1060-1113,2105-2123,2383-2420`). | 3 | **Fully transferred.** Retain; it prevents reuse of planar CDF caches. | Changes results only by invalidating obsolete cached tables | No |
| P9 | CLARITAS 76 used the logically awkward `not (U > albedo)` scatter test. CLARITAS 77 now absorbs on `U >= albedo`, exactly equivalent to the reference rule `scatter = U < albedo`; its random uniform is also constrained to an open interval. | 2 | **Fully transferred.** Endpoint albedos now have exact semantics. | Negligible except at exact finite-lattice equality/endpoints | No |
| P10 | Production permits a list of wavelengths for phase/output loops, but primary cross sections and bulk coefficients are constructed once from `scatter_probability_wavelength = wavelengths[0]`. The CPU reference constructs the complete medium for the requested wavelength. | 1/3 | **Deferred for multi-wavelength runs.** The shipped configuration contains only 622 nm and is internally consistent. If multiple wavelengths are enabled, rebuild `sigma`, albedo, event CDF, and `mu_t` inside a wavelength-specific production preparation step rather than reusing the first wavelength. | None for the current one-wavelength configuration; potentially large otherwise | Yes, because optical coefficients are wavelength dependent |
| P11 | CLARITAS 76 tabulated `0 <= theta < 181` degrees even though physical polar scattering ends at 180 degrees; the extra tail was later suppressed only because negative `sin(theta)` values were clipped. The CPU reference uses the closed `0..pi` domain. CLARITAS 77 now uses an inclusive `0..180` grid, validates both endpoints, and changes the cache version. | 2/3 | **Fully transferred.** Retain. This removes a nonphysical, zero-mass table tail and prevents an old cache with the former shape from loading. | Normally negligible because the old tail carried zero probability; reduces table size and removes endpoint ambiguity | No change to the intended physics; it enforces the physical angular domain |

## Transfer matrix: direction, geometry, and transport

| ID | Algorithmic difference and evidence | Category | Transfer decision and reason | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| T1 | CLARITAS 76 stored `(x,y)` and `(vx,vy)` and applied signed planar turns (`CLARITAS_76.py:2576-2582,2781-2791,2882-2901`). CLARITAS 77 stores `(x,y,z)` and `(vx,vy,vz)`, samples uniform azimuth, builds a stable local basis, and renormalizes (`CLARITAS_77.py:2586-2627,2629-2677,2926-2936,3021-3044`), matching `geometry.py:18-42` and `transport.py:139-143`. | 1 | **Fully transferred.** Retain. | Large | Yes; paths now explore full solid angle |
| T2 | CLARITAS 76 used a 2-D circular outer domain (`2584-2610`). CLARITAS 77 uses the full spherical norm and quadratic distance (`2702-2729,2749-2768`), matching `geometry.py:6-15`. | 1 | **Fully transferred as the reference-parity geometry.** Retain as an explicit assumption, not as proof of the actual vessel shape. | Large | Yes; sphere is a geometry model change |
| T3 | CLARITAS 76 stepped until a heatmap sample crossed the boundary, overshooting it (`2625-2660`). CLARITAS 77 calculates the analytic boundary distance, limits the travelled segment, samples the heatmap from immutable segment start coordinates, then updates the transport position once (`CLARITAS_77.py:2749-2803`). This matches `transport.py:109-127`. | 2 | **Fully transferred.** The heatmap no longer determines endpoint or path length. | Corrects exit positions and paths; most visible near detector edges | No, for a fixed geometry |
| T4 | CLARITAS 76 classified `MU_T<=0` as absorption (`2617-2620`). CLARITAS 77 makes the boundary distance the travelled path and exits (`CLARITAS_77.py:2761-2768,2802-2803`), matching `transport.py:100-107`. | 1 | **Fully transferred.** Retain and include in the CUDA validation suite. | Complete correction at zero concentration | Restores the correct no-interaction limit |
| T5 | CLARITAS 76 classified the interaction cap as absorption (`2612-2615`). CLARITAS 77 has separate `truncated` and `terminal_state` outputs and passes the cap as an integer (`CLARITAS_77.py:2630,2667-2669,2750-2754,3055-3057,3206`). This follows `transport.py:19-24,146-150` without copying its `max_events+1` off-by-one. | 3 | **Fully transferred.** Retain. | Changes absorbed/truncated fractions for capped histories | No |
| T6 | CLARITAS 76 wrote path only for successful exits (`2916-2921`). CLARITAS 77 writes path for exited, absorbed, and truncated rays (`CLARITAS_77.py:3049-3073`) and stores explicit HDF5 outcome arrays (`3137-3164`). | 4 | **Mostly transferred.** This is sufficient for the requested path and absorption diagnostics. Final position/direction are still NaN for absorbed/truncated rays, unlike the reference’s complete state arrays; retain that only if those locations are not required. | Diagnostics only | No |
| T7 | Missed/tangent-invalid launches now receive terminal state 4, zeroed event/path bookkeeping, and remain distinct from absorption/truncation. HDF5 metadata records the state-code mapping. | 3 | **Mostly transferred.** Retain. The canonical helper deliberately reconstructs mutually exclusive states from explicit flags plus finite exit geometry for legacy compatibility. | Accounting labels only | No |
| T8 | The kernel still stores position, direction, path, `MU_T`, and theta in float, whereas the reference uses float64 throughout. Event/albedo/CDF uniforms, CDF comparisons/interpolation, and exponential free-path transforms now use double before results enter the float transport state. | 2 | **Partially transferred.** Add a full double-precision reference-parity CUDA mode first; only retain float state after quantitative CPU/GPU and float/double gates. | Usually small, but may alter boundary and long-history threshold decisions | No |
| T9 | Production’s existing mirrored-beta/Gaussian-offset source remains the default under `SOURCE_MODE="production_beta"`. CLARITAS 77 also supplies `SOURCE_MODE="reference_collimated"`, which starts at `(0,-R,0)` along `+y`, and the production screen exposes `--reference-parity`, which selects that source and disables flocs. | 1/3 | **Fully transferred as an explicit validation mode; production source deliberately retained.** This permits primary-only CPU/GPU statistical comparison without silently replacing the instrument assumption. | Yes when the selected source mode changes | Yes; source distribution is a declared physical input |
| T10 | CLARITAS 76 silently flipped any launch with `vy<0`. CLARITAS 77 removes that unmodelled reflection; an outward or missing source ray is classified as `MISSED_SAMPLE`. | 3 | **Fully transferred cleanup.** Retain. | None for the current beta support; corrects other source inputs | Removes an unphysical source transformation |
| T11 | Roughness remains a production-only optional Gaussian angular perturbation, now applied as a 3-D rotation (`CLARITAS_77.py:96-103,3003-3012,3032-3044`). It is currently zero and absent from the reference. | 1 | **Retained production behaviour, hard-disabled.** Do not enable without independent physical constraint; do not use it to fit detector curves. | None at the current zero setting | Yes if enabled |
| T12 | Heatmap accumulation remains a projected x-y diagnostic with floating atomic adds (`CLARITAS_77.py:2771-2795`). | 4 | **Retained production behaviour.** It is now decoupled from transport, which is the required transfer. Exclude heatmap values from bitwise reproducibility because atomic order is nondeterministic. | Heatmap last bits only | No |

## Transfer matrix: detector probability and accounting

| ID | Algorithmic difference and evidence | Category | Transfer decision and reason | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| D1 | CLARITAS 76 used signed planar `atan2(x,y)` and discarded one half-plane in the main detector path (`CLARITAS_76.py:4361-4364,4390-4410`). CLARITAS 77 uses `acos(y/sqrt(x²+y²+z²))` (`CLARITAS_77.py:3296-3308,4618-4623`), matching `geometry.py:45-54`. | 1 | **Fully transferred for the ideal-annular reference detector.** Retain with that model label. | Large | Yes; it changes detector geometry |
| D2 | CLARITAS 76 independently accepted a ray in every detector window, double-counting overlaps. CLARITAS 77 assigns only the nearest accepted detector (`CLARITAS_77.py:3815-3842,3845-3867,4651-4668`), matching `geometry.py:57-67` and `transport.py:150-156`. | 3 | **Fully transferred.** Retain. A ray now contributes to at most one detector. | Changes raw counts and total detected fraction | No |
| D3 | CLARITAS 76 suppressed ballistic contributions to detector centres at or above 90 degrees (`3164-3168,3323-3327,3504-3509,4404-4408`). CLARITAS 77 removes that class-specific rule from the detector helpers and main assignment. | 1 | **Fully transferred.** Retain; geometry alone should determine detection. | Rear response can change | Yes; removes an empirical selection rule |
| D4 | CLARITAS 77 stores all three exit coordinates/direction components and a polar exit-direction angle (`CLARITAS_77.py:3143-3149,4413-4421,4625-4647`), replacing CLARITAS 76’s signed planar angle whose histogram dropped negative angles (`4351-4357`). | 4 | **Fully transferred.** Retain. | Corrects bulk/exit distributions | No |
| D5 | The same exclusive assignment algorithm is used throughout, but detector indices are still recomputed in several legacy helpers and manually once in the main path rather than passed as one canonical array (`CLARITAS_77.py:3315-3345,3472-3510,3656-3694,3815-3867,4651-4669`). | 3 | **Partially transferred.** Numerically consistent now; future cleanup should compute once and pass `detector_index` everywhere to prevent rule drift. | None currently | No |
| D6 | The ideal detector is an annular band on a spherical boundary. Neither the reference nor CLARITAS 77 models finite detector area, orientation, walls, Fresnel transmission, or responsivity. | 1 | **Retained reference assumption, not instrument validation.** Report `absolute_detector_efficiency` as simulated ideal-band capture, not calibrated instrument efficiency. | Large relative to a finite real detector | Yes |

## Transfer matrix: reproducibility and chunking

| ID | Algorithmic difference and evidence | Category | Transfer decision and reason | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| R1 | CLARITAS 76 used unseeded global NumPy calls for source and per-chunk seeds (`2456,3018-3020,4115-4119`). CLARITAS 77 defines `SIMULATION_SEED`, uses `default_rng` for source angles, and derives fixed device seeds through `SeedSequence` (`CLARITAS_77.py:19,2496-2504,3166-3170,4349-4364`). | 3 | **Fully transferred for deterministic seeding.** Retain. | Makes repeated configurations reproducible in principle | No |
| R2 | CLARITAS 76 keyed device streams by chunk-local `tid` (`2559-2560`). CLARITAS 77 keys distinct transport/azimuth streams with the 64-bit `ray_index_offset+tid` and maps each supported global ID bijectively onto a nonzero xorshift32 seed modulo `2^32-1`. The host rejects a run larger than the finite unique-stream space. | 3 | **Fully transferred for chunk-independent ray identity and bounded uniqueness.** Retain and test across chunk sizes. Philox remains the correct later upgrade for a larger counter space and stronger generator quality. | Removes chunk-size dependence and silent stream wrapping within the declared limit | No |
| R3 | CLARITAS 77 still uses xorshift32, not Philox. It now maps each 32-bit word into a double open interval for exponential paths, event selection, albedo, and inverse-CDF sampling, avoiding exact endpoints and the former float-tail clipping; azimuth/gaussian draws enter float geometry. | 2 | **Partially transferred / Philox deferred.** Replace the generator with Philox or another tested counter-based engine while retaining global ray ID and double open-interval probability sampling. | Improved path/probability tails, but generator quality/correlation can still matter over large populations | No |
| R4 | CLARITAS 76’s in-loop chunk reduction could skip unwritten ray ranges because Python `range` retained the old step. CLARITAS 77 uses `while start < N`, catches allocation/launch OOM, halves the active chunk, retries the same global ray index, and advances only after all per-ray arrays are written. | 3 | **Fully transferred.** No silent gap is possible, and adaptive recovery is retained. | Large only when OOM occurs | No |
| R5 | Chunk limits are named production constants; the estimate now budgets 128 bytes/ray for roughly 96 bytes of current per-ray buffers plus allocator/conversion headroom. Allocation occurs inside the protected retry path, and initial/final chunk sizes plus retry count are saved. | 3 | **Fully transferred for correctness and practical robustness.** Static tables and the heatmap remain outside the per-ray estimate but are allocated before chunk sizing. | Affects run completion and chosen chunk, not valid completed-ray physics | No |
| R6 | HDF5 records version, master and derived seeds, RNG seed mapping and unique-initial-state limit, RNG/source/geometry/detector models, material, concentration, wavelength, PSD semantics, floc switches, phase measure/cache hash/grid, source parameters, radius, cap, ray count, optical-depth coefficients, terminal-state mapping, chunk retry provenance, and SHA-256 hashes of the exact generated Python case and embedded CUDA source. | 3 | **Fully transferred for the executable production case.** Retain; a normal repository revision can be added later without replacing the content hashes. | No | No |
| R7 | CUDA per-ray repeatability and chunk invariance have not been executed in this environment. Floating heatmap atomics cannot be bitwise invariant even when ray records are. | 4 | **Deferred validation.** Require identical per-ray HDF5 records for same GPU/seed/configuration across repeated runs and chunk sizes; compare heatmaps statistically only. | Validation only | No |

## Transfer matrix: diagnostics and measured comparison

| ID | Algorithmic difference and evidence | Category | Transfer decision and reason | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| V1 | CLARITAS 77 adds explicit HDF5 arrays for z, direction vector, path, absorbed, truncated, terminal state, floc displacement z, internal scatter count, and floc extinction count (`CLARITAS_77.py:3136-3164,4413-4435`). | 4 | **Fully transferred.** Retain. | Diagnostic outputs only | No |
| V2 | A canonical diagnostic pass now reports mutually exclusive escaped/absorbed/truncated/unclassified fractions, unique detected fraction, ballistic/single/multiple fractions, mean scatter/extinction counts, and event totals (`claritas_production_diagnostics.py:304-490`). | 4 | **Fully transferred.** Treat these `run_summary_*` fields as authoritative over older overview files. | Reported values and denominators become correct and explicit | No |
| V3 | Scatter histograms, path histograms, absorption probability versus scatter order, detector contribution versus order, detector efficiency, extinction/scatter statistics, exit-direction/position/azimuth distributions, Cartesian marginals, and integrity checks are automatically saved (`claritas_production_diagnostics.py:493-908`). | 4 | **Fully transferred.** Retain. | Diagnostics only | No |
| V4 | CLARITAS 77 adds `floc_extinction_count` before the albedo decision (`CLARITAS_77.py:2661-2664,2806-2822,3050-3054`) so canonical floc/primary event fractions can use all extinction events; successful floc/primary scattering fractions remain separate (`claritas_production_diagnostics.py:359-402,464-490`). | 4 | **Fully transferred and improved beyond the primary-only reference.** Retain. | Corrects event-fraction diagnostics | No |
| V5 | The canonical diagnostic defines ballistic as zero outer scatters, single as one, and multiple as at least two (`claritas_production_diagnostics.py:345-357`). Some older CLARITAS plotting helpers still use “quasi-ballistic 1–5” and “multiply scattered >=6” (`CLARITAS_77.py:4176-4198,4240-4265`). | 4 | **Partially transferred.** Keep old subdivisions only as compatibility diagnostics; do not call them the canonical single/multiple fractions. | Reporting only | No |
| V6 | Measured comparison now computes unit-sum RMSE, MAE, and forward/middle/rear metrics (`claritas_measured_comparison.py:346-395`) while separately reporting raw simulated detector, escaped, absorbed, truncated, and detected fractions (`894-1138`). | 4 | **Fully transferred.** Retain the separation of shape and absolute simulation accounting. | New metrics only | No |
| V7 | `sediment_data.csv` curves are explicitly recognized as shape-only; they have no absolute radiometric scale (`claritas_measured_comparison.py:3-18,184-198`). | 4 | **Correct retained interpretation.** Do not invent an absolute measured residual. “Absolute detector efficiency” is a model output only. | Prevents invalid scoring | No |
| V8 | The main script compares its configured material/concentration after every run. The existing production screen driver now copies the measured CSV into each run, invokes CLARITAS 77 with the helper modules on `PYTHONPATH`, and records shape metrics plus unique detector, escaped, absorbed, truncated, and unclassified fractions for all four named datasets. | 3 | **Fully transferred through the existing production driver.** Use `--case baseline` for the four-dataset scientific comparison without parameter tuning. | Workflow only | No |

## Floc-model caveats

The following changes cannot inherit the CPU reference’s validation because
`claritas_reference/physics.py` and `transport.py` intentionally contain only
primary-particle transport.

| ID | Floc difference and evidence | Category | Decision | Numerical-result effect | Underlying-physics effect |
|---|---|---:|---|---|---|
| F1 | CPU floc-domain diagnostics changed from a uniformly sampled 2-D chord in a circle to uniform illumination over the projected disk of a spherical floc, with a 3-D random walk (`CLARITAS_76.py:1551-1604`; `CLARITAS_77.py:1583-1655`). | 1 | **Floc-model caveat.** Geometrically sensible, but it is a floc physics change and needs dedicated tests. | Potentially large phase-profile change | Yes |
| F2 | CUDA internal floc transport changed from `(lx,ly)` circle intersections and signed planar turns to `(lx,ly,lz)` sphere intersections and uniform-azimuth 3-D rotations (`CLARITAS_76.py:2714-2835`; `CLARITAS_77.py:2843-3012`). | 1 | **Floc-model caveat.** Required for compatibility with nonzero bulk `vz`, but not independently validated. Do not describe floc-on runs as validated merely because primary transport is validated. | Potentially large, concentration-dependent | Yes |
| F3 | CPU floc diagnostics sample a distribution of impact parameters, but the CUDA true-domain encounter enters each spherical floc at the central upstream point `-v R` (`CLARITAS_77.py:1583-1596` versus `2864-2872`). | 1 | **Deferred floc consistency issue.** Choose and validate one encounter-impact distribution before scientific interpretation. | Yes | Yes |
| F4 | If the internal scatter cap is reached, CUDA projects the photon to the floc boundary and resumes it rather than marking an internal truncation (`CLARITAS_77.py:2941-2970`). | 1 | **Retained floc behaviour, not reference validated.** At minimum report the cap-hit fraction separately; ideally resolve it in the later floc redesign. | Yes when the cap is active | Yes |
| F5 | A finite-floc displacement can carry a photon outside the outer sphere, after which the loop treats the displaced position as an exit without intersecting the outer boundary along that displacement (`CLARITAS_77.py:2972-2986,3014-3018`). | 2/1 | **Deferred floc-boundary correction.** Clip the entry-to-exit displacement at the first outer-boundary intersection if that segment crosses the sample boundary. | Exit position and path can be biased for near-wall floc events | Yes through geometry |
| F6 | The pre-existing floc architecture remains: geometric floc cross-sections and mean-chord albedo are combined with representative-monomer internal scattering. The 3-D transfer does not resolve that architectural inconsistency (`CLARITAS_77.py:712-751,1026-1047,2820-3018`). | 1 | **Retained production behaviour pending floc redesign.** No new empirical parameter was added, but no new validation claim should be made. | Potentially dominant at high concentration | Yes |

## Items that are cosmetic or structural only

| Item | Category | Decision | Numerical effect | Physics effect |
|---|---:|---|---|---|
| Version banner and printed statements now identify 3-D geometry, seed, and physical phase measure (`CLARITAS_77.py:1-2,980-986,2391-2420`). | 5 | Retain; ensure wording says “spherical model assumption,” not measured vessel geometry. | No | No |
| Obsolete transport arguments `reflection_path_length`, `n_medium`, and `n_external` were removed from `trace_rays_gpu`; unused `seed1` was replaced by named transport/azimuth seeds (`CLARITAS_77.py:2675-2677,3105-3114,3251-3253,4400-4408`). | 3/5 | Fully transferred cleanup. | No | No |
| Dead reflection-probability/size-threshold constants and obsolete particle-roughness aliases were removed. Boundary Fresnel/refraction remains explicitly disabled and its outside index is labelled metadata only. | 3/5 | Retain; this prevents unused knobs from being mistaken for active detector/boundary tuning. | No | No |
| HDF5 and CSV outputs add explicit z/vector/state columns and clearer names. | 3/5 | Retain. | No | No |

## Required production follow-up, in priority order

1. **Run the CUDA acceptance suite before treating CLARITAS 77 as validated.**
   At minimum: zero-extinction ballistic exit, Beer–Lambert pure absorption,
   low-optical-depth ballistic probability, event-bin sampling, Mie phase
   sampling, 3-D isotropy, direction norm, known detector assignments,
   terminal-state closure, same-seed replay, and cross-chunk invariance.

2. **Replace xorshift32 with Philox and generate double open-interval
   uniforms.** Preserve global ray ID as the counter identity. Record the RNG
   algorithm and seed in HDF5.

3. **Run the centred/collimated source-parity mode.** The mode and production
   driver flag now exist. Use `--reference-parity` for CPU/GPU statistical
   comparison with `validate_claritas_77_reference_parity.py`; keep the
   beta/Gaussian source separately labelled and do not tune it against detector
   curves.

4. **Add a double-precision reference-parity kernel mode.** Establish CPU versus
   CUDA statistical agreement there before accepting float as an optimization.

5. **Verify provenance in workstation outputs.** Input probability tables fail
   closed, and each HDF5 records the physical/RNG/cache configuration plus
   exact Python/CUDA content hashes. A normal repository revision can be added
   later as complementary metadata.

6. **Treat the new 3-D floc domain as unvalidated floc physics.** Add dedicated
   impact-parameter, internal free-path, internal isotropy, cap-hit, and
   near-wall crossing tests before interpreting floc-on benchmark changes.

7. **Use the comprehensive diagnostic files as canonical.** Legacy plotting
   summaries with historical scatter-class labels may remain for compatibility,
   but they should not drive quantitative conclusions.

## Overall disposition

CLARITAS 77 should remain the production development branch. It has absorbed the
reference implementation’s most important transport algorithms without
replacing the CUDA project. Its primary-only 3-D transport is structurally ready
for workstation validation.

The correct present claim is:

> CLARITAS 77 contains the validated reference algorithms for solid-angle phase
> sampling, 3-D direction rotation, spherical boundary transport, exclusive
> ideal-annular detector assignment, explicit probability accounting, and a
> primary-only reference source mode, but its CUDA execution, RNG quality,
> float precision, and modified 3-D floc domain still require separate
> validation.
