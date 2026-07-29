# CLARITAS physics implementation review

Date: 27 July 2026  
Compared implementations: `CLARITAS_18_17-06-2026.py` and `CLARITAS_76.py`  
Scope: read-only code and existing-output review; no simulation code was modified

## 1. Executive summary

### Main conclusion

CLARITAS_18 did not achieve its Loess 0.5 g/L agreement because it contained a more correct, uniquely identified physical model. It achieved that agreement with a particular coupled surrogate:

- a concentration-dependent pooled-mass floc population;
- prescribed floc diameters and densities;
- Mie scattering applied to each effective floc as if it were a homogeneous sphere;
- direct sampling of Mie intensity per degree rather than the three-dimensional solid-angle probability;
- a fixed-step Bernoulli transport approximation; and
- an explicit 7% probability of reversing every floc event by 180°.

The last two angular choices are especially important. CLARITAS_18 samples `I(theta)` without the required `sin(theta)` solid-angle Jacobian and then applies the sampled polar angle directly in the two-dimensional transport plane. It also adds π to 7% of floc-event directions. Those choices can readily shape a circumference-detector curve, but neither constitutes a predictive three-dimensional scattering model. The near-perfect Loess 0.5 g/L fit therefore cannot be treated as proof that the CLARITAS_18 physics was correct.

The current branch corrects important transport defects: it samples exponential extinction paths, weights object selection by `n_i sigma_t,i`, implements explicit absorption through single-scattering albedo, removes the reflection reversal, and records extensive transport diagnostics. These changes should be retained.

However, the current floc implementation is not yet a self-consistent physical model. It:

1. creates the floc population with an assumed collision-length formula rather than aggregation kinetics or measured floc data;
2. maps each source-size band over a prescribed lognormal floc-diameter kernel;
3. assigns each floc a geometric scattering cross-section `pi r^2`, independent of the computed aggregate structure;
4. labels every surviving floc encounter as a scattering event before determining whether any internal monomer scatter occurs;
5. uses a representative primary particle for all internal events in a floc;
6. applies floc absorption using a mean-chord outer albedo, but does not apply absorption along the actual internal random-walk path; and
7. still samples angular CDFs with `ANGULAR_CDF_POWER = 0`, i.e. without a three-dimensional solid-angle measure.

Thus the branch combines a geometric collision model, a hand-constructed aggregate population, Mie monomer phase functions, and a two-dimensional internal random walk without a single derivation that makes their cross-sections, probabilities, optical depths, and phase function mutually consistent.

### Recommendation

Choose **Option C: build a controlled reference architecture, using both versions as evidence but neither as the new foundation**.

- Preserve CLARITAS_18 as a frozen behavioural benchmark, not as a physics baseline.
- Preserve CLARITAS_76 as a diagnostic and feature donor, not as the production architecture.
- First build a minimal, independently testable three-dimensional transport reference with primary particles only.
- Validate absolute transmission, absorption, angular distributions, concentration scaling, and detector geometry separately.
- Add one floc representation only after its population, extinction cross-section, absorption, and phase function are defined consistently and constrained independently of the four detector curves.

If only A or B were allowed, B would be safer than A because it permits controlled ablation. Nevertheless, directly restarting from CLARITAS_18 risks canonising the very angular and reflection surrogates that produced the attractive fit.

## 2. Evidence and limitations

This review used:

- the two requested source files;
- `sediment_data.csv`;
- the existing parameter-screen ranking and per-dataset results;
- existing optical-budget and transport diagnostics.

The repository contains no stored CLARITAS_18 benchmark run paired with the measured curve under a common scoring script. Therefore the statement that CLARITAS_18 was “almost perfect” is accepted as project history, but its exact RMSE and Monte Carlo uncertainty could not be independently reproduced from the stored outputs. This matters because the current screen uses unit-sum-normalised curves and CLARITAS_18 did not write the same diagnostic set.

The review does not infer correctness merely from agreement. Four normalised curves are insufficient to identify the many population, optical, angular, and transport parameters currently active.

## 3. Major differences

| Component | CLARITAS_18 | CLARITAS_76 | Assessment |
|---|---|---|---|
| Bulk path sampling | Fixed `STEP_SIZE = 0.1/mu_s`; scatter probability is 1 each step | Exponential path `-log(U)/mu_t` | Current method is the correct homogeneous-medium event sampler |
| Event object | `n_i sigma_s,i` | `n_i sigma_t,i` then scatter/absorb by albedo | Current method is correct if each object's `sigma_t`, `sigma_s`, and `sigma_a` are correct |
| Absorption | None | Complex index, Mie absorption for primaries, mean-chord absorption for flocs | Primary implementation is a real improvement; floc implementation is inconsistent with internal path transport |
| Primary phase CDF | Cumulative raw `I(theta)` | Still raw `I(theta)` because `ANGULAR_CDF_POWER=0` | Both omit the 3-D `sin(theta)` measure |
| Direction update | Signed polar angle directly in a 2-D plane | Same for bulk and internal transport | Not a 3-D photon direction rotation |
| Floc population | One prescribed floc per eligible source band, prescribed density | Source-to-floc lognormal kernel, fractal mass/density | More expressive, but underconstrained and not derived from aggregation dynamics |
| Floc cross-section | Mie `Q_sca pi r^2` using an effective refractive index | `FLOC_SCATTER_EFFICIENCY pi r^2` | Current cross-section is geometric and disconnected from its structure calculation |
| Floc angular response | Homogeneous-sphere Mie plus 7% reversal | Representative-monomer internal random walk; synthetic structure mostly bypassed in true-domain mode | Removal of reversal is correct; replacement is not yet self-consistent |
| Detector response | Boundary exit-position bins; ballistic rear-bin suppression | Essentially the same rule, with more diagnostics | This is an ideal circumference counter, not yet a complete explicit detector optical model |
| Boundary optics | Indices passed but no Fresnel/refraction operation in the kernel | Indices removed from transport kernel | Neither version implements the stated sample-wall/interface optics |

## 4. Primary-particle physics

### 4.1 Cross-sections and refractive index

CLARITAS_18 uses a real primary index of 1.59 and calculates `qsca` using `miepython.efficiencies_mx`; it therefore has no material absorption. Its event rate is based only on `sigma_s`.

CLARITAS_76 uses a complex primary index `1.59 - i k`, calls the same Mie efficiency routine, constructs `sigma_t`, `sigma_s`, and `sigma_a = sigma_t - sigma_s`, and uses their derived single-scattering albedo. This is the preferable primary-particle formulation.

The sign convention is handled deliberately: the code stores `1.59 - i k`, then ensures the effective index retains a non-positive imaginary component. That is compatible with the convention expected by the current Mie calls, subject to verification against the installed `miepython` version.

### 4.2 Phase-function normalisation

Both versions evaluate the unpolarised intensity

`I(theta) = 0.5 (|S1|^2 + |S2|^2)`.

Neither converts this into a physical three-dimensional polar-angle probability. For an azimuthally symmetric phase function, the probability density in polar angle is proportional to:

`I(theta) sin(theta)`.

CLARITAS_18 explicitly cumulatively sums `I` and says not to use `sin(theta)`. CLARITAS_76 makes the measure configurable but sets `ANGULAR_CDF_POWER = 0`, which again gives a unit measure. This strongly overweights the endpoints per unit solid angle, particularly the very narrow forward lobe, and changes the relation between Mie efficiencies and sampled directions.

This is not a harmless detector convention. Cross-sections from Mie theory are three-dimensional integrals, while the transport samples a different angular measure. The rate and direction parts of the scattering kernel are therefore not representations of the same differential cross-section.

### 4.3 Angular transport

CLARITAS_18 calculates a proper azimuthal projection candidate but does not use it; it sets `theta_projected = sign * theta_3d`. CLARITAS_76 does the same. A predictive model should maintain a 3-D direction vector, sample azimuth uniformly, and rotate the direction in a local orthonormal basis. Projection to detector coordinates should happen only when a ray intersects a physical detector surface.

### 4.4 Why CLARITAS_18 could fit Loess 0.5 g/L

The most credible code-based explanation is cancellation:

- raw-per-degree Mie sampling supplies a particular strongly forward/end-point-weighted curve;
- effective homogeneous-sphere floc Mie functions supply large-size angular structure;
- 7% of every floc event is reversed by π;
- the detector counts boundary position, not ray incidence on a finite detector;
- ballistic contributions are explicitly suppressed for detector centres at and above 90°.

At 0.5 g/L, where rays experience relatively few events, the detector curve is highly sensitive to the single-event phase function and the explicit reversal probability. This gives the CLARITAS_18 surrogate enough leverage to match a normalised angular curve closely. At 4 g/L, repeated application compounds the surrogate kernel and exposes its wrong concentration dependence.

## 5. Monte Carlo transport

### 5.1 CLARITAS_18 fixed-step method

CLARITAS_18 sets `STEP_SIZE = 0.1/mu_s` but passes `scatter_prob_per_step = 1.0`. Consequently every completed step scatters. The effective collision spacing is deterministic `0.1/mu_s`, not exponentially distributed with mean `1/mu_s`.

This has two effects:

- the mean scattering rate is approximately ten times the `mu_s` used to derive the step;
- free-path variance is suppressed to zero.

The code's printed statement “scattering probability per physical step” does not repair this mismatch. A Bernoulli discretisation would require `p = 1-exp(-mu_s STEP_SIZE)`, approximately 0.0952 for this step, not 1.

This is the largest transport regression *in CLARITAS_18 itself* and a likely reason its apparent fit does not generalise with concentration.

### 5.2 CLARITAS_76 event-driven method

The current kernel correctly samples a bulk extinction free path from `mu_t`, checks whether the boundary is reached first, samples the responsible object from `n_i sigma_t,i`, and chooses absorption versus scattering from the object's albedo.

This should definitely be retained. It will not, however, produce correct results when the floc `sigma_t` and albedo are not consistent with the floc internal model.

### 5.3 Boundary handling

Both kernels intersect the initial ray with a circular sample and terminate when the next motion goes beyond the circle. Neither applies:

- Fresnel reflection/transmission;
- Snell refraction;
- sample-container wall geometry;
- detector aperture, orientation, or angular acceptance in ray direction;
- wavelength-dependent detector responsivity.

The `n_medium` and `n_external` arguments in CLARITAS_18 are unused inside its kernel. Their presence should not be interpreted as implemented boundary optics.

The detector rule is based on exit-position angle within ±6.5° of nominal detector centre. The exiting direction is recorded but not used to test whether a photon enters a detector. This can be an intentional idealised circumference detector, but it is not yet an “explicit detector geometry” in the optical sense.

## 6. PSD and concentration scaling

### 6.1 What is correct

For `PSD_WEIGHT_MODE = "mass_fraction"`, both versions compute:

`n_i = C w_i / m_i`.

They then compute event strengths from number density times cross-section. The dimensional concentration conversion is correct because 1 g/L equals 1 kg/m³.

### 6.2 What remains unverified

The source meaning of the Loess and Kaolin PSD weights is not established in code. Treating a laser-diffraction volume distribution as mass fraction may be reasonable at uniform mineral density; treating a number distribution that way would be severely wrong. This must be resolved from measurement metadata, not selected by detector fit.

### 6.3 Nonlinear floc population scaling

Both versions use:

`f_floc = L / (L + n_eligible^(-1/3))`.

Since `n_eligible` is proportional to concentration, this introduces an assumed `C^(1/3)` dependence into the pooled mass fraction. It is not a collision probability derived from Smoluchowski kinetics, residence time, shear rate, differential settling, sticking efficiency, or breakup.

CLARITAS_76 additionally increased the eligible primary diameter to 50 µm and uses 250 µm rather than CLARITAS_18's 50 µm collision length. Existing diagnostics show that for Loess:

- at 0.5 g/L, 46.47% of mass is floc and flocs generate 45.18% of extinction events;
- at 4.0 g/L, 55.01% of mass is floc and flocs generate 58.98% of extinction events.

Meanwhile total extinction rises from about 22.59 m⁻¹ to 163.88 m⁻¹, and valid exits fall from 78.43% to 27.14%. Thus concentration changes not only optical depth but the assumed scatterer population and its event mix. Any error in the floc model is amplified at 4 g/L.

### 6.4 Explanation of the recurring 0.5-versus-4.0 tradeoff

The tradeoff is expected when fitting a wrong one-event kernel and a wrong event-count scaling simultaneously:

- low concentration mainly constrains source distribution, detector geometry, and zero-/one-/few-event phase functions;
- high concentration additionally constrains extinction magnitude, path statistics, absorption, repeated angular convolution, and the concentration-dependent floc population.

A change that sharpens the one-event curve can improve 0.5 g/L while repeated application makes 4.0 g/L too forward, too diffuse, or too selectively absorbed. Conversely, broadening or suppressing long paths to repair 4.0 g/L damages the low-concentration shape.

The current results support this diagnosis. Baseline Loess RMSE is 0.0246 at 0.5 g/L but 0.0871 at 4.0 g/L. Raising `k` to 0.002 slightly worsens 0.5 g/L to 0.0255 but improves 4.0 g/L to 0.0718. The dominant issue is therefore concentration/path-class dependent, not a global angular offset.

## 7. Critical assessment of the floc architecture

### 7.1 Population and pooling

Mass conservation is implemented carefully: pooled and residual mass fractions sum to the original source-band mass. This is worth retaining.

The rest of the mapping is a hypothesis, not yet a physical prediction. Each source band is distributed over specified floc diameters with a lognormal kernel of fixed width. The preferred diameter is determined by interpolation across array positions. This means the source-to-floc mapping depends partly on how many bins happen to be listed and their ordering, not solely on physical state variables.

### 7.2 Representative-primary approximation

For each effective floc cell, the representative primary is the closest PSD bin to the geometric midpoint of the source band's min/max diameters. Physical monomer count is then `floc_mass / representative_monomer_mass`.

This loses:

- the actual within-band primary number and mass distribution;
- composition and refractive-index variation;
- polydisperse pair correlations;
- preferential aggregation by size;
- the distinction between mass-equivalent and optically representative monomers.

It may be acceptable as a deliberately tested reduced-order approximation, but it cannot be presumed accurate.

### 7.3 Synthetic fractal geometry

Points are independently drawn with radius `R u^(1/Df)` and isotropic direction, then recentered. This creates the intended radial mass scaling in expectation but does not construct a connected, non-overlapping aggregate. It permits monomer overlap, does not enforce contacts, and does not guarantee the requested outer radius after recentering.

The synthetic count is clipped to 8–384 even when the physical count differs. Pair-distance structure calculations therefore represent the clipped synthetic cloud, while internal number density uses the physical count. These are two different aggregates.

Moreover, with true CUDA internal-domain transport enabled and synthetic outer phase disabled, the bulk kernel uses representative-monomer Mie CDFs for internal scatters. Much of the sophisticated structure-factor path is then diagnostic rather than causally active in the production transport.

### 7.4 Cross-section/event inconsistency

The current floc cross-section is:

`sigma_s,floc = FLOC_SCATTER_EFFICIENCY pi r_floc^2`.

With efficiency 1, every geometric floc encounter is declared a scattering cross-section. After the extinction event survives absorption, the kernel increments `scatter_count` before running the internal domain. Yet an optically thin floc can exit without any internal monomer scatter and without a direction change. The outer event is still counted and was sampled from a cross-section that labelled it scattering.

A consistent finite-domain formulation would use a geometric encounter rate and then allow no-interaction traversal, scattering, or absorption according to internal extinction along the chord. Alternatively, it would use an aggregate-level `sigma_s`, `sigma_a`, and phase function and treat the floc as a point event. The current code mixes both formulations.

### 7.5 Absorption inconsistency

Floc absorption is calculated before transport using a mean chord and becomes an outer single-scattering albedo. The actual internal walk can have a much longer path and is recorded, but absorption is not sampled along that path. Increasing `k` therefore changes whether the ray enters the internal scattering branch at all; it does not implement Beer–Lambert loss along the simulated internal trajectory.

This is strong evidence that the fitted `k` can compensate for excessive or incorrectly distributed internal scattering.

### 7.6 Dimensionality

The synthetic aggregate is 3-D, its structure factor is based on 3-D separations, but the CUDA internal walk is in a 2-D circular domain with signed 2-D deflections. Its mean chord is set to `2D/3`, the mean chord of a 3-D sphere, not the corresponding 2-D disk chord under the actual entry distribution. This mixes dimensional models.

### Verdict on floc architecture

The architecture contains useful components and diagnostics, but it is not sound as an integrated predictive model. It should not be tuned further until a single consistent choice is made between:

1. aggregate-level optical objects with independently calculated `sigma_s`, `sigma_a`, and phase functions; or
2. explicit geometric aggregate encounters with monomer-resolved 3-D transport, including no-interaction traversal and pathwise absorption.

The first is much more computationally practical and should be the next reference implementation.

## 8. Parameter assessment

| Parameter | Physical status | Current role and risk |
|---|---|---|
| `FLOC_COLLISION_LENGTH_M` | Not a directly standard floc property in the implemented formula | A lumped surrogate for mixing, time, collision kernel, sticking, and breakup. It controls concentration-dependent pooled mass and is compensatory unless independently calibrated from hydrodynamics/floc-size measurements. |
| `FLOC_FRACTAL_DIMENSION` | Genuine measurable aggregate property | Physically meaningful, but a single universal value for all sizes, materials, and concentrations is a strong assumption. In the current model it changes mass, density, number density, optical depth, and synthetic geometry simultaneously. |
| `PRIMARY_REFRACTIVE_INDEX_IMAG_K` | Genuine material optical constant | Must be wavelength- and material-specific and independently measured/literature-constrained. Its current fitted improvement is suspicious because the same `k` is screened across Loess and Kaolin and floc absorption is not pathwise. |
| `FLOC_SCATTER_EFFICIENCY` | Could correspond to `Q_sca` only if derived | At a fixed value of 1 it is an assumption. If fitted, it would be precisely a forbidden scatter multiplier. It is disconnected from the synthetic structure and internal optical depth. |
| `FLOC_POOL_KERNEL_LOG_SIGMA` | Could represent measured floc-size dispersion | Currently an assumed mapping width. Without independent floc PSD data or a population-balance derivation, it is a shape/tuning parameter that redistributes mass and event probability. |

### Is `k = 0.002` compensating?

Probably, although the screen alone cannot prove it.

Evidence:

- Doubling `k` improves Loess 4.0 g/L strongly but slightly worsens Loess 0.5 g/L.
- It also worsens Kaolin 4.0 g/L substantially: RMSE changes from 0.0519 to 0.0655.
- At Loess 4.0 g/L the valid-exit fraction falls from 27.14% at baseline to 20.97% at `k=0.002`, while mean scatter count among valid exits falls from 3.82 to 1.24.
- All scored curves are normalised to unit sum, so absolute attenuation information—the principal constraint on absorption—is discarded.
- Floc absorption is applied using a mean chord before an internal walk rather than along its actual path.

The optimisation is therefore selecting a path-class filter: higher absorption preferentially removes long/multiply scattered histories and reshapes the *normalised survivor distribution*. That is not evidence that the mineral `k` is correct.

## 9. Improvements to retain

1. Event-driven exponential bulk free paths.
2. Extinction-event weighting by `n_i sigma_t,i`.
3. Explicit per-object `sigma_s`, `sigma_a`, `sigma_t`, and single-scattering albedo.
4. Complex-index Mie calculations for primary particles.
5. Mass-conserving PSD transformation and explicit per-bin particle masses.
6. Removal of the 7% floc reversal and other detector-specific angular corrections.
7. Separate diagnostics for event probability, optical budget, scatter class, last-scatter bin, exit position/direction, and absorption.
8. Deterministic cache keys and deterministic aggregate seeds for reproducibility.
9. CPU/GPU phase sampling validation infrastructure.

These should be ported into a tested modular implementation, not retained merely by continuing to edit the current monolithic script.

## 10. Regressions or unresolved defects since CLARITAS_18

“Regression” here means loss of correctness or internal consistency, not simply loss of curve fit.

1. The prescribed CLARITAS_18 floc Mie object was replaced by a multi-layer model whose event cross-section, internal transport, structure factor, and absorption do not describe the same optical object.
2. The floc population expanded from 9 prescribed bins to 155 source/floc cells without independent constraints, increasing non-identifiability.
3. Eligibility extended to 50 µm and collision length increased fivefold, substantially changing population and concentration response at the same time as transport changed.
4. Geometric floc `sigma_s` replaced computed Mie scattering efficiency without a validated aggregate optical calculation.
5. Actual internal path length was introduced but not coupled to absorption.
6. A 3-D aggregate description was coupled to 2-D direction transport.
7. The angular-measure defect was retained, despite removal of other empirical angular choices.
8. The detector remains based on boundary position only, so added exit-direction modelling does not yet make detector accumulation more physical.

## 11. Prioritised investigations

### Priority 0 — freeze evidence and make comparisons valid

1. Run CLARITAS_18 and CLARITAS_76 through one immutable scoring pipeline with fixed random seeds and at least 10 independent replicates.
2. Report Monte Carlo confidence intervals, not one-run RMSE.
3. Score both absolute detector power/transmission and normalised angular shape. Do not let unit-sum normalisation erase absorption constraints.
4. Record all source, sample, wall, detector, wavelength, concentration, and PSD metadata.

### Priority 1 — validate the primary-only transport kernel

5. Disable flocs and absorption; compare Beer–Lambert uncollided transmission to `exp(-mu_s L)` in slab and circular geometries.
6. Compare sampled free-path and event-bin histograms to their analytic exponential and categorical distributions.
7. Replace planar signed-angle updates with 3-D vector rotations and use `I(theta) sin(theta)`; verify sampled moments, including Mie `g`.
8. Validate single-scatter detector response against numerical integration of the differential Mie cross-section over the actual detector aperture.
9. Add Fresnel/refraction/container optics only after the no-interface reference passes.

### Priority 2 — establish concentration and measurement semantics

10. Determine whether each supplied PSD is number-, area-, volume-, or mass-weighted from instrument metadata.
11. Verify actual solids concentration, optical path, sediment density, dispersion protocol, settling time, and detector exposure normalisation.
12. Use dilution-series absolute transmission to test whether `mu_t` scales linearly before introducing concentration-dependent aggregation.

### Priority 3 — constrain absorption independently

13. Obtain wavelength-dependent complex refractive indices separately for Loess and Kaolin, or measure absorption on a geometry that separates it from scattering.
14. Confirm the `miepython` complex-index convention and reproduce benchmark efficiencies from a trusted Mie reference.
15. Do not screen `k` against unit-normalised detector shapes.

### Priority 4 — introduce one consistent floc model

16. Obtain measured floc-size distributions versus material, concentration, shear/mixing, and elapsed time.
17. Start with aggregate-level tables from a validated method (e.g. RDG where valid, multi-sphere T-matrix/DDA for selected representatives, or measured phase functions), providing mutually consistent `sigma_s`, `sigma_a`, and phase functions.
18. Treat fractal dimension, prefactor, monomer-size distribution, and floc PSD as measured inputs with uncertainty.
19. Validate the aggregate table independently before enabling multiple scattering in the bulk.
20. Only consider explicit internal transport if aggregate-level optics demonstrably fails and a 3-D, pathwise-absorbing implementation can be afforded.

### Priority 5 — controlled reintroduction

21. Perform an ablation matrix with one change per row: free paths, event weighting, absorption, 3-D angle sampling, detector optics, floc population, floc optics.
22. Require each addition to pass analytic/unit tests and improve predictive performance across held-out concentrations/materials without retuning previously fixed physical inputs.

## 12. Development strategy

### Why not A

Continuing directly from CLARITAS_76 would preserve extensive diagnostics, but the physics is too entangled for parameter screening to identify the faulty layer. More tuning will continue to exploit compensating errors.

### Why not B as written

CLARITAS_18 is small enough for controlled reintroduction, but it contains a deterministic over-scattering transport, non-solid-angle phase sampling, planar propagation, a 7% floc reversal, and idealised detector filtering. Restarting from it risks treating its attractive Loess curve as a scientific acceptance test and rejecting correct changes because they remove error cancellation.

### Recommended Option C

Create a new, small reference implementation with:

- primary particles only;
- verified PSD semantics;
- complex-index Mie cross-sections;
- physical 3-D phase sampling;
- event-driven extinction;
- explicit, testable geometry;
- absolute energy accounting;
- deterministic/reproducible runs.

Use CLARITAS_18 as a frozen behavioural oracle to explain the historical fit. Use CLARITAS_76 as a donor for its event sampler, diagnostics, and data plumbing. Introduce aggregate-level floc optics only after the reference passes analytic tests and independent floc measurements exist.

This strategy preserves the real advances, avoids rebuilding on known CLARITAS_18 defects, and restores scientific identifiability. It has the highest probability of yielding a genuinely predictive model.

## 13. Bottom line

The current branch is worse at fitting the benchmark primarily because it removed some powerful error-cancelling behaviours while adding a floc model whose components are not mutually consistent. The increased absorption selected by the screen acts mainly as a survivor/path-class filter and should not be accepted as a material property.

The correct response is not to restore reflection or tune angular weighting. It is to separate and validate:

1. primary optical properties;
2. three-dimensional single-event scattering;
3. bulk event transport and concentration scaling;
4. detector/interface optics;
5. floc population;
6. aggregate optical cross-sections and phase functions.

Only after those layers are independently correct should the four measured curves be used as a joint predictive test.
