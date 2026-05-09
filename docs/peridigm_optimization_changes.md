# Changes to the Peridigm Fracture Simulation & Optimization Integration

## 1. Damage Evaluation Timing — 2nd Wave Reflection

**Previous**: Damage was read at the final simulation timestep (150 µs), which includes artificial damage compounding from multiple wave reflections bouncing within the shell.

**New**: Damage is evaluated at the **2nd wave reflection time**, computed from assembly geometry:

t_2nd = 2L / v_wave

where L is the pole-to-pole assembly length (read from the discretization file, ~69.5 mm) and v_wave = 1420 m/s is the PLA longitudinal wave speed. This captures one full compression cycle (wave travels from cone-side pole to wall-side pole and returns) before later reflections compound damage artificially. For the current geometry, t_2nd ≈ 98 µs.

## 2. Per-Band, Per-Hemisphere Damage Extraction

**Previous**: Global scalar metrics only — mean damage in fracture zone (D_frac) and mean damage in shell body (D_shell), averaged over the entire shell.

**New**: Damage is computed per theta band (10 bands, edges at [0, 9, 18, ..., 90]° from pole) and per hemisphere:
- **Lower hemisphere** (cone-side, z < z_mid): where the impactor enters
- **Upper hemisphere** (wall-side, z > z_mid): constrained by fishing line

For each band b, four quantities are computed:
- D_frac_lower(b), D_frac_upper(b) — mean fracture zone damage
- D_prot_lower(b), D_prot_upper(b) — mean protected shell damage

The **worst-case** (maximum of the two hemispheres) drives the optimization:
- D_frac_worst(b) = max(D_frac_lower, D_frac_upper)
- D_prot_worst(b) = max(D_prot_lower, D_prot_upper)

A **damage penetration depth** metric is also computed — the highest theta band index where D_frac_worst > 0.1. This indicates how far from the pole the fracture zone is being engaged by the impactor.

## 3. Cone Angle as an Adaptive Optimization Parameter

**Previous**: Cone half-angle was hardcoded at 30° throughout the pipeline.

**New**: The cone half-angle is an adaptive parameter that adjusts each iteration based on damage penetration depth:
- **Too shallow** (penetration depth < band 3, i.e. < 27°): damage is concentrated near the pole → widen cone by +5° to distribute load further
- **Too deep** (penetration depth > band 6, i.e. > 54°): damage reaches the equator → narrow cone by -5° to focus load
- **Sweet spot** (bands 3–6): no change

Bounded to [15°, 60°]. Initial value: 30°. The updated cone angle is stored in the optimization state and read by the SLURM orchestrator before each Peridigm submission.

## 4. Per-Band Damage-Driven Fracture Factor Updates

**Previous**: Binary breach-based logic:
- Breach detected → thicken all fracture zone bands uniformly by +0.03
- Blocked at a specific band → thin that band and neighbors by -0.06/-0.03

**New**: Continuous per-band damage-driven adjustments using the worst-case hemisphere damage. For each band b:

| Condition | Action |
|-----------|--------|
| D_prot_worst(b) > 0.05 | Protected shell damaged → thicken fracture factor by +0.05 |
| D_frac_worst(b) > 0.3 (target) | Breaking well → thin by up to -0.04, scaled by excess |
| D_frac_worst(b) < 0.15 (half target) | Not breaking enough → thicken by up to +0.03, scaled by gap |
| Between 0.15 and 0.3 | Acceptable range → no change |

The previous breach-based logic is retained as a fallback when per-band damage data is unavailable (e.g., first iteration).

## 5. Per-Band Fracture-Aware Thickness Cap

**Previous**: Global cap — if overall D_frac < 0.3, block ALL thickness increases (prevents runaway thickening when fracture zone isn't breaking).

**New**: Per-band cap — only block thickness increases on bands where that specific band's fracture zone isn't breaking (D_frac_worst(b) < 0.3). Bands with adequate fracture damage are free to thicken if FEA stress demands it.

## 6. Configuration Parameters Added

Under peridigm:
- pla_wave_speed_mps: 1420.0 — PLA longitudinal wave speed for 2nd reflection timing
- cone_angle_adaptation:
  - enabled: true
  - min_angle_deg: 15
  - max_angle_deg: 60
  - step_deg: 5
  - target_depth_min: 3 (band 3, ~27-36°)
  - target_depth_max: 6 (band 6, ~54-63°)
  - damage_threshold: 0.1

Under optimization:
- use_per_band_damage: true — enables hemisphere-aware damage feedback
- target_band_fracture_damage: 0.3 — per-band fracture zone damage target
- max_band_protected_damage: 0.05 — per-band protected shell damage threshold

## 7. Existing Peridigm Simulation Parameters (Unchanged)

For reference, the following parameters remain from the existing configuration:
- Flyback split-core simulation with steel halves separated by 0.5mm gap
- Impact velocity: 200 m/s
- Simulation time: 150 µs (damage plateaus by ~100 µs)
- Horizon: 2.0 mm
- PLA critical stretch (shell body): 0.024 (from Gc=6000 J/m², K=2.98 GPa)
- Fracture zone critical stretch: 0.015 (intentionally weaker than shell)
- Cone half-angle: 30° initial (now adaptive)
- Cone height: 10 mm, sharp tip
- 8 MPI tasks per Peridigm run
- Block IDs: 1,4 = steel core, 2 = protected shell, 3 = fracture zone, 99 = cone
