## Resumen ejecutivo no normativo

This repository is governed by BASURIN_README_SUPER.md (single source of truth).
If an executive contract fails, the run does not exist. No downstream.


## Resumen ejecutivo (contract-first)
- Principio: “Si falla un contrato ejecutivo, el run no existe; abort; no hay resultados interesantes de runs inválidos.”
- IO determinista bajo `runs/<run_id>/`, prohibido escribir fuera, y estructura mínima de stage:
  - `runs/<run_id>/<stage_name>/{manifest.json, stage_summary.json, outputs/}`
- Contrato soberano: `RUN_VALID` debe ser `PASS` antes de downstream; si falla, abort.
- Stages canónicos (v1):
  - `geometry`
  - `spectrum` (produce `spectrum.h5`)
  - `dictionary` (no gobierna existencia)
  - `ringdown_synth` (único generador autorizado de sintéticos ringdown; índice canónico `synthetic_events.json`)
  - `RUN_INDEX` (inventario/auditoría; gated por `RUN_VALID==PASS`)
- Regla de experimentos:
  - viven en `runs/<run_id>/experiment/<name>/`
  - no generan sintéticos nuevos
  - no reconstruyen rutas “a mano”; consumen índices canónicos (p.ej. `synthetic_events.json`)
  - abort si `RUN_VALID != PASS`

Referencia normativa: lee y sigue BASURIN_README_SUPER.md; este README no sustituye contratos.

**Audience:** automated agents (AI), collaborators, auditors.
**Normative authority:** `BASURIN_README_SUPER.md` (single source of truth for contracts and IO).

> If this document contradicts `BASURIN_README_SUPER.md`, the sovereign wins and this text is invalid.

---

## 0. Scientific Objective (one sentence)

**Demonstrate — or refute — that a canonical, contract-first, auditable pipeline can extract real geometric information from ringdown observables (f, τ, Q) and map it to stable decisions (geometry ranking by spectral ratios), even under controlled perturbations of the data and of time.**

---

## 1. What BASURIN Is

BASURIN is an epistemological testbed, not a physics simulator. It exists to answer:

1. **Can dimensionless spectral ratios derived from (f, τ) carry geometric information more stably than absolute values?** This is the "Phi bridge" conjecture (`TESIS_PUENTE_PHI.md`). BASURIN either confirms it survives noise, temporal warps, PSD variations, and real data — or localizes exactly where it breaks (estimator, featuremap, selector).

2. **Does the ranking change because the physics changes, or because the pipeline is fragile?** Experiments like `EXP_RINGDOWN_TIME_00` (reparametrization invariance) attack this directly: apply controlled temporal transformations and check if the geometric conclusion is stable.

3. **Where is the actual information limit?** With single-mode ringdown (2 independent observables: f and Q), the discriminable atlas size is ~16 geometries at 5% noise. BASURIN documents this as a scientific result, not a bug, and defines what additional data (overtones, amplitude ratios with excitation contracts) would be needed to reach N=128.

---

## 2. What BASURIN Is Not

- ❌ Not a QNM fitting exercise.
- ❌ Not a machine learning model or metric optimizer.
- ❌ Not an attempt to prove or refute GR.
- ❌ Not plot generation or ad-hoc scripting.

BASURIN does not pursue direct physical truth. It pursues **epistemological robustness of the procedure**: every result is reproducible, every decision auditable, every failure localized. A negative result ("this invariance breaks here") is informative, not waste.

---

## 3. Core Pipeline (current state)

```
geometry → spectrum → dictionary/atlas
                                  ↓
ringdown_synth → estimator → featuremap (Φ⁻¹) → geometry_select → verdict
                                  ↑                     ↑
                           (f,τ,Q) → ratios        atlas ranking
```

### Key stages

| Stage | Role | Key artifact |
|---|---|---|
| `geometry` | Produce AdS domain-wall geometry | `geometry.h5` |
| `spectrum` | Solve Sturm-Liouville → eigenvalues | `spectrum.h5` |
| `dictionary` | Build atlas of (Δ, M²₀, ratios) | `atlas.json` |
| `ringdown_synth` | Generate synthetic ringdown events | `synthetic_events.json` |
| `ringdown_featuremap_v0` | Map (f, τ) → holographic ratio space (the Phi bridge) | `mapped_features.json` |
| `geometry_select_v0` | Rank atlas by proximity, evaluate accuracy | `geometry_ranking.json` |

### The Phi bridge (conjectured, falsifiable)

Forward (atlas → predicted ringdown):
```
ω₀ = √|M²₀| / L,  f = ω₀/(2π),  γ = α·ω₀·(r₁−1),  τ = 1/γ
```

Inverse (observed ringdown → ratio space):
```
Q = π·f·τ,  r₁ = 1 + 1/(2·α·Q)
```

Falsification thresholds (N=128, 5% noise): `accuracy_top1 ≥ 70%`, `accuracy_top3 ≥ 95%`.
Current status: **FAIL at N=128 with single-mode** (insufficient observables). PASS at N≤16. This is an information limit, documented in `test_end_to_end_geometry_recovery.py`.

---

## 4. What Counts as Success

BASURIN produces **verdicts**, not just data:

- "This invariance holds up to this noise level and atlas size."
- "This estimator is insufficient under these temporal transformations."
- "This ranking is not stable on real data v0."

A negative verdict that localizes the failure is a successful experiment. The outputs are falsifiable theses with explicit thresholds, hash-based traceability, and reproducible gates.

---

## 5. Governance (summary — normative details in `BASURIN_README_SUPER.md`)

**Non-negotiable rule:** If `RUN_VALID` fails, the run does not exist. Zero downstream.

- All IO under `runs/<run_id>/`. No writes outside.
- Every stage produces: `manifest.json` + `stage_summary.json` + `outputs/`.
- Experiments live under `runs/<run_id>/experiment/<name>/`, must not generate synthetic data, must not mutate canonical artifacts.
- Executive pre-checks in order: RUN_VALID → artifacts exist → semantic capabilities verified → only then compute.

---

## 6. Active Experiments and Next Steps

| Experiment | Purpose | Status |
|---|---|---|
| `EXP_RINGDOWN_TIME_00` | Reparametrization invariance: do temporal warps change geometric conclusions? | Design phase |
| Phi bridge validation (synthetic) | End-to-end geometry recovery at varying N and noise | Tests exist, N≤16 passes |
| Overtone extension | Add (f₂₂₁, τ₂₂₁) to break spectral degeneracy at N=128 | Planned |
| Alpha bootstrap calibration | Calibrate α on synthetic atlas, test convergence | Planned |

---

## 7. Key Insight for AI Agents

When working on BASURIN:

1. **Read `BASURIN_README_SUPER.md` first** for all contract and IO questions.
2. **The goal is not to make tests pass** — it's to determine *whether* they can pass and *why or why not*.
3. **A result of "this doesn't work because X" is valuable.** Don't optimize around failures; document them.
4. **Amplitude ratios require excitation contracts.** Amplitudes depend on initial conditions (the "kick"), not just geometry. Never use them to break degeneracy without explicit separation of geometric vs. initial-state information.
5. **No ad-hoc scripts.** Every computation must flow through a stage with manifest, summary, and hash traceability.

