# Informe teórico: Viabilidad multimodo y consistencia Kerr para BASURIN

**Versión:** 3.0  
**Fecha:** 2026-03-03  
**Contexto:** BASURIN — Clasificación de viabilidad multimodo (220+221), determinismo puro  
**Cambios respecto a v2:** Se integra canal formal de evidencia para nueva física (`science_evidence`), gate automático de sistemáticas (`systematics_gate`), hipótesis H1 mínima determinista ($\Delta R_f$), y carril de veto humano hasheado. Se mantiene determinismo puro end-to-end.

---

## 1. Resumen para decisiones

1. **Dos responsabilidades separadas, nunca mezcladas.** El *gate de viabilidad* decide si hay información (`MULTIMODE_OK` / `SINGLEMODE_ONLY` / `RINGDOWN_NONINFORMATIVE`). El *canal de evidencia* cuantifica desviación respecto de Kerr — solo si la viabilidad y las sistemáticas lo permiten.

2. **Los ratios $R_f(\chi)$ y $R_Q(\chi)$ son oráculo determinista libre de masa**, calculables desde `qnm`. Constituyen tanto el test de consistencia (gate) como la base de la hipótesis alternativa mínima (evidencia).

3. **H1 mínima es determinista hoy**: $\Delta R_f = R_f^{\text{obs, median}} - R_f^{\text{Kerr}}(\chi^*)$, donde $\chi^*$ es el spin Kerr más cercano al ratio observado. No requiere sampler, priors, ni evidencia bayesiana. Es aritmética sobre cuantiles ya disponibles.

4. **El bloque `science_evidence` existe siempre**, aunque sea `NOT_EVALUATED`. El contrato reserva el hueco para que el pipeline no sea estructuralmente ciego a nueva física. Slots futuros (`delta_f_221`, `delta_tau_221`, `log_bayes_factor`) arrancan como `null`.

5. **Gate automático de sistemáticas** (`systematics_gate`): determinista, con checks tipo `t0_plateau_verdict == STABLE AND chi_psd < threshold`. Se evalúa solo si `class == MULTIMODE_OK`.

6. **Override humano solo como veto degradante**: si existe `annotations/systematics_override.json` (hasheado), solo puede forzar `NOT_EVALUATED`, nunca promover a `EVALUATED`. Esto preserva determinismo y trazabilidad.

7. **El `delta_bic` de `model_comparison.json` entra como flag de severidad**, no como gate único ni como evidencia bayesiana. Es determinista.

8. **Todos los umbrales son parámetros explícitos** registrados en los artefactos. Convenciones operativas, no constantes físicas.

9. **"Inconsistente con Kerr" activa el canal de evidencia, no un claim**. El pipeline produce el objeto de evidencia ($\Delta R_f$, intervalo, diagnósticos); la interpretación es humana.

10. **10 tests de regresión** cubren: clasificación (6), invariantes físicos (4 de Kerr), y canal de evidencia (4 de separación/no-claim/existencia/override).

---

## 2. Modelo mínimo de ringdown y parámetros observables

### 2.1 Señal como suma de modos

$$h(t) = \sum_{\ell m n} A_{\ell m n}\, e^{-t/\tau_{\ell m n}}\, \cos(2\pi f_{\ell m n}\, t + \phi_{\ell m n}), \qquad t \geq t_0 \tag{2.1}$$

Modos relevantes: fundamental $(2,2,0)$ y primer overtone $(2,2,1)$.

### 2.2 Parámetros observables por modo

$$f_{\ell m n}, \quad Q_{\ell m n} \equiv \pi f_{\ell m n} \tau_{\ell m n} \tag{2.2}$$

### 2.3 Separación masa vs. spin

$$f_{\ell m n} = \frac{c^3}{2\pi G M}\, \omega_{\ell m n}(\chi), \qquad Q_{\ell m n} = Q_{\ell m n}(\chi) \tag{2.3}$$

El ratio $f_{221}/f_{220} = \omega_{221}(\chi)/\omega_{220}(\chi)$ depende solo de $\chi$. La masa se cancela exactamente.

---

## 3. Ratios Kerr como oráculo determinista

### 3.1 Definición

$$R_f(\chi) \equiv \frac{f_{221}(\chi)}{f_{220}(\chi)}, \qquad R_Q(\chi) \equiv \frac{Q_{221}(\chi)}{Q_{220}(\chi)} \tag{3.1}$$

Funciones deterministas de $\chi \in [0, 0.998]$, tabulables desde `qnm` (Leaver continued-fraction).

### 3.2 Comportamiento (según fits estándar publicados)

$R_f$: aproximadamente creciente con $\chi$, rango $\approx [0.88, 1.00]$.  
$R_Q$: siempre $< 1$ (overtone decae más rápido), rango $\approx [0.25, 0.85]$.

Valores exactos deben generarse ejecutando `qnm`. Los rangos aquí son orientativos.

### 3.3 Banda Kerr permitida

$$[R_f^{\min}, R_f^{\max}] = [\min_\chi R_f(\chi),\; \max_\chi R_f(\chi)] \tag{3.2}$$

---

## 4. Espacio de modelos: H0 vs H1

### 4.1 Hipótesis nula H0 (Kerr/GR)

Todos los QNMs del remanente están determinados unívocamente por $(M, \chi)$. El ratio observado satisface:

$$R_f^{\text{obs}} = R_f^{\text{Kerr}}(\chi) \quad \text{para algún } \chi \in [0, 0.998] \tag{4.1}$$

### 4.2 Hipótesis alternativa H1 (espectro deformado)

Se permite una desviación del ratio observado respecto de la predicción Kerr más cercana:

$$R_f^{\text{obs}} = R_f^{\text{Kerr}}(\chi^*) + \Delta R_f \tag{4.2}$$

donde $\chi^*$ es el spin Kerr que minimiza $|R_f^{\text{obs, median}} - R_f^{\text{Kerr}}(\chi)|$, y $\Delta R_f$ es el parámetro de desviación efectivo.

**$\Delta R_f$ es determinista hoy**: se calcula como diferencia aritmética entre la mediana del bootstrap y el valor Kerr más cercano en el grid. No requiere sampler, priors, ni integración de evidencia.

### 4.3 Slots futuros (no implementados, contrato reservado)

Para cuando exista un sampler:

$$f_{221} = f_{221}^{\text{Kerr}}(M, \chi) \cdot (1 + \delta f_{221}) \tag{4.3}$$

$$\tau_{221} = \tau_{221}^{\text{Kerr}}(M, \chi) \cdot (1 + \delta\tau_{221}) \tag{4.4}$$

Estos slots arrancan como `null` en el artefacto. El contrato existe; la implementación no.

### 4.4 Principio de diseño

H1 no obliga a "creer" en nueva física. Define el canal formal para cuantificarla **si** los datos lo piden **y** las sistemáticas están controladas. Sin este canal, el pipeline es estructuralmente incapaz de producir evidencia para modelos alternativos, independientemente de la calidad de los datos.

---

## 5. Test de consistencia Kerr (gate de viabilidad)

### 5.1 Intervalo observado (muestra a muestra del bootstrap)

Para cada muestra $k$ del bootstrap de s3b:

$$R_f^{(k)} = \frac{f_{221}^{(k)}}{f_{220}^{(k)}} \tag{5.1}$$

Del ensemble se extraen cuantiles:

$$R_f^{\text{obs}} \in [R_f^{q_{05}}, R_f^{q_{95}}] \tag{5.2}$$

### 5.2 Regla determinista de consistencia

$$\text{consistent} = (R_f^{q_{05}} \leq R_f^{\max}) \;\text{AND}\; (R_f^{q_{95}} \geq R_f^{\min}) \tag{5.3}$$

### 5.3 Niveles de consistencia

$$\text{overlap} = \max\left(0,\; \min(R_f^{q_{95}}, R_f^{\max}) - \max(R_f^{q_{05}}, R_f^{\min})\right) \tag{5.4}$$

$$\text{overlap\_frac\_obs} = \frac{\text{overlap}}{R_f^{q_{95}} - R_f^{q_{05}}} \tag{5.5}$$

| Nivel | Condición | Interpretación |
|-------|-----------|----------------|
| **Consistente fuerte** | `overlap_frac_obs > 0.5` | Más de la mitad del posterior en la banda Kerr |
| **Consistente débil** | `0 < overlap_frac_obs ≤ 0.5` | Contacto marginal |
| **Inconsistente** | `overlap_frac_obs == 0` | Ningún solapamiento |

---

## 6. Métrica de informatividad del 221

### 6.1 Informatividad espectral

$$\mathcal{I}_{R_f} \equiv 1 - \min\left(1,\; \frac{\Delta R_f^{\text{obs}}}{\Delta R_f^{\text{Kerr}}}\right) \tag{6.1}$$

$\mathcal{I}_{R_f} \to 1$: alta restricción del spin. $\mathcal{I}_{R_f} \to 0$: el 221 no discrimina.

### 6.2 Estabilidad del posterior del 221

$$\text{stability}_{221} = \frac{\text{IQR}(f_{221})}{\text{median}(f_{221})} \tag{6.2}$$

### 6.3 Integración con `delta_bic`

$\Delta\text{BIC} = \text{BIC}_{1\text{mode}} - \text{BIC}_{2\text{modes}}$ de `model_comparison.json`. Positivo → datos prefieren 2 modos. Entra como flag de severidad, no como gate único.

---

## 7. Reglas de clasificación: gate de viabilidad

### 7.1 Inputs (todos producidos por s3b)

```
INPUTS:
  valid_fraction_220      : float
  valid_fraction_221      : float
  f_220_median            : float   # Hz
  f_220_iqr               : float   # Hz
  f_221_median            : float | None
  f_221_iqr               : float | None
  Q_220_median            : float
  Q_220_iqr               : float
  Q_221_median            : float | None
  Q_221_iqr               : float | None
  Rf_bootstrap_quantiles  : {q05: float, q50: float, q95: float} | None
  Rf_kerr_band            : [float, float]   # de kerr_ratio_reference.json
  spin_at_floor_frac_221  : float
  delta_bic               : float | None      # de model_comparison.json
  two_mode_preferred      : bool | None
```

### 7.2 Pseudocódigo

```python
def classify_multimode_viability(inputs: dict) -> dict:
    """
    Función pura. Sin IO, sin estado, sin aleatoriedad.
    Responsabilidad ÚNICA: decidir si hay información multimodo.
    NO emite veredicto sobre nueva física.
    """
    reasons = []
    metrics = {}

    # ── Umbrales ──
    T = {
        "MIN_VALID_FRAC_220":    0.50,
        "MIN_VALID_FRAC_221":    0.30,
        "MAX_REL_IQR_F220":     0.50,
        "MAX_REL_IQR_F221":     0.60,
        "MAX_SPIN_FLOOR_FRAC":  0.30,
        "INFORMATIVE_THRESHOLD": 0.30,
        "DELTA_BIC_SUPPORTIVE":  2.0,
        "STABILITY_221_MAX":     0.60,
        "SEVERE_COUNT_LIMIT":    2,
    }

    # ════════════════════════════════════════════
    # GATE 0: ¿El 220 es fiable?
    # ════════════════════════════════════════════

    vf_220 = inputs["valid_fraction_220"]
    metrics["valid_fraction_220"] = vf_220

    if vf_220 < T["MIN_VALID_FRAC_220"]:
        reasons.append(
            f"valid_fraction_220={vf_220:.3f} < {T['MIN_VALID_FRAC_220']}: "
            "fundamental mode posterior unreliable"
        )
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, T)

    rel_iqr_f220 = inputs["f_220_iqr"] / inputs["f_220_median"]
    metrics["rel_iqr_f220"] = rel_iqr_f220

    if rel_iqr_f220 > T["MAX_REL_IQR_F220"]:
        reasons.append(
            f"rel_iqr_f220={rel_iqr_f220:.3f} > {T['MAX_REL_IQR_F220']}: "
            "fundamental frequency poorly constrained"
        )
        return _viability_result("RINGDOWN_NONINFORMATIVE", reasons, metrics, T)

    # ════════════════════════════════════════════
    # GATE 1: ¿El 221 tiene calidad mínima?
    # ════════════════════════════════════════════

    vf_221 = inputs["valid_fraction_221"]
    metrics["valid_fraction_221"] = vf_221

    if vf_221 < T["MIN_VALID_FRAC_221"]:
        reasons.append(
            f"valid_fraction_221={vf_221:.3f} < {T['MIN_VALID_FRAC_221']}: "
            "overtone posterior insufficient"
        )
        return _viability_result("SINGLEMODE_ONLY", reasons, metrics, T)

    # ════════════════════════════════════════════
    # GATE 2: Colección de flags de severidad
    # ════════════════════════════════════════════

    severe = []

    # 2a. Spin floor
    sf = inputs.get("spin_at_floor_frac_221", 0.0)
    metrics["spin_at_floor_frac_221"] = sf
    if sf > T["MAX_SPIN_FLOOR_FRAC"]:
        severe.append("SPIN_AT_PHYSICAL_FLOOR")
        reasons.append(
            f"spin_at_floor_frac_221={sf:.3f} > {T['MAX_SPIN_FLOOR_FRAC']}"
        )

    # 2b. delta_bic
    dbic = inputs.get("delta_bic")
    metrics["delta_bic"] = dbic
    if dbic is not None and dbic < T["DELTA_BIC_SUPPORTIVE"]:
        severe.append("DELTA_BIC_UNSUPPORTIVE")
        reasons.append(
            f"delta_bic={dbic:.2f} < {T['DELTA_BIC_SUPPORTIVE']}: "
            "data do not prefer two-mode model"
        )

    # 2c. Estabilidad del 221
    stab = None
    if inputs["f_221_median"] is not None and inputs["f_221_median"] > 0:
        stab = inputs["f_221_iqr"] / inputs["f_221_median"]
        metrics["stability_221"] = stab
        if stab > T["STABILITY_221_MAX"]:
            severe.append("UNSTABLE_221_POSTERIOR")
            reasons.append(f"stability_221={stab:.3f} > {T['STABILITY_221_MAX']}")

    # 2d. Informatividad Rf
    Rf_q = inputs.get("Rf_bootstrap_quantiles")
    Rf_kerr = inputs["Rf_kerr_band"]
    informativity = None

    if Rf_q is not None:
        Rf_lo, Rf_hi = Rf_q["q05"], Rf_q["q95"]
        dRf_obs = Rf_hi - Rf_lo
        dRf_kerr = Rf_kerr[1] - Rf_kerr[0]
        informativity = 1.0 - min(1.0, dRf_obs / dRf_kerr) \
                        if dRf_kerr > 0 else 0.0
        metrics["informativity_Rf"] = informativity
        metrics["delta_Rf_obs"] = dRf_obs
        metrics["delta_Rf_kerr"] = dRf_kerr

        overlap_lo = max(Rf_lo, Rf_kerr[0])
        overlap_hi = min(Rf_hi, Rf_kerr[1])
        overlap = max(0.0, overlap_hi - overlap_lo)
        metrics["overlap_with_kerr"] = overlap
        metrics["overlap_frac_obs"] = overlap / dRf_obs if dRf_obs > 0 else 0.0
        metrics["overlap_frac_kerr"] = overlap / dRf_kerr if dRf_kerr > 0 else 0.0
        metrics["kerr_consistent"] = overlap > 0

        if informativity < T["INFORMATIVE_THRESHOLD"]:
            severe.append("UNINFORMATIVE_RF")
            reasons.append(
                f"informativity_Rf={informativity:.3f} < {T['INFORMATIVE_THRESHOLD']}"
            )
    else:
        metrics["informativity_Rf"] = None
        metrics["kerr_consistent"] = None
        severe.append("RF_NOT_COMPUTABLE")
        reasons.append("Rf_obs not available")

    metrics["n_severe_flags"] = len(severe)
    metrics["severe_flags"] = severe

    # ════════════════════════════════════════════
    # GATE 3: Decisión
    # ════════════════════════════════════════════

    if len(severe) >= T["SEVERE_COUNT_LIMIT"]:
        reasons.append(
            f"n_severe={len(severe)} >= {T['SEVERE_COUNT_LIMIT']}: "
            "degraded to SINGLEMODE_ONLY"
        )
        return _viability_result("SINGLEMODE_ONLY", reasons, metrics, T)

    if metrics.get("kerr_consistent") is False:
        reasons.append(
            "Rf_obs outside Kerr band: potential inconsistency "
            "(verify systematics before interpreting)"
        )

    reasons.append("overtone passes minimum viability checks")
    return _viability_result("MULTIMODE_OK", reasons, metrics, T)


def _viability_result(cls, reasons, metrics, thresholds):
    return {
        "class": cls,
        "reasons": reasons,
        "metrics": metrics,
        "thresholds_used": thresholds,
        "schema_version": "multimode_viability_v1",
    }
```

---

## 8. Gate de sistemáticas (`systematics_gate`)

### 8.1 Responsabilidad

Decide si las condiciones instrumentales/algorítmicas permiten interpretar los observables como señal astrofísica. Se evalúa **solo si** `multimode_viability.class == MULTIMODE_OK`.

### 8.2 Checks deterministas

| Check | Input | Métrica | Condición PASS | Condición FAIL |
|-------|-------|---------|---------------|----------------|
| `t0_plateau` | t0 sweep results (oracle report) | `plateau_detected` + `f_std_over_plateau_hz` | `plateau_detected == True` AND `f_std_over_plateau_hz < threshold` | Plateau no detectado o variación excesiva |
| `psd_sanity` | Método Brunete ($\chi_{\text{PSD}}$) | $\chi_{\text{PSD}}$ en la frecuencia del 221 | $\chi_{\text{PSD}} < 0.1$ | Contaminación PSD domina curvatura |
| `estimator_resolution` | $Q_{221}$ del posterior | `Q_221_median` | $Q_{221}^{\text{median}} > 1.5$ | Resolución espectral insuficiente |

### 8.3 Lógica

```python
def evaluate_systematics_gate(inputs: dict) -> dict:
    """
    Función pura. Evalúa checks de sistemáticas.
    Retorna verdict_auto ∈ {PASS, FAIL, NOT_AVAILABLE}.
    """
    checks = {}

    # ── t0 plateau ──
    t0 = inputs.get("t0_plateau")
    if t0 is None:
        checks["t0_plateau"] = {
            "verdict": "NA",
            "metric": None,
            "threshold": None,
        }
    else:
        t0_pass = (
            t0["plateau_detected"] is True
            and t0["f_std_over_plateau_hz"] < T_T0_STD_MAX  # e.g. 5.0 Hz
        )
        checks["t0_plateau"] = {
            "verdict": "PASS" if t0_pass else "FAIL",
            "metric": t0["f_std_over_plateau_hz"],
            "threshold": T_T0_STD_MAX,
        }

    # ── PSD sanity ──
    psd = inputs.get("chi_psd_at_f221")
    if psd is None:
        checks["psd_sanity"] = {
            "verdict": "NA",
            "metric": None,
            "threshold": None,
        }
    else:
        psd_pass = psd < T_CHI_PSD_MAX  # e.g. 0.10
        checks["psd_sanity"] = {
            "verdict": "PASS" if psd_pass else "FAIL",
            "metric": psd,
            "threshold": T_CHI_PSD_MAX,
        }

    # ── Estimator resolution ──
    Q221 = inputs.get("Q_221_median")
    if Q221 is None:
        checks["estimator_resolution"] = {
            "verdict": "NA",
            "metric": None,
            "threshold": None,
        }
    else:
        res_pass = Q221 > T_Q221_MIN  # e.g. 1.5
        checks["estimator_resolution"] = {
            "verdict": "PASS" if res_pass else "FAIL",
            "metric": Q221,
            "threshold": T_Q221_MIN,
        }

    # ── Verdict ──
    verdicts = [c["verdict"] for c in checks.values()]
    if "FAIL" in verdicts:
        verdict_auto = "FAIL"
    elif all(v == "PASS" for v in verdicts):
        verdict_auto = "PASS"
    else:
        verdict_auto = "NOT_AVAILABLE"

    return {
        "schema_version": "systematics_gate_v1",
        "verdict_auto": verdict_auto,
        "checks": checks,
        "thresholds_used": {
            "T_T0_STD_MAX": T_T0_STD_MAX,
            "T_CHI_PSD_MAX": T_CHI_PSD_MAX,
            "T_Q221_MIN": T_Q221_MIN,
        },
    }
```

### 8.4 Umbrales del gate de sistemáticas

| Parámetro | Valor sugerido | Justificación |
|-----------|---------------|---------------|
| `T_T0_STD_MAX` | 5.0 Hz | Desviación estándar de $f$ sobre el plateau del t0 sweep |
| `T_CHI_PSD_MAX` | 0.10 | $\chi_{\text{PSD}} > 0.1$ → curvatura dominada por instrumento (Método Brunete §7.14) |
| `T_Q221_MIN` | 1.5 | $Q_{221} < 1.5$ → menos de 1.5 ciclos, el Lorentziano no resuelve el modo |

### 8.5 Override humano (opcional, solo degradante)

Si existe `runs/<run_id>/annotations/systematics_override.json`:

```json
{
  "schema_version": "systematics_override_v1",
  "verdict_human": "FAIL",
  "reason": "Manual glitch identification at t=0.42s in L1",
  "analyst_id": "NRG",
  "timestamp_utc": "2026-03-03T10:00:00Z",
  "sha256_self": "..."
}
```

**Reglas de combinación (deterministas):**

| `verdict_auto` | `verdict_human` | `verdict_final` | Reason |
|----------------|-----------------|------------------|--------|
| PASS | null (no override) | PASS | — |
| PASS | FAIL | FAIL | `HUMAN_VETO` |
| PASS | PASS | PASS | (override redundante, no cambia nada) |
| FAIL | PASS | FAIL | (override no puede promover) |
| FAIL | FAIL | FAIL | — |
| NOT_AVAILABLE | cualquiera | NOT_AVAILABLE | (sin datos, no hay qué overridear) |

El override **solo degrada, nunca promueve**. Si `verdict_auto == FAIL`, ninguna anotación humana lo convierte en PASS.

---

## 9. Canal de evidencia (`science_evidence`)

### 9.1 Responsabilidad

Cuantifica la desviación observada respecto de Kerr. Se evalúa **solo si** ambas condiciones se cumplen:

1. `multimode_viability.class == MULTIMODE_OK`
2. `systematics_gate.verdict_final == PASS`

Si alguna falla, `status = NOT_EVALUATED` con `reason_if_skipped` explícito.

### 9.2 H1 mínima: $\Delta R_f$ (determinista)

**Cálculo:**

1. Sea $R_f^{\text{obs, med}} = $ mediana del bootstrap de $R_f$.
2. Sea $\chi^* = \arg\min_{\chi_i} |R_f^{\text{obs, med}} - R_f^{\text{Kerr}}(\chi_i)|$ (búsqueda en grid del golden file).
3. $\Delta R_f = R_f^{\text{obs, med}} - R_f^{\text{Kerr}}(\chi^*)$.

**Intervalo:** de los cuantiles del bootstrap:

$$\Delta R_f^{\text{lo}} = R_f^{q_{05}} - R_f^{\text{Kerr}}(\chi^*), \qquad \Delta R_f^{\text{hi}} = R_f^{q_{95}} - R_f^{\text{Kerr}}(\chi^*) \tag{9.1}$$

**Interpretación:**

- Si el intervalo $[\Delta R_f^{\text{lo}}, \Delta R_f^{\text{hi}}]$ contiene 0: consistente con H0.
- Si no contiene 0 **y** `systematics_gate == PASS`: candidato a investigación bajo H1.

### 9.3 Lógica

```python
def evaluate_science_evidence(
    viability: dict,
    systematics: dict,
    Rf_bootstrap_quantiles: dict | None,
    Rf_kerr_grid: list[float],
    chi_grid: list[float],
    override: dict | None,
) -> dict:
    """
    Función pura. Produce science_evidence con status y H1_min.
    """
    reason_if_skipped = []

    # ── Precondición: viabilidad ──
    if viability["class"] != "MULTIMODE_OK":
        reason_if_skipped.append("MULTIMODE_GATE")

    # ── Precondición: sistemáticas ──
    verdict_final = _combine_verdicts(
        systematics["verdict_auto"],
        override.get("verdict_human") if override else None
    )
    if verdict_final != "PASS":
        if verdict_final == "FAIL":
            reason_if_skipped.append("SYSTEMATICS_FAIL")
        else:
            reason_if_skipped.append("SYSTEMATICS_NOT_AVAILABLE")

    if reason_if_skipped:
        return {
            "schema_version": "science_evidence_v1",
            "status": "NOT_EVALUATED",
            "reason_if_skipped": reason_if_skipped,
            "H1_min": {"delta_Rf": None},
            "future_slots": {
                "delta_f_221": None,
                "delta_tau_221": None,
                "log_bayes_factor": None,
            },
        }

    # ── Evaluar H1_min ──
    if Rf_bootstrap_quantiles is None:
        return {
            "schema_version": "science_evidence_v1",
            "status": "NOT_EVALUATED",
            "reason_if_skipped": ["RF_NOT_COMPUTABLE"],
            "H1_min": {"delta_Rf": None},
            "future_slots": {
                "delta_f_221": None,
                "delta_tau_221": None,
                "log_bayes_factor": None,
            },
        }

    Rf_med = Rf_bootstrap_quantiles["q50"]
    Rf_lo = Rf_bootstrap_quantiles["q05"]
    Rf_hi = Rf_bootstrap_quantiles["q95"]

    # Encontrar chi* más cercano
    diffs = [abs(Rf_med - Rf_k) for Rf_k in Rf_kerr_grid]
    idx_star = diffs.index(min(diffs))
    chi_star = chi_grid[idx_star]
    Rf_kerr_star = Rf_kerr_grid[idx_star]

    delta_Rf = Rf_med - Rf_kerr_star
    delta_Rf_lo = Rf_lo - Rf_kerr_star
    delta_Rf_hi = Rf_hi - Rf_kerr_star
    contains_zero = (delta_Rf_lo <= 0.0 <= delta_Rf_hi)

    return {
        "schema_version": "science_evidence_v1",
        "status": "EVALUATED",
        "reason_if_skipped": [],
        "H1_min": {
            "delta_Rf": {
                "value": delta_Rf,
                "interval": [delta_Rf_lo, delta_Rf_hi],
                "quantiles": [0.05, 0.95],
                "chi_star": chi_star,
                "Rf_kerr_at_chi_star": Rf_kerr_star,
                "contains_zero": contains_zero,
                "definition": "Rf_obs_median - Rf_Kerr(chi_star)",
            },
        },
        "future_slots": {
            "delta_f_221": None,
            "delta_tau_221": None,
            "log_bayes_factor": None,
        },
    }


def _combine_verdicts(verdict_auto, verdict_human):
    """Override solo degrada, nunca promueve."""
    if verdict_auto == "NOT_AVAILABLE":
        return "NOT_AVAILABLE"
    if verdict_auto == "FAIL":
        return "FAIL"
    # verdict_auto == PASS
    if verdict_human == "FAIL":
        return "FAIL"
    return "PASS"
```

---

## 10. Failure modes y falsos positivos

| # | Escenario | Mecanismo | Métrica de distinción | Check en `systematics_gate` |
|---|-----------|-----------|----------------------|-----------------------------|
| F1 | **Ventana $t_0$ mal elegida** | Merger contamina $f_{221}$ | Variación de $R_f$ en t0 sweep | `t0_plateau` |
| F2 | **SNR insuficiente del 221** | Ruido domina posterior | $\mathcal{I}_{R_f} < 0.3$, `valid_fraction_221` bajo | (gate de viabilidad) |
| F3 | **Sesgo del estimador** | $Q_{221} < 1.5$: Lorentziano no resuelve | $Q_{221}^{\text{median}}$ | `estimator_resolution` |
| F4 | **Contaminación PSD** | Línea instrumental cerca de $f_{221}$ | $\chi_{\text{PSD}}$ (Brunete §7.14) | `psd_sanity` |
| F5 | **No-estacionariedad** | Glitches durante ringdown | Anderson-Darling residuos | (override humano si identificado) |
| F6 | **Degeneración a spins bajos** | $dR_f/d\chi$ pequeño: inversión inestable | Evaluar derivada en $\chi^*$ | (diagnóstico, no gate) |
| F7 | **SPIN_AT_PHYSICAL_FLOOR** | Estimador satura spin | `spin_at_floor_frac_221 > 0.30` | (gate de viabilidad) |

**Principio:** el canal de evidencia (`science_evidence.status == EVALUATED`) solo se abre si ninguno de los failure modes controlables está activo.

---

## 11. Especificación de artefactos (contract-first)

### 11.1 Bloque completo en `stage_summary.json`

```json
{
  "multimode_viability": {
    "schema_version": "multimode_viability_v1",
    "class": "MULTIMODE_OK",
    "reasons": ["..."],
    "metrics": {"...": "..."},
    "thresholds_used": {"...": "..."},
    "Rf_obs_interval": [0.905, 0.947],
    "Rf_kerr_band": [0.88, 1.00]
  },

  "systematics_gate": {
    "schema_version": "systematics_gate_v1",
    "verdict_auto": "PASS",
    "verdict_final": "PASS",
    "checks": {
      "t0_plateau": {"verdict": "PASS", "metric": 2.3, "threshold": 5.0},
      "psd_sanity": {"verdict": "PASS", "metric": 0.054, "threshold": 0.10},
      "estimator_resolution": {"verdict": "PASS", "metric": 1.85, "threshold": 1.5}
    },
    "thresholds_used": {
      "T_T0_STD_MAX": 5.0,
      "T_CHI_PSD_MAX": 0.10,
      "T_Q221_MIN": 1.5
    },
    "inputs": {
      "t0_sweep_ref": "runs/<run_id>/experiment/oracle_t0_ringdown/outputs/oracle_report.json",
      "psd_ref": "runs/<run_id>/s6c_brunete/outputs/curvature_diagnostics.json"
    }
  },

  "science_evidence": {
    "schema_version": "science_evidence_v1",
    "status": "EVALUATED",
    "reason_if_skipped": [],
    "H1_min": {
      "delta_Rf": {
        "value": 0.003,
        "interval": [-0.012, 0.018],
        "quantiles": [0.05, 0.95],
        "chi_star": 0.67,
        "Rf_kerr_at_chi_star": 0.935,
        "contains_zero": true,
        "definition": "Rf_obs_median - Rf_Kerr(chi_star)"
      }
    },
    "future_slots": {
      "delta_f_221": null,
      "delta_tau_221": null,
      "log_bayes_factor": null
    }
  },

  "annotations": {
    "systematics_override_ref": null,
    "systematics_override_sha256": null
  }
}
```

**Nota:** los valores numéricos son ficticios para ilustrar la estructura.

### 11.2 `outputs/kerr_ratio_reference.json` (golden file)

Sin cambios respecto a v2 (§8.2). Grid de $R_f(\chi)$, $R_Q(\chi)$, $Q_{220}(\chi)$, $Q_{221}(\chi)$ con SHA256.

### 11.3 `annotations/systematics_override.json` (opcional)

Schema en §8.5. Solo degrada. Hasheado en manifest.

---

## 12. Tests de regresión

### 12.1 Tests de clasificación (viabilidad)

```python
def test_multimode_ok():
    """Posterior estrecho, compatible, delta_bic positivo."""
    r = classify_multimode_viability(FIXTURE_GOOD)
    assert r["class"] == "MULTIMODE_OK"
    assert r["metrics"]["kerr_consistent"] is True

def test_singlemode_broad():
    """221 ancho + delta_bic bajo → 2 flags → SINGLEMODE_ONLY."""
    r = classify_multimode_viability(FIXTURE_BROAD)
    assert r["class"] == "SINGLEMODE_ONLY"
    assert r["metrics"]["n_severe_flags"] >= 2

def test_noninformative():
    """220 con valid_fraction < 0.50 → RINGDOWN_NONINFORMATIVE."""
    r = classify_multimode_viability(FIXTURE_BAD_220)
    assert r["class"] == "RINGDOWN_NONINFORMATIVE"

def test_kerr_inconsistent_narrow():
    """Posterior estrecho pero fuera de banda Kerr."""
    r = classify_multimode_viability(FIXTURE_INCONSISTENT)
    assert r["class"] == "MULTIMODE_OK"
    assert r["metrics"]["kerr_consistent"] is False

def test_spin_floor_plus_uninformative():
    """Dos flags severos → SINGLEMODE_ONLY."""
    r = classify_multimode_viability(FIXTURE_SPIN_FLOOR)
    assert r["class"] == "SINGLEMODE_ONLY"

def test_determinism():
    """Mismos inputs → mismo resultado exacto."""
    r1 = classify_multimode_viability(FIXTURE_GOOD)
    r2 = classify_multimode_viability(FIXTURE_GOOD)
    assert r1 == r2
```

### 12.2 Tests del canal de evidencia

```python
def test_evidence_not_evaluated_if_singlemode():
    """Si class != MULTIMODE_OK → status NOT_EVALUATED + MULTIMODE_GATE."""
    viab = {"class": "SINGLEMODE_ONLY"}
    syst = {"verdict_auto": "PASS"}
    ev = evaluate_science_evidence(viab, syst, RF_Q_FIXTURE, RF_KERR, CHI_GRID, None)
    assert ev["status"] == "NOT_EVALUATED"
    assert "MULTIMODE_GATE" in ev["reason_if_skipped"]

def test_evidence_not_evaluated_if_systematics_fail():
    """Si systematics FAIL → NOT_EVALUATED + SYSTEMATICS_FAIL."""
    viab = {"class": "MULTIMODE_OK"}
    syst = {"verdict_auto": "FAIL"}
    ev = evaluate_science_evidence(viab, syst, RF_Q_FIXTURE, RF_KERR, CHI_GRID, None)
    assert ev["status"] == "NOT_EVALUATED"
    assert "SYSTEMATICS_FAIL" in ev["reason_if_skipped"]

def test_evidence_block_always_exists():
    """science_evidence siempre tiene schema_version y status."""
    for viab_class in ["MULTIMODE_OK", "SINGLEMODE_ONLY", "RINGDOWN_NONINFORMATIVE"]:
        viab = {"class": viab_class}
        syst = {"verdict_auto": "PASS"}
        ev = evaluate_science_evidence(viab, syst, RF_Q_FIXTURE, RF_KERR, CHI_GRID, None)
        assert "schema_version" in ev
        assert "status" in ev
        assert ev["schema_version"] == "science_evidence_v1"

def test_override_only_degrades():
    """Override humano PASS no promueve FAIL → sigue FAIL."""
    viab = {"class": "MULTIMODE_OK"}
    syst = {"verdict_auto": "FAIL"}
    override = {"verdict_human": "PASS"}
    ev = evaluate_science_evidence(viab, syst, RF_Q_FIXTURE, RF_KERR, CHI_GRID, override)
    assert ev["status"] == "NOT_EVALUATED"

    # Override FAIL degrada PASS
    syst2 = {"verdict_auto": "PASS"}
    override2 = {"verdict_human": "FAIL"}
    ev2 = evaluate_science_evidence(viab, syst2, RF_Q_FIXTURE, RF_KERR, CHI_GRID, override2)
    assert ev2["status"] == "NOT_EVALUATED"
    assert "SYSTEMATICS_FAIL" in ev2["reason_if_skipped"] or \
           "HUMAN_VETO" in ev2.get("reason_if_skipped", [])
```

### 12.3 Tests de invariantes físicos

```python
def test_Rf_monotonicity():
    """Rf aproximadamente creciente con chi."""
    ref = load_json("kerr_ratio_reference.json")
    Rf = ref["Rf_grid"]
    n_incr = sum(1 for i in range(len(Rf)-1) if Rf[i+1] >= Rf[i] - 1e-4)
    assert n_incr / (len(Rf) - 1) > 0.90

def test_overtone_decays_faster():
    """Q221 < Q220 para todo chi."""
    ref = load_json("kerr_ratio_reference.json")
    for q220, q221 in zip(ref["Q220_grid"], ref["Q221_grid"]):
        assert q220 > q221

def test_no_nan_in_classification():
    """Ningún float de metrics es NaN."""
    for fx in ALL_FIXTURES:
        r = classify_multimode_viability(fx)
        for k, v in r["metrics"].items():
            if isinstance(v, float):
                assert math.isfinite(v), f"NaN/Inf en {k}"

def test_classification_exhaustive():
    """Todo input produce exactamente una de las tres clases."""
    VALID = {"MULTIMODE_OK", "SINGLEMODE_ONLY", "RINGDOWN_NONINFORMATIVE"}
    for fx in ALL_FIXTURES:
        r = classify_multimode_viability(fx)
        assert r["class"] in VALID
        assert len(r["reasons"]) > 0
```

---

## 13. Supuestos explícitos

1. **Modelo de señal:** dos QNMs Kerr (220, 221), amplitudes y fases libres. No se incluyen $\ell \neq 2$, contra-rotantes, ni acoplamientos cuadráticos.
2. **Estimador:** s3b produce bootstrap espectral con muestras pareadas de $(f_{220}, f_{221})$.
3. **`delta_bic`:** determinista desde RSS de `model_comparison.json`.
4. **Fits Kerr:** `qnm` package, Leaver method, precisión $\sim 10^{-4}$.
5. **Umbrales no calibrados:** convenciones operativas iniciales.
6. **Sin dependencias del inspiral:** test puramente espectral.
7. **Override humano:** solo degrada, nunca promueve. Entra como input hasheado.

---

## 14. Qué NO concluye este informe

1. No afirma detectabilidad del 221 en eventos concretos de GWTC.
2. No calibra umbrales con inyecciones.
3. No implementa ningún sampler bayesiano. Los slots futuros (`delta_f_221`, `log_bayes_factor`) quedan como `null`.
4. No aborda modos $(\ell, m) \neq (2, 2)$.
5. No evalúa el impacto de $t_0$ — dominio del t0 sweep.
6. No trata combinación multi-evento de $\Delta R_f$ — dominio de ex4.
7. No propone inversión conjunta $(R_f, R_Q) \to \chi$.

---

## 15. Checklist para implementación

1. **[ ] Generar `kerr_ratio_reference.json`**: script standalone, `qnm`, grid $\chi \in [0, 0.998]$, SHA256.

2. **[ ] Implementar `classify_multimode_viability()`**: función pura en `mvp/multimode_viability.py`. Solo stdlib + math.

3. **[ ] Implementar `evaluate_systematics_gate()`**: función pura. Inputs: oracle report de t0 sweep, diagnósticos Brunete, $Q_{221}$ del posterior.

4. **[ ] Implementar `evaluate_science_evidence()`**: función pura. Calcula $\Delta R_f$ y rellena el bloque.

5. **[ ] Implementar `_combine_verdicts()`**: lógica de override (4 líneas).

6. **[ ] Implementar cálculo de $R_f$ muestra a muestra en s3b**: del bootstrap pareado, calcular $R_f^{(k)}$, extraer cuantiles 5%/50%/95%.

7. **[ ] Definir schema `systematics_override_v1`**: JSON con `verdict_human`, `reason`, `analyst_id`, `timestamp_utc`, `sha256_self`.

8. **[ ] Integrar los tres bloques en `stage_summary.json`**: `multimode_viability` + `systematics_gate` + `science_evidence` + `annotations`.

9. **[ ] Registrar contrato en `contracts.py`**: inputs/outputs formales.

10. **[ ] Escribir 14 tests**: 6 viabilidad + 4 evidencia + 4 invariantes.

11. **[ ] Documentar semántica de `NOT_EVALUATED` vs `EVALUATED`**: en qué condiciones exactas se abre el canal.

12. **[ ] Función helper `make_fixture()`**: constructor de inputs con defaults para tests.

---

*Fin del informe v3.0. Determinismo puro. Canal formal de evidencia. Override solo degradante.*
