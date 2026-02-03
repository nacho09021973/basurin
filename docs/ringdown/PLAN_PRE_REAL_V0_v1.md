EXP_RINGDOWN_01 — Injection–recovery (juguete realista)

Propósito: demostrar que el pipeline recupera parámetros inyectados dentro de tolerancias explícitas.

Riesgo: sesgo sistemático, bug silencioso en el estimador, o leakage de parámetros.

Inputs canónicos:

Generador sintético canónico (si ya existe: ringdown_synth_stage o equivalente).

Config explícita de inyección (masa/frecuencia/amortiguamiento/SNR/PSD) registrada.

Precheck RUN_VALID == PASS.

Outputs esperados:

outputs/injections.jsonl (truth por caso)

outputs/recoveries.jsonl (estimados por caso)

outputs/summary_metrics.json (bias, RMSE, cobertura si hay intervalos)

contract_verdict.json

Contrato(s):

R01_BIAS_BOUNDED: |bias| < umbral por parámetro.

R01_RECOVERY_RATE: fracción de casos válidos > umbral.

R01_CALIBRATION_OPTIONAL: si hay σ/intervalos, cobertura aproximada (post-hoc, no como señal).

Estado: Requiere implementación/ejecución (en tu historial quedó como “lo hacemos en chat nuevo”).

EXP_RINGDOWN_02 — Model selection / identifiability (familias mínimas)

Propósito: decidir si necesitas selección de modelo (p.ej. 1 modo vs 2 modos, o distinta parametrización) para no arrastrar underfit/overfit a “real”.

Riesgo: degeneración estructural (varios modelos explican igual), o selección inestable.

Inputs canónicos:

Dataset de EXP01 (injections+recoveries) o dataset sintético dedicado con dos familias controladas.

Precheck RUN_VALID.

Outputs esperados:

outputs/model_comparison.json (métricas + penalización)

outputs/per_case_model_scores.jsonl

contract_verdict.json

Contrato(s):

R02_SELECTION_STABLE: el modelo ganador se mantiene bajo re-muestreo determinista o folds deterministas.

R02_DELTA_IC: diferencia mínima en criterio (AIC/BIC/ELPD) para declarar victoria; si no, INCONCLUSIVE (y aborta downstream).

Estado: No verificado aquí (probable “por hacer” salvo que ya exista script/test en tu repo).

EXP_RINGDOWN_03 — Observable mínimo adicional (romper degeneración)

Propósito: identificar el observable mínimo que rompe una degeneración detectada (análogo a lo que ya hiciste en el track “dual spectrum” en otro dominio).

Riesgo: pipeline inverso colapsa a baja dimensión; resultados no identificables.

Inputs canónicos:

Dataset base (p.ej. outputs de EXP01/EXP02).

Un segundo observable formalizado (si no existe, se define como artefacto canónico nuevo).

Outputs esperados:

outputs/identifiability_report.json (dim efectiva, métricas)

outputs/ablations.jsonl (con/sin observable)

contract_verdict.json

Contrato(s):

R03_IDENTIFIABILITY_GAIN: evidencia cuantitativa de ruptura (p.ej. rank efectivo, mutual info estimada, o mejora fuera de tolerancia predefinida).

R03_MINIMALITY: el observable propuesto es el mínimo de un set discreto (barrido determinista).

Estado: No verificado aquí (conceptualmente ya lo trabajaste en otra línea; para ringdown hay que cerrarlo con datos/artefactos ringdown).

EXP_RINGDOWN_04 — (Bloqueante) Stage/definición faltante

Propósito: no definido en el estado actual (aparece como hueco histórico).

Riesgo: cronograma 00..08 no es ejecutable “end-to-end” si falta un gate numerado/consumido downstream.

Inputs/Outputs/Contratos: no se deben inventar.

➡️ Acción mínima recomendada: formalizar EXP_RINGDOWN_04 como gate explícito con una de estas dos opciones (elige una, pero no avanzamos sin fijarla):

Opción A (infra real v0): “Data ingest & canonicalization” (lectura de strain real/sintético, normalización, segmentación) → produce un artefacto canónico gw_strain.h5/strain.npy + psd.h5 con trazabilidad.

Opción B (robustez de estimación): “Likelihood/PSD validity gate” (asegura PSD bien condicionada + whitening sin singularidades) → produce psd_diagnostics.json + contract_verdict.

Estado: FALTANTE (bloqueante) hasta que se formalice.

EXP_RINGDOWN_05 — Robustez a priors / hiperparámetros

Propósito: demostrar que conclusiones no dependen de priors razonables o hiperparámetros del estimador.

Riesgo: “prior-dominated inference” o sensibilidad a regularización.

Inputs canónicos:

Dataset de EXP01 (o dataset sintético representativo fijado).

Configs de priors/hyperparams enumeradas determinísticamente.

Outputs esperados:

outputs/prior_sweep.json (comparativas)

outputs/per_case.jsonl

contract_verdict.json

Contrato(s):

R05_PRIOR_SENSITIVITY_BOUNDED: variación acotada en métricas clave.

R05_FAILURE_MODE_CAP: tasa de fallos no aumenta por encima de umbral.

Estado: No verificado aquí.

EXP_RINGDOWN_06 — Robustez a PSD / ruido coloreado (mis-especificación)

Propósito: probar sensibilidad a PSD mal estimada / variabilidad de PSD (sin todavía usar real).

Riesgo: resultados que cambian por elección de PSD.

Inputs canónicos:

Dataset sintético con varias PSD predefinidas o perturbaciones deterministas de PSD.

Outputs esperados:

outputs/psd_sweep_metrics.json

outputs/psd_cases.jsonl

contract_verdict.json

Contrato(s):

R06_PSD_ROBUSTNESS: degradación limitada (p.ej. RMSE no empeora > X%).

R06_DIAGNOSTICS_COMPLETE: diagnósticos mínimos presentes.

Estado: No verificado aquí.

EXP_RINGDOWN_07 — No-estacionariedad / glitches (estrés controlado)

Propósito: stress test contra no-estacionariedad (drift, líneas, glitches) con inyecciones controladas.

Riesgo: “real v0” se rompe en presencia de condiciones realistas.

Inputs canónicos:

Generador de ruido no-estacionario determinista (si no existe → stage canónico nuevo).

Inyecciones controladas (EXP01).

Outputs esperados:

outputs/nonstationary_report.json

outputs/failure_catalog.jsonl

contract_verdict.json

Contrato(s):

R07_ROBUST_UNDER_NONSTATIONARY: tasa de fallos y sesgo dentro de umbral.

R07_FAIL_CATEGORIZED: todo FAIL debe quedar categorizado (no “unknown crash”).

Estado: No verificado aquí.

EXP_RINGDOWN_08 — “Real harness” sin pasar a real (smoke pre-real)

Propósito: validar la infra de “real v0” (rutas, formatos, trazabilidad, runners) sin aún ejecutar datos reales si no están todos los PASS anteriores.

Riesgo: bloqueo operativo al llegar a real (paths, formatos, tiempos, reproducibilidad).

Inputs canónicos:

Artefactos canónicos finales de 00..07 (o su subset mínimo para smoke).

Outputs esperados:

outputs/harness_report.json

contract_verdict.json

Contrato(s):

R08_END2END_CANONICAL_IO: todos los artefactos esperados existen y están dentro de runs/<run_id>/.

R08_ABORT_ON_UPSTREAM_FAIL: si algún upstream no PASS, aborta y no produce outputs.

Estado: No verificado aquí (pero es donde conviene anclar el runner secuencial tipo run_ringdown_pre_real.sh).