# Revisión metodológica y conceptual de la propuesta BASURIN sobre sectores geométricos y Rényi

## Resumen ejecutivo

La propuesta plantea un cambio de fase en BASURIN: dado que (según el propio documento) “ya está hecha” la extracción de poblaciones geométricas compatibles por evento, el siguiente paso sería (i) **clasificar** esas geometrías en **sectores geométricos**, (ii) **medir** concentración de soporte por sector, y (iii) **solo** para el sector hiperbólico, explorar si una **entropía de Rényi condicional** (entendida explícitamente como *observable informacional*, no como entropía termodinámica del remanente) aporta una señal útil y reproducible. fileciteturn0file0

El enfoque tiene potencial real en dos sentidos: (a) introduce disciplina epistemológica (“esto es experimental, no canónico”) y (b) intenta convertir un artefacto ya extraído en una **capa de lectura cuantitativa** para comparar eventos, detectar concentraciones y priorizar investigación. fileciteturn0file0 Sin embargo, tal como está, su defendibilidad científica depende críticamente de dos puntos que hoy aparecen **subespecificados**: **qué significa exactamente “sector geométrico”** (y con qué invariantes se decide) y **qué distribución** se alimenta a Rényi (y con qué regla de pesos auditable). fileciteturn0file0

Mi conclusión ejecutiva: la propuesta es metodológicamente razonable **si** se reformula en términos de contratos explícitos (definiciones + reglas de decisión + artefactos + tests) y se blinda contra (i) taxonomías ambiguas y (ii) “p-hacking” por exploración de múltiples relaciones/órdenes de Rényi sin control formal. fileciteturn0file0turn14file3turn8search4

## Diagnóstico de coherencia interna y supuestos implícitos

El documento es coherente en su narrativa de gobernanza: separa explícitamente “carril experimental” vs “posible promoción a canónico” y exige IO determinista bajo `runs/<run_id>/experiment/...` con gating por `RUN_VALID`. fileciteturn0file0turn5file4turn14file3 Esta alineación es importante porque BASURIN ya implementa un patrón **contract-first / fail-fast** centralizado (contratos por stage, hashing de artefactos, `require_run_valid`, `manifest.json` y `stage_summary.json`). fileciteturn14file3turn5file4

El salto conceptual “extracción completada → nueva fase de relaciones” es defendible como estrategia de investigación, pero solo si se hace explícito qué artefacto es la **fuente de verdad** de “población compatible”. En BASURIN existen al menos dos representaciones canónicas cercanas a esa idea: (a) el filtrado por compatibilidad del atlas (con ranking y pesos derivados de distancias) y (b) el artefacto canónico de “support region” por evento para la rama explícita “golden geometry”. fileciteturn5file13turn5file2turn5file9 Si la propuesta no fija cuál usa (o cómo concilia ambas), se abre un riesgo de resultados no comparables (p. ej., Rényi midiendo “concentración” de una población distinta entre eventos por artefactos distintos, no por física o información). fileciteturn0file0turn5file2turn5file13

El propio texto reconoce el principal riesgo epistemológico: la “hipótesis de Rényi” es una apuesta personal y no debe tratarse como afirmación consolidada; además explícitamente niega que esté afirmando termodinámica de agujero negro. fileciteturn0file0turn1search0turn0search2 Esto es un punto fuerte, porque en la literatura existe un uso de Rényi como sustituto termodinámico (p. ej. enfoques “Hawking–Rényi” o modelos no extensivos), lo cual es un marco *distinto* del de “observable informacional” y, por tanto, debe evitarse como lectura automática en BASURIN. citeturn1search0turn1search4

## Evaluación crítica metodológica y plausibilidad física

La extracción de geometrías posibles por evento es un buen punto de partida si (y solo si) la población está representada como un artefacto estable con trazabilidad y, preferiblemente, con pesos/diagnósticos ya disponibles. BASURIN ya materializa eso: el stage que filtra el atlas produce (al menos) un `compatible_set.json` y un `ranked_all_full.json` por evento/run, que sirven como base para cualquier consolidación evento–geometría con scores/pesos. fileciteturn5file13turn14file3 Asimismo, si el análisis pretende cruzar la geometría compatible con filtros físicos (multimodo, Hawking/área, dominio), BASURIN ya genera un artefacto canónico por evento que consolida esas decisiones en `event_support_region.json` (incluyendo estado downstream y razones). fileciteturn5file2turn5file9

La clasificación de “soluciones/candidatos” en sectores geométricos es, tal como está escrita, el punto más frágil: el catálogo propuesto incluye `HYPERBOLIC`, `SPHERICAL`, `ELLIPTIC`, `EUCLIDEAN`, `UNKNOWN`. fileciteturn0file0 Ese set es problemático por dos razones:

Primero, **ambigüedad semántica**: en geometría de curvatura constante, “hiperbólica/euclídea/elíptica” suele ser la tripleta estándar (negativa/cero/positiva) y “esférica” puede solaparse con “elíptica” (dependiendo de si se identifica antipodalmente o se habla de recubrimiento universal). citeturn4search52turn4search50turn4search47 Si BASURIN no dispone de invariantes globales/topológicos, mantener simultáneamente `SPHERICAL` y `ELLIPTIC` puede inducir pseudo-precisión: se etiqueta algo como “elíptico” sin tener base para distinguirlo de “esférico” más allá de la intuición. fileciteturn0file0turn4search50

Segundo, **problema de observabilidad**: la propia propuesta exige (correctamente) que la etiqueta sectorial se deduzca de “información intrínseca del atlas” y no de observables espectrales que no soporten la etiqueta. fileciteturn0file0 Esto es metodológicamente sano, pero es también una restricción dura: si el atlas no contiene invariantes que definan sector (curvatura espacial/horizonte/topología/estructura geométrica), entonces la clasificación no es “difícil”: es *indeterminada*. fileciteturn0file0turn5file13 En ese caso, lo científicamente defendible no es inventar una taxonomía, sino introducir explícitamente un **nuevo artefacto canónico de caracterización del atlas** (o declarar “UNKNOWN” masivo y asumir que la clasificación no está disponible). fileciteturn14file3turn0file0

En plausibilidad física, hay una alerta estructural si “sector hiperbólico” se interpreta como “horizonte de curvatura negativa/topología no esférica”: en GR 4D asintóticamente plana y bajo hipótesis estándar, existen restricciones topológicas fuertes para horizontes estacionarios (resultado clásico sobre topología esférica). citeturn2search48 A la vez, existen escenarios teóricos donde se admiten horizontes no esféricos (p. ej. en AdS y/o identificaciones topológicas), y se han estudiado agujeros negros con horizontes hiperbólicos/planos/esféricos. citeturn3search0turn3search5 Por tanto, si BASURIN está operando en un contexto “astrophysical ringdown ~ Kerr asintóticamente plano”, cualquier lectura “sector hiperbólico privilegiado físicamente” debe tratarse como hipótesis fuerte y probablemente incorrecta por defecto; el propio documento acierta al prohibir ese salto. fileciteturn0file0turn2search48

La transición “ya tenemos poblaciones” a “buscamos relaciones” también tiene un riesgo estadístico típico de exploración: en cuanto se buscan muchas relaciones (múltiples features, múltiples α de Rényi, múltiples filtros), la probabilidad de falsos positivos crece y requiere control explícito (p. ej. FDR o procedimientos de corrección) si se pretende tomar decisiones de promoción a canónico. citeturn8search4turn8search1

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["hyperbolic geometry poincare disk model","spherical geometry great circles illustration","euclidean vs hyperbolic vs spherical curvature comparison diagram"],"num_per_query":1}

## Rényi como observable informacional y no como entropía física

La parte más defendible de la hipótesis de Rényi es precisamente su posicionamiento: usar Rényi como **funcional sobre una distribución discreta** (de pesos/soporte sobre geometrías candidatas) es matemáticamente estándar. citeturn0search1turn0search2 El documento acierta al exigir una “definición operacional” y al negar su lectura termodinámica; esto reduce el choque con trabajos donde Rényi se introduce como reemplazo termodinámico del área/entropía de entity["people","Jacob Bekenstein","physicist"]–entity["people","Stephen Hawking","physicist"] y se discute estabilidad/fases. fileciteturn0file0turn1search0turn2search48

Dicho eso, hay dos problemas técnicos que deben resolverse para que “Rényi condicional” no sea un concepto flotante:

Primero, **“entropía de Rényi condicional” no es única** en la literatura: existen definiciones alternativas (p. ej. la de Arimoto y variantes relacionadas) precisamente porque la propiedad de cadena de Shannon no se generaliza automáticamente. citeturn0search2turn0search0 Esto significa que, si BASURIN usa el término “condicional”, debe declarar qué objeto computa: (a) *Rényi de la distribución condicionada* (lo más simple y auditables) o (b) alguna definición formal tipo Arimoto/Sibson (más sutil y fácil de malinterpretar). citeturn0search2turn0search0

Segundo, en el contexto propuesto, lo más robusto es evitar la ambigüedad y definir explícitamente el observable como:

- Sea `G` la variable discreta “geometry_id” y `S` la etiqueta de sector.
- Sea `w(g)` una distribución de pesos sobre geometrías candidatas (normalizada).
- Definir masa sectorial `P(S=s)=Σ_{g:S(g)=s} w(g)`.
- Para el sector hiperbólico `H`, definir la distribución condicionada `w_H(g)=w(g)/P(S=H)` sobre `g ∈ H` (si `P(S=H)>0`).
- Definir entonces `H_α(G | S=H) := (1/(1-α)) log Σ_{g∈H} w_H(g)^α` como *Rényi de la distribución condicionada al sector hiperbólico*. citeturn0search1turn0search2

Esta formulación es auditables y evita reclamar propiedades que solo tiene Shannon (α→1). citeturn0search2turn0search1 Además, ofrece una interpretación directa en términos de **“número efectivo”** de geometrías hiperbólicas: `N_eff,α = exp(H_α)` (o `2^{H_α}` si se usa log base 2), que es un estándar de interpretación de entropías generalizadas como “cardinalidad efectiva” (muy usado vía números de Hill). citeturn5search0turn5search1

La limitación conceptual clave: este observable será tan “físicamente significativo” como lo sea la **regla que produce w(g)**. Si `w(g)` es un softmax arbitrario sobre distancias, Rényi solo medirá propiedades de esa heurística. Si, en cambio, `w(g)` deriva de un artefacto ya tratado en BASURIN como *proxy de verosimilitud relativa* (p. ej. pesos tipo `exp(-½ d²)` registrados con trazabilidad), entonces Rényi se convierte en un resumen informacional razonable del “grado de concentración del soporte” bajo ese modelo. fileciteturn5file13turn5file9turn0file0

## Propuesta de mejora y plan experimental auditable

El diseño F0–F4 del documento es una buena estructura (consolidar → clasificar → definir distribución → calcular Rényi/relaciones → decidir promoción), pero necesita convertirse en un contrato operativo: definiciones, inputs/outputs, fallos, y tests de regresión. fileciteturn0file0turn14file3 Propongo el siguiente rediseño mínimo que respeta BASURIN (IO determinista, abort semantics, hashing) y reduce ambigüedad:

En F0, no construir una tabla “evento–geometría” ad hoc si ya existe información canónica. La forma robusta es que el “run experimental” consuma `s5_aggregate/outputs/aggregate.json` (como índice de runs/eventos) y, para cada run fuente, lea explícitamente `s4_geometry_filter/outputs/ranked_all_full.json` (para pesos y scores) y opcionalmente `s4k_event_support_region/outputs/event_support_region.json` (para “support region” filtrada físicamente). fileciteturn5file9turn5file13turn5file2 Esto fija trazabilidad por SHA-256 vía `check_inputs` y evita que “población” signifique cosas distintas por evento. fileciteturn14file3turn0file0

En F1, la clasificación por sectores debe formalizarse como un **mapa de sectores del atlas** versionado y auditable, no como algo “inventado” durante el cálculo de Rényi. En BASURIN, esto encaja como input externo hashado (y, preferiblemente, snapshot copiado a outputs). fileciteturn14file3turn0file0 Recomendación concreta: reemplazar el catálogo `SPHERICAL`+`ELLIPTIC` por una taxonomía mínimamente defendible según invariantes disponibles:

- Si solo hay signo de curvatura (local): `{NEGATIVE, ZERO, POSITIVE, UNKNOWN}` o `{HYPERBOLIC, EUCLIDEAN, POSITIVE, UNKNOWN}`. citeturn4search52turn4search47  
- Si existen invariantes globales/topológicos reales (p. ej. identificación antipodal / cocientes / género), entonces sí diferenciar “esférico” vs “elíptico” como una decisión **global**, no local. citeturn4search50turn4search47  

Si el atlas no tiene invariantes, el resultado correcto es `UNKNOWN` y se aborta la fase F2/F3 (porque no hay sector hiperbólico definible). fileciteturn0file0turn14file3

En F2, definir la distribución `w(g)` con una política explícita y testeada. Mi recomendación “mínimo cambio” es declarar dos políticas y compararlas:

- `weight_policy = posterior_from_ranked_all_full`: usar pesos ya registrados por el stage de filtrado (si están presentes). fileciteturn5file13  
- `weight_policy = uniform_over_support`: distribución uniforme sobre el conjunto de geometrías seleccionadas (compatible o “golden”), como baseline para comprobar que Rényi no es un reflejo directo de la propia función de scoring. fileciteturn0file0

En F3, calcular Rényi sobre `w_H(g)` para un conjunto pequeño y pre-registrado de órdenes α (p. ej. `{0, 1, 2, ∞}` o aproximaciones discretas razonables), porque cada α responde a una noción distinta (α→0 cuenta soporte, α→1 se acerca a Shannon, α→2 enfatiza colisiones, α→∞ enfatiza el máximo). citeturn0search1turn0search2 Hacer “barridos” densos de α sin control incrementa el riesgo de hallazgos espurios. citeturn8search4turn0search2

En F3 (relaciones), imponer desde el principio un protocolo de **validación por permutación** y control de múltiples tests: si se exploran muchas relaciones, usar permutaciones que preserven estructura (p. ej. barajar etiquetas de sector o barajar pesos dentro de evento bajo hipótesis nula) para construir distribuciones nulas, y controlar FDR para decidir qué relaciones sobreviven. citeturn9search2turn8search4 Este tipo de test es especialmente apropiado en BASURIN porque la unidad de análisis (eventos) puede ser pequeña y la forma de los estadísticos no necesita supuestos paramétricos fuertes si se respeta intercambiabilidad bajo nulo. citeturn9search2turn9search5

### Plan experimental sugerido con comandos, artefactos y tests

Ejecución propuesta como un nuevo stage experimental con nombre de carpeta canónico (slash en stage): `experiment/geometry_sectors_renyi`. Esto respeta el patrón de contratos de BASURIN (`init_stage`, `check_inputs`, `finalize`, `abort`) y deja outputs bajo `runs/<run_id>/experiment/geometry_sectors_renyi/...`. fileciteturn14file3turn0file0

Comando de ejecución (modelo de CLI; la idea es que el stage corra sobre el **run agregado** que contiene `s5_aggregate`):

```bash
python -m mvp.experiment.geometry_sectors_renyi \
  --run-id <AGG_RUN_ID> \
  --sector-map-path <PATH_A_SECTOR_MAP_JSON> \
  --population-source compatible_set \
  --weight-policy posterior_from_ranked_all_full \
  --alphas 0,1,2,inf
```

La estructura anterior es consistente con cómo BASURIN define y ejecuta experimentos/stages (contratos centralizados, check de `RUN_VALID`, y outputs hashados). fileciteturn14file3turn19file18

Artefactos esperados (mínimos para auditoría):

- `runs/<AGG_RUN_ID>/experiment/geometry_sectors_renyi/outputs/sector_map_snapshot.json` (copia exacta del mapa de sectores usado, para reproducibilidad). fileciteturn14file3turn0file0  
- `.../outputs/event_geometry_sector_table.csv` (o `.jsonl`), con columnas mínimas: `source_run_id,event_id,geometry_id,sector,weight,rank,d2_or_distance,selected_population`. fileciteturn5file13turn5file9  
- `.../outputs/renyi_per_event.json` con: `P_hyperbolic`, `n_hyperbolic`, `H_alpha_conditional` y `N_eff_alpha` por α. citeturn0search2turn5search1  
- `.../outputs/relations_report.json` que incluya: relaciones probadas, estadísticos, p-valores por permutación, y ajuste FDR (y parámetros exactos de permutación: nº permutaciones, semilla si aplica; idealmente sin aleatoriedad salvo pseudo-azar determinista controlado). citeturn9search2turn8search4  
- `manifest.json` y `stage_summary.json` auto-generados por `finalize`, con hash de outputs e inputs registrados. fileciteturn14file3

Tests mínimos que previenen regresión (unitarios, deterministas):

- Test de Rényi sobre distribución toy conocida (p. ej. 2–3 geometrías hiperbólicas con pesos exactos) y comparación con valor cerrado. citeturn0search1turn0search2  
- Test de “sector vacío”: si `P(S=H)=0`, el output debe marcar `H_alpha=null` + razón explícita (y no inventar un número). fileciteturn0file0turn14file3  
- Test de “mapa incompleto”: si un `geometry_id` aparece en población seleccionada y no existe en `sector_map`, abort con error que liste `geometry_id` faltantes (para no degradar silenciosamente a `UNKNOWN`). fileciteturn0file0turn14file3  
- Test de invariancia de orden: reordenar rows de input no debe cambiar outputs (ordenación estable por claves). fileciteturn14file3  

Criterios de promoción a canónico (operacionalizar lo ya dicho en el documento):

- Congelar taxonomía y reglas en una versión (`sector_map_version`, `weight_policy_version`) y demostrar estabilidad frente a cambios razonables (p. ej., `weight_policy` baseline vs posterior). fileciteturn0file0turn14file3  
- Evidencia de que Rényi aporta información no trivial frente a conteos simples (p. ej., `n_hyperbolic` vs `N_eff_α` divergen en casos reales de concentración). fileciteturn0file0turn5search1  
- Relaciones que sobreviven a validación por permutación y control FDR con criterio predefinido. citeturn9search2turn8search4  

## Señales de alerta y versión refinada de la propuesta

Señales de alerta (errores interpretativos que conviene cortar de raíz):

No confundir “Rényi sobre distribución de geometrías” con entropía física del remanente; existe literatura que usa Rényi en termodinámica de agujeros negros, pero ese es un marco distinto del planteado aquí (y mezclarlo inflaría indebidamente las conclusiones). fileciteturn0file0turn1search0turn1search4

No reinterpretar “sector hiperbólico” como propiedad física de horizontes en escenarios astrofísicos estándar sin una cadena explícita de supuestos: en GR 4D estacionaria asintóticamente plana hay restricciones topológicas hacia geometría esférica, mientras que horizontes hiperbólicos aparecen en otros marcos (p. ej. AdS). citeturn2search48turn3search0

No presentar como “señal” lo que puede ser un artefacto de (i) umbrales de compatibilidad, (ii) tamaño del atlas, o (iii) regla de pesos; por eso el baseline uniforme y la comparación entre políticas de pesos es parte del DoD. fileciteturn0file0turn5file13

No abrir un espacio de exploración masivo (muchos α, muchas relaciones, muchos cortes) sin control de múltiples tests: en un pipeline auditable, la promoción a canónico debe basarse en relaciones que sobreviven controles tipo permutación + FDR, no en hallazgos anecdóticos. citeturn9search2turn8search4

Versión refinada (núcleo reescrito, lista para base de trabajo):

BASURIN dispone de artefactos por evento que describen un conjunto discreto de geometrías candidatas (compatibles con observables del ringdown) y que contienen, cuando están disponibles, scores/pesos derivados del filtro del atlas. La siguiente fase experimental consiste en introducir una caracterización adicional del atlas: un mapa versiónado `geometry_id → sector` derivado exclusivamente de invariantes del modelo/atlas (no de los observables del evento). Con ese mapa, para cada evento se construye una distribución `w(g)` sobre geometrías candidatas (con política de pesos explícita y baseline uniforme). A partir de `w(g)`, se define la masa del sector hiperbólico `P(S=H)` y, si `P(S=H)>0`, la distribución condicionada `w_H(g)`. El observable experimental es la entropía de Rényi (o, preferiblemente, el número efectivo `N_eff,α`) calculada sobre `w_H(g)` para un conjunto pre-registrado de órdenes α. El objetivo no es afirmar una entropía física del remanente, sino medir concentración/dispersión informacional del soporte dentro del sector hiperbólico y evaluar si este resumen se relaciona de forma reproducible con otros filtros ya presentes en BASURIN (multimodo, Hawking/área, estado de dominio, etc.). Todo el experimento debe correr bajo `runs/<run_id>/experiment/...` con gating por `RUN_VALID`, registrar inputs por SHA-256 y emitir `manifest.json` y `stage_summary.json`. La promoción a canónico solo se considera si la taxonomía sectorial es estable, la política de pesos es auditable, y las relaciones sobreviven validación por permutación y control FDR. fileciteturn0file0turn14file3turn5file2turn5file13turn9search2turn8search4