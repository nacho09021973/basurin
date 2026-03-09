# Event metadata (documentación)

Ficheros JSON con metadatos por evento (nombres GW* o fechas) usados para documentación y referencia.
No deben generarse durante runs. Si un stage necesita metadatos, deben resolverse como `external_input`
bajo `runs/<run_id>/external_inputs/...` y registrarse en manifest.

Campos opcionales adicionales aceptados por el router de familias:

- `source_class`: `binary_black_hole`, `binary_neutron_star`, etc.
- `preferred_families`: lista ordenada, por ejemplo `["GR_KERR_BH"]` o `["BNS_REMNANT", "GR_KERR_BH"]`
- `multimessenger`: `true/false`
- `family_priors`: bloque de priors por familia. Actualmente `s8b_family_bns` lee `family_priors.BNS_REMNANT`
  con claves como:
  - `remnant_mass_msun_range`
  - `radius_1p6_km_range`
  - `classes`
  - `n_mass_points`
  - `n_radius_points`
  - `collapse_time_ms_values`
