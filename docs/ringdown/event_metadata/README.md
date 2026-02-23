# Event metadata (documentación)

Ficheros JSON con metadatos por evento (nombres GW* o fechas) usados para documentación y referencia.
No deben generarse durante runs. Si un stage necesita metadatos, deben resolverse como `external_input`
bajo `runs/<run_id>/external_inputs/...` y registrarse en manifest.
