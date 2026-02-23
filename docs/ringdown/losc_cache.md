# Caché local LOSC (opcional)

BASURIN puede reutilizar HDF5 locales en `data/losc/` para evitar descargas repetidas.
Esta carpeta es *local-only* (gitignored) y nunca debe ser escrita por stages.

Orden recomendado en `s1_fetch_strain`:
1) `data/losc/<event_id>/...hdf5` si existe + verificación SHA256 (`INVENTORY.sha256`/`sha256.txt`)
2) `runs/<run_id>/external_inputs/...` si está presente
3) descarga canónica a `runs/<run_id>/external_inputs/...`
