# BRUNETE text fixes propuestos

## Discrepancia A.5 vs forma cerrada A.4/6.7 para `J0(σ)`

### Observación
La implementación y los tests de BASURIN toman como **source of truth** la forma cerrada:

\[
J_0(\sigma)=\pi\left[\left(\sigma+\frac{1}{2}\right)e^{\sigma}\,\mathrm{erfc}(\sqrt{\sigma})-\sqrt{\frac{\sigma}{\pi}}\right].
\]

Con esta definición:
- `J0(0) = π/2`.
- Para `σ \to \infty`, la caída dominante es `\propto σ^{-3/2}`.

En el documento `metodo_brunete.md` (A.5) aparece una ley `\propto π/(2σ)`, que no coincide con el asintótico derivado de la forma cerrada anterior.

### Comportamiento derivado de la forma cerrada
1. `J0(σ)` se evalúa directamente con (A.4/6.7), sin forzar A.5.
2. Los tests unitarios validan igualdad numérica contra esa forma cerrada.
3. Se mantiene un test de discrepancia para detectar cambios silenciosos que intenten alinear código con A.5 sin corregir el texto.

### Propuesta de corrección al texto
1. Mantener A.4/6.7 como definición normativa para implementación.
2. Corregir A.5 para que refleje el asintótico consistente con A.4/6.7 (orden `σ^{-3/2}`),
   o aclarar explícitamente si A.5 describe una magnitud reescalada distinta.
3. Añadir nota editorial en `metodo_brunete.md` para evitar ambigüedad entre definición cerrada y comentario asintótico.
