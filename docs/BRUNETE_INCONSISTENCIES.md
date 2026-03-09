# BRUNETE inconsistencias detectadas (contract-first)

## Asintótico de `\mathcal{J}_0(\sigma)` en A.5

**Source of truth:** la forma cerrada de `\mathcal{J}_0(\sigma)` en (6.7)/(A.4):

\[
\mathcal{J}_0(\sigma)=\pi\left[\left(\sigma+\frac{1}{2}\right)e^{\sigma}\,\mathrm{erfc}(\sqrt{\sigma})-\sqrt{\frac{\sigma}{\pi}}\right].
\]

Al expandir para `\sigma\to\infty` usando `\mathrm{erfcx}(x)=e^{x^2}\mathrm{erfc}(x)`:

\[
\mathcal{J}_0(\sigma) = \frac{3\sqrt{\pi}}{8\sigma^{3/2}} + O(\sigma^{-5/2}),
\]

por lo que la caída dominante es `\sigma^{-3/2}`.

**CORREGIDO** en `docs/metodo_brunete.md` (2026-02-27): A.5 y la tabla de régimen en §6.5 ahora reflejan el asintótico correcto $3\sqrt{\pi}/(8\sigma^{3/2})$ con nota editorial de reconciliación. Inconsistencia cerrada.
