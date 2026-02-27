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

Sin embargo, en A.5 está escrito:

\[
\mathcal{J}_0(\sigma) = \frac{\pi}{2\sigma}+O(\sigma^{-2}),
\]

lo cual no es consistente con (A.4).

### Propuesta de fix al texto

1. Mantener (A.4) como definición normativa.
2. Corregir A.5 para reflejar el asintótico derivado de (A.4), o marcar explícitamente si A.5 se refiere a otra cantidad/reescalamiento distinto.
3. Añadir nota breve de reconciliación entre §6.5 y Apéndice A para evitar ambigüedad en implementación numérica.
