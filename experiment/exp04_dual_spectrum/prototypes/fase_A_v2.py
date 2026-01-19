"""
Fase A v2: Búsqueda del Punto de Degeneración de Borg-Levinson
==============================================================
Versión mejorada con solver más robusto y búsqueda más agresiva.
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

# --- CONSTANTES Y CONFIGURACIÓN ---
Z_MIN, Z_MAX, N_GRID = 0.05, 6.0, 800  # Mayor resolución
Z = np.linspace(Z_MIN, Z_MAX, N_GRID)
DZ = Z[1] - Z[0]
N_MODES = 8  # Más modos para mejor discriminación

def solve_spectrum_tridiag(theta, bc='dirichlet', n_modes=N_MODES):
    """
    Solver robusto usando descomposición tridiagonal directa.
    Más estable que ARPACK para este problema.
    """
    alpha, beta, gamma = theta
    
    # Potencial GK simplificado: V(z) = 3/(4z²) + αz² + βz^γ
    # Regularización en z pequeño para evitar singularidad
    V = (0.75 / (Z**2 + 0.01)) + alpha * Z**2 + beta * np.power(Z, gamma)
    
    # Elementos de la matriz tridiagonal (discretización de -d²/dz² + V)
    diag = 2.0 / DZ**2 + V
    off_diag = -1.0 / DZ**2 * np.ones(N_GRID - 1)
    
    # Modificación para condición de Neumann en IR (último punto)
    if bc == 'neumann':
        # Condición ψ'(z_max) = 0 implica ψ[N] = ψ[N-2] (punto fantasma)
        # Esto modifica el coeficiente off-diagonal final
        off_diag[-1] = -2.0 / DZ**2
    
    # Resolver usando eigh_tridiagonal (más robusto que eigsh)
    try:
        eigenvalues = eigh_tridiagonal(diag, off_diag, eigvals_only=True,
                                        select='i', select_range=(0, n_modes-1))
        return eigenvalues
    except Exception as e:
        print(f"Error en solver: {e}, theta={theta}, bc={bc}")
        return np.ones(n_modes) * 1e6  # Valor de penalización


def test_solver():
    """Verificar que el solver funciona correctamente."""
    theta_test = np.array([1.0, 0.5, 2.5])
    spec_D = solve_spectrum_tridiag(theta_test, bc='dirichlet')
    spec_N = solve_spectrum_tridiag(theta_test, bc='neumann')
    print("Test del solver:")
    print(f"  Theta: {theta_test}")
    print(f"  Espectro Dirichlet: {spec_D}")
    print(f"  Espectro Neumann:   {spec_N}")
    print(f"  Diferencia D-N:     {spec_D - spec_N}")
    return spec_D, spec_N


# --- CONFIGURACIÓN DEL PUNTO DE REFERENCIA ---
THETA_0 = np.array([1.0, 0.5, 2.5])  # (alpha, beta, gamma)

print("="*60)
print("FASE A: Búsqueda del Punto de Degeneración de Borg-Levinson")
print("="*60)

# Test inicial
spec_D_0, spec_N_0 = test_solver()
print(f"\nPunto de referencia Theta_0: {THETA_0}")
print(f"Espectro Dirichlet: {spec_D_0}")
print(f"Espectro Neumann:   {spec_N_0}")


# --- BÚSQUEDA DE DEGENERACIÓN ---
def objective_degeneracy(theta):
    """
    Buscar theta que minimice distancia Dirichlet
    pero que esté lejos de theta_0 en espacio de parámetros.
    """
    # Distancia paramétrica (queremos que sea significativa)
    dist_param = np.linalg.norm(theta - THETA_0)
    
    # Penalizar solución trivial
    if dist_param < 0.3:
        return 1e6
    
    # Calcular espectro Dirichlet
    spec_D = solve_spectrum_tridiag(theta, bc='dirichlet')
    
    # Error espectral (queremos minimizar)
    err_spectral = np.sum((spec_D - spec_D_0)**2)
    
    # Objetivo: minimizar error espectral, bonificar distancia paramétrica
    # Esto busca puntos lejanos en parámetros pero cercanos en espectro
    return err_spectral - 0.01 * dist_param

print("\n" + "-"*60)
print("Buscando punto de degeneración...")
print("-"*60)

# Búsqueda global con evolución diferencial
bounds = [
    (0.3, 2.5),   # alpha
    (0.1, 2.0),   # beta  
    (1.5, 4.0)    # gamma
]

result = differential_evolution(
    objective_degeneracy, 
    bounds, 
    maxiter=500,
    tol=1e-8,
    seed=42,
    workers=1,
    disp=True
)

THETA_1 = result.x

# --- CALCULAR ESPECTROS EN EL PUNTO ENCONTRADO ---
spec_D_1 = solve_spectrum_tridiag(THETA_1, bc='dirichlet')
spec_N_1 = solve_spectrum_tridiag(THETA_1, bc='neumann')

dist_D = np.linalg.norm(spec_D_1 - spec_D_0)
dist_N = np.linalg.norm(spec_N_1 - spec_N_0)
dist_param = np.linalg.norm(THETA_1 - THETA_0)

print("\n" + "="*60)
print("RESULTADOS DE LA BÚSQUEDA")
print("="*60)
print(f"Theta_0 (referencia):  {THETA_0}")
print(f"Theta_1 (encontrado):  {THETA_1}")
print(f"Distancia paramétrica: {dist_param:.6f}")
print(f"\nEspectro Dirichlet Theta_0: {spec_D_0}")
print(f"Espectro Dirichlet Theta_1: {spec_D_1}")
print(f"Distancia Dirichlet:        {dist_D:.8f}")
print(f"\nEspectro Neumann Theta_0:   {spec_N_0}")
print(f"Espectro Neumann Theta_1:   {spec_N_1}")
print(f"Distancia Neumann:          {dist_N:.8f}")

# --- RATIO CLAVE: ¿Neumann discrimina mejor? ---
ratio_N_D = dist_N / (dist_D + 1e-12)
print(f"\n*** RATIO NEUMANN/DIRICHLET: {ratio_N_D:.4f} ***")

if ratio_N_D > 5:
    print(">>> ¡ÉXITO! El espectro Neumann discrimina significativamente mejor.")
    print(">>> Borg-Levinson confirmado: la información dual rompe degeneración.")
elif ratio_N_D > 1.5:
    print(">>> Parcial: Neumann aporta información adicional, pero moderada.")
else:
    print(">>> No se encontró degeneración significativa en esta región.")
    print(">>> Intentar con otros bounds o más iteraciones.")


# --- CÁLCULO DEL IRBL-θ ---
print("\n" + "-"*60)
print("Calculando IRBL-theta en el punto de degeneración...")
print("-"*60)

def compute_jacobian_theta(theta, bc='dirichlet', epsilon=1e-5):
    """Jacobiana del espectro respecto a parámetros θ."""
    n_params = len(theta)
    spec_0 = solve_spectrum_tridiag(theta, bc=bc)
    n_modes = len(spec_0)
    
    J = np.zeros((n_modes, n_params))
    for i in range(n_params):
        perturb = np.zeros(n_params)
        perturb[i] = epsilon
        f_plus = solve_spectrum_tridiag(theta + perturb, bc=bc)
        f_minus = solve_spectrum_tridiag(theta - perturb, bc=bc)
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)
    return J

# Jacobianas en theta_1
J_D = compute_jacobian_theta(THETA_1, bc='dirichlet')
J_N = compute_jacobian_theta(THETA_1, bc='neumann')
J_DN = np.vstack([J_D, J_N])

# Valores singulares
s_D = np.linalg.svd(J_D, compute_uv=False)
s_N = np.linalg.svd(J_N, compute_uv=False)
s_DN = np.linalg.svd(J_DN, compute_uv=False)

print(f"\nValores singulares J_D:  {s_D}")
print(f"Valores singulares J_N:  {s_N}")
print(f"Valores singulares J_DN: {s_DN}")

sigma_min_D = s_D[-1]
sigma_min_DN = s_DN[-1]
IRBL = sigma_min_DN / (sigma_min_D + 1e-12)

print(f"\nσ_min(J_D):  {sigma_min_D:.6f}")
print(f"σ_min(J_DN): {sigma_min_DN:.6f}")
print(f"\n*** IRBL-theta = {IRBL:.4f} ***")

# Número de condición (otra métrica útil)
cond_D = s_D[0] / (s_D[-1] + 1e-12)
cond_DN = s_DN[0] / (s_DN[-1] + 1e-12)
IRBL_cond = cond_D / cond_DN

print(f"\nNúmero de condición J_D:  {cond_D:.2f}")
print(f"Número de condición J_DN: {cond_DN:.2f}")
print(f"IRBL_cond (ratio):        {IRBL_cond:.4f}")


# --- VISUALIZACIÓN ---
print("\n" + "-"*60)
print("Generando visualización...")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Potenciales
ax1 = axes[0, 0]
V_0 = (0.75 / (Z**2 + 0.01)) + THETA_0[0] * Z**2 + THETA_0[1] * np.power(Z, THETA_0[2])
V_1 = (0.75 / (Z**2 + 0.01)) + THETA_1[0] * Z**2 + THETA_1[1] * np.power(Z, THETA_1[2])
ax1.plot(Z, V_0, 'b-', linewidth=2, label=f'V(θ₀) = ({THETA_0[0]:.2f}, {THETA_0[1]:.2f}, {THETA_0[2]:.2f})')
ax1.plot(Z, V_1, 'r--', linewidth=2, label=f'V(θ₁) = ({THETA_1[0]:.2f}, {THETA_1[1]:.2f}, {THETA_1[2]:.2f})')
ax1.set_xlabel('z (coordenada radial)')
ax1.set_ylabel('V(z)')
ax1.set_title('Potenciales GK')
ax1.set_ylim(0, 50)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Espectros Dirichlet
ax2 = axes[0, 1]
modes = np.arange(N_MODES)
width = 0.35
ax2.bar(modes - width/2, spec_D_0, width, label='θ₀ Dirichlet', color='blue', alpha=0.7)
ax2.bar(modes + width/2, spec_D_1, width, label='θ₁ Dirichlet', color='red', alpha=0.7)
ax2.set_xlabel('Modo n')
ax2.set_ylabel('λₙ')
ax2.set_title(f'Espectros Dirichlet (dist = {dist_D:.6f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Espectros Neumann
ax3 = axes[1, 0]
ax3.bar(modes - width/2, spec_N_0, width, label='θ₀ Neumann', color='blue', alpha=0.7)
ax3.bar(modes + width/2, spec_N_1, width, label='θ₁ Neumann', color='red', alpha=0.7)
ax3.set_xlabel('Modo n')
ax3.set_ylabel('λₙ')
ax3.set_title(f'Espectros Neumann (dist = {dist_N:.6f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Resumen IRBL
ax4 = axes[1, 1]
metrics = ['σ_min(D)', 'σ_min(D+N)', 'IRBL', 'Ratio N/D']
values = [sigma_min_D, sigma_min_DN, IRBL, ratio_N_D]
colors = ['blue', 'green', 'purple', 'orange']
bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
ax4.set_ylabel('Valor')
ax4.set_title('Métricas de Resolvabilidad')
ax4.set_yscale('log')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{val:.2f}', ha='center', va='bottom', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/fase_A_results.png', dpi=150, bbox_inches='tight')
print("Figura guardada en: /home/claude/fase_A_results.png")

# --- RESUMEN FINAL ---
print("\n" + "="*60)
print("RESUMEN EJECUTIVO")
print("="*60)
print(f"""
Punto de referencia θ₀: α={THETA_0[0]:.3f}, β={THETA_0[1]:.3f}, γ={THETA_0[2]:.3f}
Punto encontrado   θ₁: α={THETA_1[0]:.3f}, β={THETA_1[1]:.3f}, γ={THETA_1[2]:.3f}

DISTANCIAS:
  - Paramétrica (||θ₁-θ₀||):     {dist_param:.4f}
  - Espectral Dirichlet:         {dist_D:.8f}
  - Espectral Neumann:           {dist_N:.8f}
  - Ratio Neumann/Dirichlet:     {ratio_N_D:.4f}

ÍNDICE DE RESOLVABILIDAD (IRBL):
  - σ_min(J_Dirichlet):          {sigma_min_D:.6f}
  - σ_min(J_Dual):               {sigma_min_DN:.6f}
  - IRBL = σ_DN / σ_D:           {IRBL:.4f}

VEREDICTO:
""")

if IRBL > 10:
    print("  ✓ DEGENERACIÓN FUERTE DETECTADA")
    print("  ✓ Borg-Levinson CONFIRMADO: el dual spectrum es necesario")
    print("  ✓ Sin información de Neumann, la geometría NO es identificable")
elif IRBL > 2:
    print("  ~ Degeneración moderada detectada")
    print("  ~ El dual spectrum mejora la identificabilidad")
else:
    print("  ✗ No se detectó degeneración significativa")
    print("  ✗ El problema parece ser inyectivo con solo Dirichlet en esta región")

print("="*60)
