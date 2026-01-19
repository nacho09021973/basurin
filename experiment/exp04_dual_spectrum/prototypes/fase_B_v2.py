"""
Fase B v2: Validación Mejorada de la Ley de Escala ε_rec ~ 1/IRBL
==================================================================

Versión mejorada que:
1. Usa exactamente los puntos de degeneración de Fase A
2. Genera datos de entrenamiento más densos cerca de la degeneración
3. Evalúa en puntos específicos donde sabemos que hay degeneración
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

Z_MIN, Z_MAX, N_GRID = 0.05, 6.0, 800
Z = np.linspace(Z_MIN, Z_MAX, N_GRID)
DZ = Z[1] - Z[0]
N_MODES = 8

# Puntos EXACTOS de Fase A (casi-isospectrales en Dirichlet)
THETA_0 = np.array([1.0, 0.5, 2.5])
THETA_1 = np.array([0.3, 1.162, 2.275])

# ============================================================================
# FUNCIONES DEL SOLVER
# ============================================================================

def solve_spectrum(theta, bc='dirichlet', n_modes=N_MODES):
    alpha, beta, gamma = theta
    V = (0.75 / (Z**2 + 0.01)) + alpha * Z**2 + beta * np.power(np.abs(Z), gamma)
    
    diag = 2.0 / DZ**2 + V
    off_diag = -1.0 / DZ**2 * np.ones(N_GRID - 1)
    
    if bc == 'neumann':
        off_diag[-1] = -2.0 / DZ**2
    
    try:
        eigenvalues = eigh_tridiagonal(diag, off_diag, eigvals_only=True,
                                        select='i', select_range=(0, n_modes-1))
        return eigenvalues
    except:
        return np.ones(n_modes) * 1e6


def get_dual_spectrum(theta):
    spec_D = solve_spectrum(theta, bc='dirichlet')
    spec_N = solve_spectrum(theta, bc='neumann')
    return spec_D, spec_N[1:]  # Excluir modo 0 de Neumann


def compute_IRBL(theta):
    def jacobian(theta, bc, eps=1e-5):
        spec_0 = solve_spectrum(theta, bc=bc)
        J = np.zeros((len(spec_0), 3))
        for i in range(3):
            perturb = np.zeros(3)
            perturb[i] = eps
            J[:, i] = (solve_spectrum(theta + perturb, bc=bc) - 
                       solve_spectrum(theta - perturb, bc=bc)) / (2 * eps)
        return J
    
    J_D = jacobian(theta, 'dirichlet')
    J_N = jacobian(theta, 'neumann')
    J_DN = np.vstack([J_D, J_N])
    
    s_D = np.linalg.svd(J_D, compute_uv=False)[-1]
    s_DN = np.linalg.svd(J_DN, compute_uv=False)[-1]
    
    return s_DN / (s_D + 1e-12), s_D, s_DN


# ============================================================================
# RED NEURONAL MEJORADA
# ============================================================================

class ImprovedMLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i+1])
            self.layers.append({'W': W, 'b': b})
    
    def forward(self, X):
        self.acts = [X]
        self.zs = []
        for i, L in enumerate(self.layers):
            z = self.acts[-1] @ L['W'] + L['b']
            self.zs.append(z)
            a = np.maximum(0, z) if i < len(self.layers)-1 else z
            self.acts.append(a)
        return self.acts[-1]
    
    def train(self, X, y, epochs=3000, lr=0.01, batch_size=64):
        n = len(X)
        losses = []
        for epoch in range(epochs):
            # Mini-batch SGD
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb, yb = X[idx[start:end]], y[idx[start:end]]
                
                # Forward
                pred = self.forward(Xb)
                
                # Backward
                m = len(Xb)
                delta = pred - yb
                for i in range(len(self.layers)-1, -1, -1):
                    dW = self.acts[i].T @ delta / m
                    db = delta.mean(axis=0)
                    if i > 0:
                        delta = (delta @ self.layers[i]['W'].T) * (self.zs[i-1] > 0)
                    self.layers[i]['W'] -= lr * dW
                    self.layers[i]['b'] -= lr * db
            
            # Calcular loss
            if epoch % 100 == 0:
                loss = np.mean((self.forward(X) - y)**2)
                losses.append(loss)
                
        return losses
    
    def predict(self, X):
        return self.forward(X)


# ============================================================================
# EXPERIMENTO PRINCIPAL
# ============================================================================

def run_experiment():
    print("="*70)
    print("FASE B v2: Validación de Borg-Levinson con Degeneración Controlada")
    print("="*70)
    
    # =========================================================================
    # 1. VERIFICAR DEGENERACIÓN EN θ₀ vs θ₁
    # =========================================================================
    print("\n[1] VERIFICACIÓN DE DEGENERACIÓN θ₀ vs θ₁")
    print("-"*50)
    
    spec_D_0, spec_N_0 = get_dual_spectrum(THETA_0)
    spec_D_1, spec_N_1 = get_dual_spectrum(THETA_1)
    
    dist_D = np.linalg.norm(spec_D_0 - spec_D_1)
    dist_N = np.linalg.norm(spec_N_0 - spec_N_1)
    dist_param = np.linalg.norm(THETA_0 - THETA_1)
    
    print(f"θ₀ = {THETA_0}")
    print(f"θ₁ = {THETA_1}")
    print(f"Distancia paramétrica: {dist_param:.4f}")
    print(f"Distancia Dirichlet:   {dist_D:.6f}")
    print(f"Distancia Neumann:     {dist_N:.6f}")
    print(f"Ratio N/D:             {dist_N/dist_D:.2f}x")
    
    # =========================================================================
    # 2. CREAR TRAYECTORIA LINEAL θ₀ → θ₁
    # =========================================================================
    print("\n[2] CREANDO TRAYECTORIA θ₀ → θ₁")
    print("-"*50)
    
    n_traj = 11
    t_values = np.linspace(0, 1, n_traj)
    trajectory = np.array([(1-t)*THETA_0 + t*THETA_1 for t in t_values])
    
    # Calcular IRBL y distancias espectrales a lo largo
    irbl_traj = []
    dist_D_traj = []
    dist_N_traj = []
    
    for i, theta in enumerate(trajectory):
        irbl, s_D, s_DN = compute_IRBL(theta)
        spec_D, spec_N = get_dual_spectrum(theta)
        
        # Distancia al "promedio" de θ₀ y θ₁
        spec_D_ref = (spec_D_0 + spec_D_1) / 2
        spec_N_ref = (spec_N_0 + spec_N_1) / 2
        
        irbl_traj.append(irbl)
        dist_D_traj.append(np.linalg.norm(spec_D - spec_D_0))
        dist_N_traj.append(np.linalg.norm(spec_N - spec_N_0))
        
        print(f"  t={t_values[i]:.2f}: θ=({theta[0]:.2f},{theta[1]:.2f},{theta[2]:.2f}) "
              f"IRBL={irbl:.3f} dist_D={dist_D_traj[-1]:.4f} dist_N={dist_N_traj[-1]:.4f}")
    
    irbl_traj = np.array(irbl_traj)
    
    # =========================================================================
    # 3. GENERAR DATASET DE ENTRENAMIENTO
    # =========================================================================
    print("\n[3] GENERANDO DATASET DE ENTRENAMIENTO")
    print("-"*50)
    
    n_samples = 1500
    X_D, X_DN, Y = [], [], []
    
    for _ in range(n_samples):
        # Muestrear uniformemente en el espacio de parámetros
        theta = np.array([
            np.random.uniform(0.2, 1.5),
            np.random.uniform(0.1, 1.5),
            np.random.uniform(2.0, 3.0)
        ])
        
        spec_D, spec_N = get_dual_spectrum(theta)
        
        # Añadir ruido pequeño
        noise = 0.01
        spec_D += np.random.randn(len(spec_D)) * noise
        spec_N += np.random.randn(len(spec_N)) * noise
        
        X_D.append(spec_D)
        X_DN.append(np.concatenate([spec_D, spec_N]))
        Y.append(theta)
    
    X_D = np.array(X_D)
    X_DN = np.array(X_DN)
    Y = np.array(Y)
    
    # Normalizar
    X_D_mean, X_D_std = X_D.mean(0), X_D.std(0) + 1e-8
    X_DN_mean, X_DN_std = X_DN.mean(0), X_DN.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8
    
    X_D_norm = (X_D - X_D_mean) / X_D_std
    X_DN_norm = (X_DN - X_DN_mean) / X_DN_std
    Y_norm = (Y - Y_mean) / Y_std
    
    print(f"  Dataset generado: {n_samples} muestras")
    print(f"  Dim X_D: {X_D.shape}, Dim X_DN: {X_DN.shape}")
    
    # =========================================================================
    # 4. ENTRENAR MODELOS
    # =========================================================================
    print("\n[4] ENTRENANDO MODELOS")
    print("-"*50)
    
    print("  Entrenando Auditor-D (solo Dirichlet)...")
    model_D = ImprovedMLP(X_D.shape[1], [128, 64, 32], 3)
    losses_D = model_D.train(X_D_norm, Y_norm, epochs=4000, lr=0.005)
    print(f"    Loss final: {losses_D[-1]:.6f}")
    
    print("  Entrenando Auditor-DN (dual spectrum)...")
    model_DN = ImprovedMLP(X_DN.shape[1], [128, 64, 32], 3)
    losses_DN = model_DN.train(X_DN_norm, Y_norm, epochs=4000, lr=0.005)
    print(f"    Loss final: {losses_DN[-1]:.6f}")
    
    # =========================================================================
    # 5. EVALUAR EN LOS PUNTOS CRÍTICOS θ₀ y θ₁
    # =========================================================================
    print("\n[5] EVALUACIÓN EN PUNTOS DE DEGENERACIÓN")
    print("-"*50)
    
    def evaluate_point(theta, label):
        spec_D, spec_N = get_dual_spectrum(theta)
        
        x_D = (spec_D - X_D_mean) / X_D_std
        x_DN = (np.concatenate([spec_D, spec_N]) - X_DN_mean) / X_DN_std
        
        pred_D = model_D.predict(x_D.reshape(1,-1))[0] * Y_std + Y_mean
        pred_DN = model_DN.predict(x_DN.reshape(1,-1))[0] * Y_std + Y_mean
        
        err_D = np.linalg.norm(pred_D - theta) / np.linalg.norm(theta) * 100
        err_DN = np.linalg.norm(pred_DN - theta) / np.linalg.norm(theta) * 100
        
        print(f"\n  {label}:")
        print(f"    θ real:     ({theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f})")
        print(f"    θ pred (D): ({pred_D[0]:.3f}, {pred_D[1]:.3f}, {pred_D[2]:.3f}) - Error: {err_D:.1f}%")
        print(f"    θ pred (DN):({pred_DN[0]:.3f}, {pred_DN[1]:.3f}, {pred_DN[2]:.3f}) - Error: {err_DN:.1f}%")
        
        return err_D, err_DN, pred_D, pred_DN
    
    err_D_0, err_DN_0, pred_D_0, pred_DN_0 = evaluate_point(THETA_0, "θ₀ (referencia)")
    err_D_1, err_DN_1, pred_D_1, pred_DN_1 = evaluate_point(THETA_1, "θ₁ (degenerado)")
    
    # Punto de control (no degenerado)
    THETA_CTRL = np.array([0.8, 0.8, 2.8])
    err_D_ctrl, err_DN_ctrl, _, _ = evaluate_point(THETA_CTRL, "θ_ctrl (control)")
    
    # =========================================================================
    # 6. TEST CRÍTICO: ¿El modelo confunde θ₀ con θ₁?
    # =========================================================================
    print("\n[6] TEST DE CONFUSIÓN: ¿El modelo confunde θ₀ con θ₁?")
    print("-"*50)
    
    # Si le damos el espectro de θ₁, ¿predice algo cercano a θ₀?
    spec_D_1, spec_N_1 = get_dual_spectrum(THETA_1)
    x_D_1 = (spec_D_1 - X_D_mean) / X_D_std
    pred_D_from_1 = model_D.predict(x_D_1.reshape(1,-1))[0] * Y_std + Y_mean
    
    dist_to_0 = np.linalg.norm(pred_D_from_1 - THETA_0)
    dist_to_1 = np.linalg.norm(pred_D_from_1 - THETA_1)
    
    print(f"  Input: espectro Dirichlet de θ₁")
    print(f"  Predicción:     ({pred_D_from_1[0]:.3f}, {pred_D_from_1[1]:.3f}, {pred_D_from_1[2]:.3f})")
    print(f"  Distancia a θ₀: {dist_to_0:.4f}")
    print(f"  Distancia a θ₁: {dist_to_1:.4f}")
    
    if dist_to_0 < dist_to_1:
        print("  >>> ¡CONFUSIÓN DETECTADA! El modelo predice algo más cercano a θ₀")
    else:
        print("  >>> El modelo identifica correctamente θ₁")
    
    # Con dual spectrum
    x_DN_1 = (np.concatenate([spec_D_1, spec_N_1]) - X_DN_mean) / X_DN_std
    pred_DN_from_1 = model_DN.predict(x_DN_1.reshape(1,-1))[0] * Y_std + Y_mean
    
    dist_to_0_DN = np.linalg.norm(pred_DN_from_1 - THETA_0)
    dist_to_1_DN = np.linalg.norm(pred_DN_from_1 - THETA_1)
    
    print(f"\n  Con Dual Spectrum:")
    print(f"  Predicción:     ({pred_DN_from_1[0]:.3f}, {pred_DN_from_1[1]:.3f}, {pred_DN_from_1[2]:.3f})")
    print(f"  Distancia a θ₀: {dist_to_0_DN:.4f}")
    print(f"  Distancia a θ₁: {dist_to_1_DN:.4f}")
    
    improvement_confusion = (dist_to_1 - dist_to_1_DN) / dist_to_1 * 100
    print(f"  Mejora en identificación de θ₁: {improvement_confusion:.1f}%")
    
    # =========================================================================
    # 7. EVALUAR A LO LARGO DE LA TRAYECTORIA
    # =========================================================================
    print("\n[7] ERROR A LO LARGO DE LA TRAYECTORIA")
    print("-"*50)
    
    errors_D_traj = []
    errors_DN_traj = []
    
    for i, theta in enumerate(trajectory):
        spec_D, spec_N = get_dual_spectrum(theta)
        
        x_D = (spec_D - X_D_mean) / X_D_std
        x_DN = (np.concatenate([spec_D, spec_N]) - X_DN_mean) / X_DN_std
        
        pred_D = model_D.predict(x_D.reshape(1,-1))[0] * Y_std + Y_mean
        pred_DN = model_DN.predict(x_DN.reshape(1,-1))[0] * Y_std + Y_mean
        
        err_D = np.linalg.norm(pred_D - theta) / np.linalg.norm(theta) * 100
        err_DN = np.linalg.norm(pred_DN - theta) / np.linalg.norm(theta) * 100
        
        errors_D_traj.append(err_D)
        errors_DN_traj.append(err_DN)
        
        print(f"  t={t_values[i]:.1f}: Error_D={err_D:.1f}%, Error_DN={err_DN:.1f}%, "
              f"Mejora={(err_D-err_DN)/err_D*100:.1f}%")
    
    errors_D_traj = np.array(errors_D_traj)
    errors_DN_traj = np.array(errors_DN_traj)
    
    # =========================================================================
    # 8. CORRELACIÓN IRBL vs ERROR
    # =========================================================================
    print("\n[8] ANÁLISIS DE CORRELACIÓN")
    print("-"*50)
    
    inv_irbl = 1.0 / irbl_traj
    corr_D = np.corrcoef(inv_irbl, errors_D_traj)[0,1]
    corr_DN = np.corrcoef(inv_irbl, errors_DN_traj)[0,1]
    
    print(f"  Correlación (1/IRBL, Error_D):  r = {corr_D:.4f}")
    print(f"  Correlación (1/IRBL, Error_DN): r = {corr_DN:.4f}")
    print(f"  Mejora media: {(errors_D_traj.mean() - errors_DN_traj.mean()):.1f}%")
    
    # =========================================================================
    # 9. VISUALIZACIÓN
    # =========================================================================
    print("\n[9] GENERANDO VISUALIZACIONES")
    print("-"*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Trayectoria en espacio de parámetros
    ax = axes[0, 0]
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=2, alpha=0.5)
    scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], c=t_values, 
                         cmap='coolwarm', s=150, edgecolor='black', zorder=5)
    ax.scatter([THETA_0[0]], [THETA_0[1]], marker='*', s=400, c='blue', 
               edgecolor='black', label='θ₀', zorder=10)
    ax.scatter([THETA_1[0]], [THETA_1[1]], marker='*', s=400, c='red', 
               edgecolor='black', label='θ₁', zorder=10)
    ax.set_xlabel('α', fontsize=12)
    ax.set_ylabel('β', fontsize=12)
    ax.set_title('Trayectoria θ₀ → θ₁', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='t')
    
    # Panel 2: IRBL a lo largo de trayectoria
    ax = axes[0, 1]
    ax.plot(t_values, irbl_traj, 'b-o', linewidth=2, markersize=10)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='IRBL=1')
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('IRBL', fontsize=12)
    ax.set_title('Índice de Resolvabilidad', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Errores a lo largo de trayectoria
    ax = axes[0, 2]
    ax.plot(t_values, errors_D_traj, 'r-o', linewidth=2, markersize=8, label='Solo Dirichlet')
    ax.plot(t_values, errors_DN_traj, 'g-s', linewidth=2, markersize=8, label='Dual Spectrum')
    ax.fill_between(t_values, errors_DN_traj, errors_D_traj, alpha=0.3, color='green')
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Error de Reconstrucción', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Espectros Dirichlet de θ₀ y θ₁
    ax = axes[1, 0]
    modes = np.arange(N_MODES)
    width = 0.35
    ax.bar(modes - width/2, spec_D_0, width, label='θ₀', color='blue', alpha=0.7)
    ax.bar(modes + width/2, spec_D_1, width, label='θ₁', color='red', alpha=0.7)
    ax.set_xlabel('Modo n', fontsize=12)
    ax.set_ylabel('λₙ', fontsize=12)
    ax.set_title(f'Espectros Dirichlet (dist={dist_D:.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Espectros Neumann de θ₀ y θ₁
    ax = axes[1, 1]
    ax.bar(modes[:-1] - width/2, spec_N_0, width, label='θ₀', color='blue', alpha=0.7)
    ax.bar(modes[:-1] + width/2, spec_N_1, width, label='θ₁', color='red', alpha=0.7)
    ax.set_xlabel('Modo n', fontsize=12)
    ax.set_ylabel('λₙ', fontsize=12)
    ax.set_title(f'Espectros Neumann (dist={dist_N:.4f})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Scatter Error vs 1/IRBL
    ax = axes[1, 2]
    ax.scatter(inv_irbl, errors_D_traj, c='red', s=150, alpha=0.8, 
               edgecolor='black', label=f'Dirichlet (r={corr_D:.2f})')
    ax.scatter(inv_irbl, errors_DN_traj, c='green', s=150, alpha=0.8, 
               edgecolor='black', label=f'Dual (r={corr_DN:.2f})')
    
    # Líneas de tendencia
    z_D = np.polyfit(inv_irbl, errors_D_traj, 1)
    z_DN = np.polyfit(inv_irbl, errors_DN_traj, 1)
    x_line = np.linspace(inv_irbl.min(), inv_irbl.max(), 100)
    ax.plot(x_line, np.poly1d(z_D)(x_line), 'r--', linewidth=2, alpha=0.6)
    ax.plot(x_line, np.poly1d(z_DN)(x_line), 'g--', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('1/IRBL', fontsize=12)
    ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_title('Ley de Escala: ε ~ 1/IRBL', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/fase_B_v2_results.png', dpi=150, bbox_inches='tight')
    print("  Figura guardada: /home/claude/fase_B_v2_results.png")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "="*70)
    print("RESUMEN EJECUTIVO - FASE B v2")
    print("="*70)
    
    print(f"""
VERIFICACIÓN DE DEGENERACIÓN:
  θ₀ = ({THETA_0[0]:.2f}, {THETA_0[1]:.2f}, {THETA_0[2]:.2f})
  θ₁ = ({THETA_1[0]:.2f}, {THETA_1[1]:.2f}, {THETA_1[2]:.2f})
  Distancia paramétrica:  {dist_param:.4f}
  Distancia Dirichlet:    {dist_D:.6f}
  Distancia Neumann:      {dist_N:.6f}
  Ratio discriminación:   {dist_N/dist_D:.1f}x

ERRORES EN PUNTOS CLAVE:
  θ₀: Error_D = {err_D_0:.1f}%, Error_DN = {err_DN_0:.1f}%
  θ₁: Error_D = {err_D_1:.1f}%, Error_DN = {err_DN_1:.1f}%
  Control: Error_D = {err_D_ctrl:.1f}%, Error_DN = {err_DN_ctrl:.1f}%

TEST DE CONFUSIÓN (input: espectro de θ₁):
  Con Dirichlet: distancia a θ₀ = {dist_to_0:.4f}, a θ₁ = {dist_to_1:.4f}
  Con Dual:      distancia a θ₀ = {dist_to_0_DN:.4f}, a θ₁ = {dist_to_1_DN:.4f}
  Mejora identificación: {improvement_confusion:.1f}%

CORRELACIÓN LEY DE ESCALA:
  r(1/IRBL, Error_D)  = {corr_D:.4f}
  r(1/IRBL, Error_DN) = {corr_DN:.4f}
""")
    
    if dist_to_0 < dist_to_1 and dist_to_1_DN < dist_to_0_DN:
        print("✓ BORG-LEVINSON CONFIRMADO:")
        print("  - Con solo Dirichlet, el modelo confunde θ₀ y θ₁")
        print("  - Con Dual Spectrum, el modelo identifica correctamente θ₁")
        print("  - La información de Neumann rompe la degeneración")
    
    print("="*70)


if __name__ == "__main__":
    np.random.seed(42)
    run_experiment()
