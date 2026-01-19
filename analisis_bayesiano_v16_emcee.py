#!/usr/bin/env python3
"""
analisis_bayesiano_v16_emcee.py
================================
ANÁLISIS BAYESIANO DE RINGDOWN GW150914 - VERSIÓN FUNCIONAL

Este script funciona. Los problemas de la v14 eran:
1. Datos L1 con DC offset masivo (todos valores negativos)
2. Prior de t0 no cubría donde estaba el pico real
3. Dynesty 3.0.0 tiene bugs con rwalk cuando los puntos iniciales son rechazados

Solución:
- Preprocesamiento robusto con remoción de DC offset
- Prior de t0 centrado en el pico detectado
- Usar emcee (MCMC) en lugar de dynesty (nested sampling)

Requisitos:
    pip install numpy scipy matplotlib emcee corner

Autor: José Ignacio Martín Gandul
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import emcee
import corner
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================
C = 299792458.0          # m/s
G = 6.67430e-11          # m³/kg/s²
MSUN = 1.98847e30        # kg
Z_GW150914 = 0.09        # Redshift

# Coeficientes de Berti para modo (l,m,n) = (2,2,0)
BERTI_F = (1.5251, -1.1568, 0.1292)
BERTI_Q = (0.7000, 1.4187, -0.4990)

# ============================================================================
# FÍSICA DEL RINGDOWN
# ============================================================================
def get_kerr_qnm(mass_msun, spin):
    """
    Calcula frecuencia y tiempo de decaimiento del modo QNM dominante (220)
    usando los ajustes de Berti et al.
    
    Args:
        mass_msun: Masa del agujero negro final [Msun]
        spin: Parámetro de spin adimensional (0 < a < 1)
    
    Returns:
        f_hz: Frecuencia del QNM [Hz]
        tau_s: Tiempo de decaimiento [s]
    """
    spin = float(np.clip(spin, 0.01, 0.99))
    
    # Frecuencia adimensional
    w_bar = BERTI_F[0] + BERTI_F[1] * np.power(1 - spin, BERTI_F[2])
    
    # Factor de calidad
    Q = BERTI_Q[0] + BERTI_Q[1] * np.power(1 - spin, BERTI_Q[2])
    
    # Tiempo gravitacional
    Tg = (G * mass_msun * MSUN) / C**3
    
    # Convertir a Hz y segundos
    f_hz = w_bar / (2 * np.pi * Tg)
    tau_s = Q / (np.pi * f_hz)
    
    return f_hz, max(tau_s, 1e-6)


def ringdown_model(time, mass, spin, amplitude, phase, t0):
    """
    Modelo de ringdown: damped sinusoid.
    
    h(t) = A * exp(-(t-t0)/τ) * cos(2πf(t-t0) + φ)  para t >= t0
    h(t) = 0                                         para t < t0
    """
    f, tau = get_kerr_qnm(mass, spin)
    
    output = np.zeros_like(time)
    mask = time >= t0
    
    if np.sum(mask) == 0:
        return output
    
    dt = time[mask] - t0
    damping = np.exp(np.clip(-dt / tau, -50, 0))
    oscillation = np.cos(2 * np.pi * f * dt + phase)
    output[mask] = amplitude * damping * oscillation
    
    return output


# ============================================================================
# PREPROCESAMIENTO DE DATOS
# ============================================================================
def preprocess_ligo_data(raw_data, fs=4096.0, fmin=20, fmax=500):
    """
    Preprocesa datos raw de LIGO:
    1. Remueve DC offset
    2. Whitening espectral
    3. Filtrado en banda [fmin, fmax]
    4. Extrae ventana centrada en el pico
    
    Returns:
        time: Array de tiempo centrado en t=0 (pico)
        data: Strain whitened y normalizado
        info: Diccionario con información diagnóstica
    """
    N = len(raw_data)
    
    # 1. Remover DC offset (crítico para L1 de GW150914)
    dc_offset = raw_data.mean()
    data_fixed = raw_data - dc_offset
    
    # 2. Aplicar ventana de Tukey para evitar artefactos
    window = tukey(N, alpha=0.1)
    data_windowed = data_fixed * window
    
    # 3. Calcular PSD para whitening
    nperseg = int(4 * fs)
    noverlap = int(nperseg * 0.5)
    freqs, psd = welch(data_windowed, fs, nperseg=nperseg, noverlap=noverlap)
    psd = np.maximum(psd, 1e-50)
    
    # 4. Interpolar PSD
    interp_psd = interp1d(freqs, psd, kind='linear', fill_value="extrapolate")
    
    # 5. FFT y whitening
    freqs_fft = np.fft.rfftfreq(N, 1/fs)
    data_fft = np.fft.rfft(data_windowed)
    
    psd_fft = np.maximum(interp_psd(freqs_fft), 1e-50)
    whitened_fft = data_fft / np.sqrt(psd_fft)
    
    # 6. Filtrado en banda
    mask = (freqs_fft >= fmin) & (freqs_fft <= fmax)
    whitened_fft[~mask] = 0
    
    whitened = np.fft.irfft(whitened_fft, n=N)
    
    # 7. Encontrar pico (evento)
    center = N // 2
    search_window = int(1 * fs)  # ±1 segundo
    search_region = whitened[center-search_window:center+search_window]
    peak_idx_local = np.argmax(np.abs(search_region))
    global_peak = center - search_window + peak_idx_local
    
    # 8. Extraer ventana de ringdown
    pre_peak = int(0.005 * fs)   # 5ms antes
    post_peak = int(0.050 * fs)  # 50ms después
    
    rd_start = global_peak - pre_peak
    rd_end = global_peak + post_peak
    
    segment = whitened[rd_start:rd_end]
    time = (np.arange(len(segment)) - pre_peak) / fs
    
    # 9. Normalizar (sigma = 1)
    sigma = np.std(segment)
    segment_norm = segment / sigma
    
    info = {
        'dc_offset': dc_offset,
        'sigma': sigma,
        'snr_peak': np.max(np.abs(segment_norm)),
        'duration_ms': len(segment) / fs * 1000,
        'global_peak_idx': global_peak
    }
    
    return time, segment_norm, info


# ============================================================================
# INFERENCIA BAYESIANA
# ============================================================================
class RingdownBayesian:
    """Clase para análisis Bayesiano del ringdown."""
    
    def __init__(self, time, data, bounds=None):
        self.time = np.asarray(time, dtype=np.float64)
        self.data = np.asarray(data, dtype=np.float64)
        
        # Priors por defecto (calibrados para GW150914)
        if bounds is None:
            self.bounds = np.array([
                [55, 90],       # mass [Msun]
                [0.4, 0.9],     # spin
                [1, 15],        # amplitude
                [0, 2*np.pi],   # phase
                [-0.005, 0.005] # t0 [s]
            ])
        else:
            self.bounds = np.asarray(bounds)
        
        self.ndim = 5
        self.param_names = ['mass', 'spin', 'amplitude', 'phase', 't0']
    
    def log_prior(self, theta):
        """Prior uniforme."""
        for i, (low, high) in enumerate(self.bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        return 0.0
    
    def log_likelihood(self, theta):
        """Gaussian likelihood (sigma=1 por normalización)."""
        model = ringdown_model(self.time, *theta)
        chi2 = np.sum((self.data - model)**2)
        return -0.5 * chi2
    
    def log_probability(self, theta):
        """Posterior = prior × likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    
    def find_map(self, n_starts=20):
        """Encuentra el Maximum A Posteriori con múltiples inicios."""
        def neg_log_prob(theta):
            return -self.log_probability(theta)
        
        best_result = None
        best_val = np.inf
        
        np.random.seed(42)
        for _ in range(n_starts):
            x0 = np.array([
                np.random.uniform(*self.bounds[i]) 
                for i in range(self.ndim)
            ])
            try:
                result = minimize(neg_log_prob, x0, method='Nelder-Mead',
                                options={'maxiter': 1000})
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result.x
            except:
                pass
        
        return best_result
    
    def run_mcmc(self, nwalkers=32, nsteps=2000, burnin=500, progress=True):
        """Ejecuta MCMC con emcee."""
        # Encontrar MAP para inicializar
        map_estimate = self.find_map()
        
        if map_estimate is None:
            raise ValueError("No se pudo encontrar MAP inicial")
        
        # Inicializar walkers cerca del MAP
        pos = map_estimate + 1e-3 * np.random.randn(nwalkers, self.ndim)
        
        # Asegurar que están dentro de los bounds
        for i in range(nwalkers):
            for j in range(self.ndim):
                pos[i, j] = np.clip(pos[i, j], 
                                   self.bounds[j, 0] + 1e-6, 
                                   self.bounds[j, 1] - 1e-6)
        
        # Ejecutar MCMC
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, 
                                        self.log_probability)
        sampler.run_mcmc(pos, nsteps, progress=progress)
        
        # Extraer muestras post-burnin
        samples = sampler.get_chain(discard=burnin, flat=True)
        
        return {
            'samples': samples,
            'chain': sampler.get_chain(),
            'map': map_estimate,
            'acceptance_fraction': np.mean(sampler.acceptance_fraction)
        }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("ANÁLISIS BAYESIANO DE RINGDOWN GW150914")
    print("="*60)
    
    # Buscar datos
    paths_to_try = [
        'runs/GW150914/raw_data/l1.npy',
        '/mnt/user-data/uploads/l1.npy',
        'l1.npy'
    ]
    
    l1_path = None
    for path in paths_to_try:
        if os.path.exists(path):
            l1_path = path
            break
    
    if l1_path is None:
        print("❌ No se encontraron datos L1")
        print("   Coloca el archivo l1.npy en el directorio actual")
        return
    
    print(f"\n📂 Cargando: {l1_path}")
    raw_data = np.load(l1_path)
    
    # Preprocesar
    print("\n🔧 Preprocesando datos...")
    time, data, info = preprocess_ligo_data(raw_data)
    
    print(f"   DC offset removido: {info['dc_offset']:.2e}")
    print(f"   Duración segmento: {info['duration_ms']:.1f} ms")
    print(f"   SNR pico: {info['snr_peak']:.1f}")
    
    # Análisis Bayesiano
    print("\n🎲 Ejecutando análisis Bayesiano...")
    bayesian = RingdownBayesian(time, data)
    
    print("   Buscando MAP inicial...")
    results = bayesian.run_mcmc(nwalkers=32, nsteps=2000, burnin=500)
    
    samples = results['samples']
    
    # Extraer posteriors
    mass_samples = samples[:, 0]
    spin_samples = samples[:, 1]
    amp_samples = samples[:, 2]
    phase_samples = samples[:, 3]
    t0_samples = samples[:, 4]
    
    # Estadísticas
    m_det = np.median(mass_samples)
    m_det_err = np.std(mass_samples)
    m_src = m_det / (1 + Z_GW150914)
    m_src_err = m_det_err / (1 + Z_GW150914)
    
    a = np.median(spin_samples)
    a_err = np.std(spin_samples)
    
    t0_med = np.median(t0_samples) * 1000
    t0_err = np.std(t0_samples) * 1000
    
    f_qnm, tau_qnm = get_kerr_qnm(m_det, a)
    
    # Resultados
    print("\n" + "="*60)
    print("🏆 RESULTADOS")
    print("="*60)
    print(f"   Masa detector:  {m_det:.1f} ± {m_det_err:.1f} Msun")
    print(f"   Masa fuente:    {m_src:.1f} ± {m_src_err:.1f} Msun")
    print(f"   Spin:           {a:.3f} ± {a_err:.3f}")
    print(f"   t0:             {t0_med:.2f} ± {t0_err:.2f} ms")
    print(f"\n   f_QNM (220):    {f_qnm:.1f} Hz")
    print(f"   τ_QNM:          {tau_qnm*1000:.2f} ms")
    print(f"\n📚 Referencia LIGO: M_src ~ 62 Msun, a ~ 0.67")
    
    # Plots
    outdir = 'out_bayesiano_v16'
    os.makedirs(outdir, exist_ok=True)
    
    print(f"\n📊 Generando plots en {outdir}/...")
    
    # Corner plot
    labels = [r'$M_{det}$ [M$_\odot$]', r'$a$', r'$A$', r'$\phi$', r'$t_0$ [s]']
    fig = corner.corner(samples, labels=labels, 
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_fmt='.2f')
    fig.savefig(f'{outdir}/corner.png', dpi=150)
    plt.close()
    
    # Fit plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    model_map = ringdown_model(time, *results['map'])
    
    axes[0].plot(time*1000, data, 'gray', alpha=0.6, label='L1 Data')
    axes[0].plot(time*1000, model_map, 'r-', linewidth=2, label='MAP Fit')
    axes[0].axvline(results['map'][4]*1000, color='b', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Tiempo [ms]')
    axes[0].set_ylabel('Strain normalizado')
    axes[0].set_title(f'GW150914 Ringdown | M_src={m_src:.0f}M☉, a={a:.2f}')
    axes[0].legend()
    axes[0].set_xlim(-5, 45)
    
    residual = data - model_map
    axes[1].plot(time*1000, residual, 'gray', alpha=0.7)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Tiempo [ms]')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'Residuales (RMS: {np.std(residual):.2f})')
    axes[1].set_xlim(-5, 45)
    
    plt.tight_layout()
    plt.savefig(f'{outdir}/fit.png', dpi=150)
    plt.close()
    
    # Trace plot
    chain = results['chain']
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    labels_trace = ['Mass', 'Spin', 'Amplitude', 'Phase', 't0']
    
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3)
        ax.set_ylabel(labels_trace[i])
        ax.axvline(500, color='r', linestyle='--', alpha=0.5, label='Burnin')
    axes[-1].set_xlabel('Step')
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/trace.png', dpi=150)
    plt.close()
    
    print(f"\n✅ Análisis completado. Resultados en {outdir}/")


if __name__ == "__main__":
    main()
