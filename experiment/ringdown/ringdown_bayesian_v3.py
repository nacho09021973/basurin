#!/usr/bin/env python3
"""
ringdown_bayesian_v3.py
=======================
ANÁLISIS BAYESIANO DE RINGDOWN CON CLASIFICACIÓN AUTOMÁTICA ROBUSTA

v3.1 Fix:
- Bandpower voting ahora usa FFT filtrada (no full), evitando sesgo LF
- Bandas de voting se ajustan a [auto_fmin, auto_fmax] para coherencia

v3 Mejoras respecto a v2:
- Auto-detección ampliada a 20-500 Hz (antes 30-500) para capturar LF
- Bandpower voting + métrica de confianza para detectar ambigüedad
- Ponderación por SNR en multi-detector (--multiweight)
- Flag --overtones para control explícito

El tratamiento óptimo depende del tipo de evento:
- Stellar-mass BBH (M ~ 20-100 Msun): f_QNM ~ 100-500 Hz
- Intermediate-mass BBH (M ~ 100-500 Msun): f_QNM ~ 30-150 Hz  
- Heavy/High-z (M_det > 500 Msun): f_QNM < 50 Hz
- Light BBH / BNS remnant: f_QNM > 500 Hz

Uso:
    python ringdown_bayesian_v3.py --data l1.npy --redshift 0.09
    python ringdown_bayesian_v3.py --data l1.npy --category stellar  # forzar categoría
    python ringdown_bayesian_v3.py --data l1.npy --multiweight snr   # ponderación SNR

Autor: José Ignacio Martín Gandul
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
import emcee
import corner
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTES FÍSICAS
# ============================================================================
C = 299792458.0
G = 6.67430e-11
MSUN = 1.98847e30

# Coeficientes de Berti (l,m,n) = (2,2,0)
BERTI_F = (1.5251, -1.1568, 0.1292)
BERTI_Q = (0.7000, 1.4187, -0.4990)

# Coeficientes para overtone (2,2,1) - opcional
BERTI_F_221 = (1.2788, -1.1568, 0.1292)  # Aproximado
BERTI_Q_221 = (0.3737, 1.2788, -0.4990)  # Aproximado


# ============================================================================
# CLASIFICACIÓN DE EVENTOS
# ============================================================================
class EventCategory(Enum):
    """Categorías de eventos según masa/frecuencia."""
    LIGHT = "light"           # M < 30 Msun, f > 400 Hz (BNS remnant, light BBH)
    STELLAR = "stellar"       # M ~ 30-100 Msun, f ~ 150-400 Hz (típico BBH)
    INTERMEDIATE = "imbh"     # M ~ 100-500 Msun, f ~ 30-150 Hz (IMBH)
    HEAVY = "heavy"           # M > 500 Msun, f < 50 Hz (muy masivo o alto z)
    UNKNOWN = "unknown"


@dataclass
class CategoryConfig:
    """Configuración optimizada para cada categoría."""
    name: str
    description: str
    
    # Filtrado
    fmin: float
    fmax: float
    
    # Priors
    mass_range: Tuple[float, float]
    spin_range: Tuple[float, float]
    
    # Ventana temporal
    pre_peak_ms: float
    post_peak_ms: float
    
    # Modelo
    include_overtone: bool
    
    # MCMC
    nwalkers: int
    nsteps: int
    burnin: int


# Configuraciones por categoría
CATEGORY_CONFIGS = {
    EventCategory.LIGHT: CategoryConfig(
        name="Light BBH / Post-merger",
        description="M < 30 Msun, f_QNM > 400 Hz",
        fmin=100.0,
        fmax=800.0,
        mass_range=(5, 40),
        spin_range=(0.0, 0.95),
        pre_peak_ms=3.0,
        post_peak_ms=20.0,  # τ muy corto
        include_overtone=True,  # Overtones más visibles
        nwalkers=32,
        nsteps=3000,
        burnin=1000
    ),
    
    EventCategory.STELLAR: CategoryConfig(
        name="Stellar-mass BBH",
        description="M ~ 30-100 Msun, f_QNM ~ 150-400 Hz (GW150914-like)",
        fmin=20.0,  # Más amplio para no perder información
        fmax=500.0,
        mass_range=(25, 120),
        spin_range=(0.0, 0.95),
        pre_peak_ms=5.0,
        post_peak_ms=50.0,
        include_overtone=False,
        nwalkers=32,
        nsteps=2500,
        burnin=800
    ),
    
    EventCategory.INTERMEDIATE: CategoryConfig(
        name="Intermediate-mass BBH",
        description="M ~ 100-500 Msun, f_QNM ~ 30-150 Hz (IMBH)",
        fmin=20.0,
        fmax=200.0,
        mass_range=(80, 600),
        spin_range=(0.0, 0.99),
        pre_peak_ms=10.0,
        post_peak_ms=100.0,  # τ más largo
        include_overtone=False,
        nwalkers=40,
        nsteps=3500,
        burnin=1200
    ),
    
    EventCategory.HEAVY: CategoryConfig(
        name="Heavy / High-redshift",
        description="M_det > 500 Msun, f_QNM < 50 Hz",
        fmin=10.0,
        fmax=100.0,
        mass_range=(400, 2000),
        spin_range=(0.0, 0.99),
        pre_peak_ms=20.0,
        post_peak_ms=200.0,
        include_overtone=False,
        nwalkers=48,
        nsteps=4000,
        burnin=1500
    ),
    
    EventCategory.UNKNOWN: CategoryConfig(
        name="Unknown / Custom",
        description="Configuración por defecto amplia",
        fmin=20.0,
        fmax=500.0,
        mass_range=(10, 500),
        spin_range=(0.0, 0.99),
        pre_peak_ms=10.0,
        post_peak_ms=100.0,
        include_overtone=False,
        nwalkers=32,
        nsteps=3000,
        burnin=1000
    )
}


def classify_event(f_dominant: float, snr: float = None) -> EventCategory:
    """
    Clasifica el evento basándose en la frecuencia dominante detectada.
    
    Args:
        f_dominant: Frecuencia dominante en Hz
        snr: SNR del pico (opcional, para ajustar confianza)
    
    Returns:
        EventCategory correspondiente
    """
    if f_dominant > 400:
        return EventCategory.LIGHT
    elif f_dominant > 150:
        return EventCategory.STELLAR
    elif f_dominant > 50:
        return EventCategory.INTERMEDIATE
    elif f_dominant > 10:
        return EventCategory.HEAVY
    else:
        return EventCategory.UNKNOWN


def mass_from_frequency(f_hz: float, spin: float = 0.7) -> float:
    """Estima masa a partir de frecuencia QNM."""
    spin = float(np.clip(spin, 0.01, 0.99))
    w_bar = BERTI_F[0] + BERTI_F[1] * np.power(1 - spin, BERTI_F[2])
    Tg = w_bar / (2 * np.pi * f_hz)
    mass_kg = Tg * C**3 / G
    return mass_kg / MSUN


# ============================================================================
# FÍSICA DEL RINGDOWN
# ============================================================================
def get_kerr_qnm(mass: float, spin: float, mode: str = "220") -> Tuple[float, float]:
    """
    Calcula frecuencia y τ para modos QNM.
    
    Args:
        mass: Masa [Msun]
        spin: Spin adimensional
        mode: "220" (fundamental) o "221" (primer overtone)
    """
    spin = float(np.clip(spin, 0.01, 0.99))
    
    if mode == "220":
        coeffs_f, coeffs_q = BERTI_F, BERTI_Q
    elif mode == "221":
        coeffs_f, coeffs_q = BERTI_F_221, BERTI_Q_221
    else:
        raise ValueError(f"Modo {mode} no soportado")
    
    w_bar = coeffs_f[0] + coeffs_f[1] * np.power(1 - spin, coeffs_f[2])
    Q = coeffs_q[0] + coeffs_q[1] * np.power(1 - spin, coeffs_q[2])
    
    Tg = (G * mass * MSUN) / C**3
    f_hz = w_bar / (2 * np.pi * Tg)
    tau_s = Q / (np.pi * f_hz)
    
    return f_hz, max(tau_s, 1e-6)


def ringdown_model_single(time: np.ndarray, mass: float, spin: float,
                          amplitude: float, phase: float, t0: float) -> np.ndarray:
    """Modelo con solo modo fundamental (220)."""
    f, tau = get_kerr_qnm(mass, spin, "220")
    
    output = np.zeros_like(time)
    mask = time >= t0
    if np.sum(mask) == 0:
        return output
    
    dt = time[mask] - t0
    output[mask] = amplitude * np.exp(-dt / tau) * np.cos(2 * np.pi * f * dt + phase)
    
    return output


def ringdown_model_overtone(time: np.ndarray, mass: float, spin: float,
                            A0: float, phi0: float, A1: float, phi1: float, 
                            t0: float) -> np.ndarray:
    """Modelo con fundamental (220) + primer overtone (221)."""
    f0, tau0 = get_kerr_qnm(mass, spin, "220")
    f1, tau1 = get_kerr_qnm(mass, spin, "221")
    
    output = np.zeros_like(time)
    mask = time >= t0
    if np.sum(mask) == 0:
        return output
    
    dt = time[mask] - t0
    
    # Modo fundamental
    h0 = A0 * np.exp(-dt / tau0) * np.cos(2 * np.pi * f0 * dt + phi0)
    
    # Overtone (decae más rápido)
    h1 = A1 * np.exp(-dt / tau1) * np.cos(2 * np.pi * f1 * dt + phi1)
    
    output[mask] = h0 + h1
    
    return output


# ============================================================================
# PREPROCESAMIENTO
# ============================================================================
def estimate_dominant_frequency(data: np.ndarray, fs: float,
                                fmin: float = 20, fmax: float = 600) -> Tuple[float, float]:
    """
    Estima frecuencia dominante y su potencia relativa.
    
    Returns:
        f_dominant: Frecuencia del pico [Hz]
        snr_freq: Ratio señal/ruido en frecuencia
    """
    N = len(data)
    center = N // 2
    window_size = min(int(0.2 * fs), N // 4)
    
    segment = data[center-window_size:center+window_size]
    
    # FFT con ventana
    win = np.hanning(len(segment))
    fft_mag = np.abs(np.fft.rfft(segment * win))
    freqs = np.fft.rfftfreq(len(segment), 1/fs)
    
    # Buscar pico en rango válido
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 200.0, 1.0
    
    fft_masked = fft_mag[mask]
    freqs_masked = freqs[mask]
    
    peak_idx = np.argmax(fft_masked)
    f_dominant = freqs_masked[peak_idx]
    
    # SNR en frecuencia
    peak_power = fft_masked[peak_idx]
    noise_power = np.median(fft_masked)
    snr_freq = peak_power / noise_power if noise_power > 0 else 1.0
    
    return f_dominant, snr_freq


def compute_bandpower(fft_mag: np.ndarray, freqs: np.ndarray, 
                      f_low: float, f_high: float) -> float:
    """
    Calcula potencia integrada en una banda de frecuencia.
    
    Args:
        fft_mag: Magnitud FFT
        freqs: Array de frecuencias
        f_low, f_high: Límites de la banda
    
    Returns:
        Potencia integrada (suma de |X(f)|²)
    """
    mask = (freqs >= f_low) & (freqs < f_high)
    if not np.any(mask):
        return 0.0
    return np.sum(fft_mag[mask]**2)


def bandpower_voting(fft_mag: np.ndarray, freqs: np.ndarray, 
                     fmin_auto: float = 10.0, fmax_auto: float = 800.0) -> Dict:
    """
    Clasifica por potencia integrada en bandas y calcula confianza.
    
    v3.1: Las bandas se ajustan a [fmin_auto, fmax_auto] para no medir
    fuera de la banda de auto-detección.
    
    Bandas nominales (ajustadas por fmin_auto/fmax_auto):
    - LIGHT:        400-800 Hz
    - STELLAR:      150-400 Hz  
    - INTERMEDIATE: 50-150 Hz
    - HEAVY:        10-50 Hz
    
    Returns:
        Dict con category, confidence, bandpowers
    """
    # v3.1: Ajustar bandas para respetar límites de auto-detección
    bands = {
        EventCategory.LIGHT: (max(400, fmin_auto), min(800, fmax_auto)),
        EventCategory.STELLAR: (max(150, fmin_auto), min(400, fmax_auto)),
        EventCategory.INTERMEDIATE: (max(50, fmin_auto), min(150, fmax_auto)),
        EventCategory.HEAVY: (max(10, fmin_auto), min(50, fmax_auto))
    }
    
    powers = {}
    for cat, (f_lo, f_hi) in bands.items():
        powers[cat] = compute_bandpower(fft_mag, freqs, f_lo, f_hi)
    
    # Normalizar por ancho de banda para comparación justa
    bw_normalized = {}
    for cat, (f_lo, f_hi) in bands.items():
        bw = f_hi - f_lo
        bw_normalized[cat] = powers[cat] / bw if bw > 0 else 0
    
    # Ordenar por potencia normalizada
    sorted_cats = sorted(bw_normalized.keys(), key=lambda c: bw_normalized[c], reverse=True)
    
    best = sorted_cats[0]
    second = sorted_cats[1] if len(sorted_cats) > 1 else None
    
    # Confianza = ratio best / second
    eps = 1e-10
    if second and bw_normalized[second] > eps:
        confidence = bw_normalized[best] / (bw_normalized[second] + eps)
    else:
        confidence = 10.0  # Muy alta si no hay competencia
    
    return {
        'category': best,
        'confidence': confidence,
        'bandpowers': powers,
        'bandpowers_normalized': bw_normalized,
        'ranking': sorted_cats
    }


def preprocess_data(raw_data: np.ndarray, fs: float, config: CategoryConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Preprocesa datos según la configuración de categoría."""
    N = len(raw_data)
    
    # 1. DC offset
    dc_offset = raw_data.mean()
    data_fixed = raw_data - dc_offset
    
    # 2. Ventana Tukey
    window = tukey(N, alpha=0.1)
    data_windowed = data_fixed * window
    
    # 3. PSD y whitening
    nperseg = min(int(4 * fs), N // 4)
    freqs_psd, psd = welch(data_windowed, fs, nperseg=nperseg, noverlap=nperseg//2)
    psd = np.maximum(psd, 1e-50)
    
    interp_psd = interp1d(freqs_psd, psd, kind='linear', fill_value="extrapolate")
    
    freqs_fft = np.fft.rfftfreq(N, 1/fs)
    data_fft = np.fft.rfft(data_windowed)
    
    psd_fft = np.maximum(interp_psd(freqs_fft), 1e-50)
    whitened_fft = data_fft / np.sqrt(psd_fft)
    
    # 4. Filtrado según categoría
    nyq = fs / 2
    fmin_norm = config.fmin / nyq
    fmax_norm = min(config.fmax / nyq, 0.99)
    
    # Filtro más suave para frecuencias bajas
    if config.fmin < 30:
        order = 2
    else:
        order = 4
    
    b, a = butter(order, [fmin_norm, fmax_norm], btype='band')
    
    whitened = np.fft.irfft(whitened_fft, n=N)
    filtered = filtfilt(b, a, whitened)
    
    # 5. Encontrar pico
    center = N // 2
    search_window = min(int(2 * fs), N // 4)
    search_region = filtered[max(0, center-search_window):min(N, center+search_window)]
    
    peak_idx_local = np.argmax(np.abs(search_region))
    global_peak = max(0, center - search_window) + peak_idx_local
    
    # 6. Extraer ventana según categoría
    pre_peak = int(config.pre_peak_ms / 1000 * fs)
    post_peak = int(config.post_peak_ms / 1000 * fs)
    
    rd_start = max(0, global_peak - pre_peak)
    rd_end = min(N, global_peak + post_peak)
    
    segment = filtered[rd_start:rd_end]
    actual_pre = global_peak - rd_start
    time = (np.arange(len(segment)) - actual_pre) / fs
    
    # 7. Normalizar
    sigma = np.std(segment)
    segment_norm = segment / sigma if sigma > 0 else segment
    
    # 8. Diagnósticos
    f_dominant, snr_freq = estimate_dominant_frequency(filtered, fs, config.fmin, config.fmax)
    
    info = {
        'dc_offset': dc_offset,
        'sigma': sigma,
        'snr_peak': np.max(np.abs(segment_norm)),
        'snr_freq': snr_freq,
        'duration_ms': len(segment) / fs * 1000,
        'f_dominant': f_dominant,
        'mass_estimate': mass_from_frequency(f_dominant),
        'fmin_used': config.fmin,
        'fmax_used': config.fmax
    }
    
    return time, segment_norm, info


# ============================================================================
# ANÁLISIS BAYESIANO
# ============================================================================
class RingdownAnalysis:
    """Análisis Bayesiano con clasificación automática robusta."""
    
    # Umbrales de confianza para detección automática
    CONFIDENCE_THRESHOLD = 1.5  # Si < 1.5, advertir al usuario
    CONFIDENCE_MINIMUM = 1.2    # Si < 1.2, pedir verificación explícita
    
    def __init__(self, data_files: List[str], detector_names: List[str] = None,
                 fs: float = 4096.0, redshift: float = None,
                 category: EventCategory = None, outdir: str = "ringdown_output",
                 multiweight: str = "equal", overtones: str = "auto",
                 auto_fmin: float = 20.0, auto_fmax: float = 500.0):
        
        self.data_files = data_files
        self.detector_names = detector_names or [f"Det{i+1}" for i in range(len(data_files))]
        self.fs = fs
        self.redshift = redshift
        self.forced_category = category
        self.outdir = outdir
        self.multiweight = multiweight
        self.overtones = overtones  # 'on', 'off', 'auto'
        self.auto_fmin = auto_fmin
        self.auto_fmax = auto_fmax
        
        self.times = []
        self.data = []
        self.info = []
        self.category = None
        self.config = None
        self.detection_confidence = None
        self.bandpower_results = None
        
    def _detect_category(self) -> Tuple[EventCategory, float, Dict]:
        """
        Detecta categoría usando bandpower voting + frecuencia de pico.
        
        v3: Ampliado a 20-500 Hz (antes 30-500) y con métrica de confianza.
        
        Returns:
            (category, confidence, bandpower_results)
        """
        all_voting_results = []
        f_values = []
        snr_values = []
        mass_estimates = []
        
        for fpath in self.data_files:
            raw = np.load(fpath)
            raw_centered = raw - raw.mean()
            N = len(raw_centered)
            
            # Whitening completo
            window = tukey(N, alpha=0.1)
            data_windowed = raw_centered * window
            
            nperseg = min(int(4 * self.fs), N // 4)
            freqs_psd, psd = welch(data_windowed, self.fs, nperseg=nperseg, noverlap=nperseg//2)
            psd = np.maximum(psd, 1e-50)
            interp_psd = interp1d(freqs_psd, psd, fill_value="extrapolate")
            
            freqs_fft = np.fft.rfftfreq(N, 1/self.fs)
            whitened_fft = np.fft.rfft(data_windowed) / np.sqrt(np.maximum(interp_psd(freqs_fft), 1e-50))
            
            # v3: Filtro ampliado a [20, 500] Hz (antes [30, 500])
            # Esto permite capturar mejor eventos con contenido LF significativo
            mask = (freqs_fft >= self.auto_fmin) & (freqs_fft <= self.auto_fmax)
            whitened_fft_filtered = whitened_fft.copy()
            whitened_fft_filtered[~mask] = 0
            
            whitened = np.fft.irfft(whitened_fft_filtered, n=N)
            
            # v3.1: Bandpower voting sobre FFT FILTRADA (no full)
            # Esto evita que ruido LF fuera de banda sesgue la confianza
            fft_mag_filtered = np.abs(whitened_fft_filtered)
            voting = bandpower_voting(fft_mag_filtered, freqs_fft, 
                                      fmin_auto=self.auto_fmin, 
                                      fmax_auto=self.auto_fmax)
            all_voting_results.append(voting)
            
            # Buscar frecuencia dominante en ventana centrada en el pico
            center = N // 2
            search_window = min(int(1 * self.fs), N // 4)
            search_region = whitened[center-search_window:center+search_window]
            
            # FFT del pico para encontrar frecuencia
            peak_idx = np.argmax(np.abs(search_region))
            peak_start = max(0, peak_idx - int(0.05 * self.fs))
            peak_end = min(len(search_region), peak_idx + int(0.05 * self.fs))
            peak_segment = search_region[peak_start:peak_end]
            
            if len(peak_segment) > 10:
                win = np.hanning(len(peak_segment))
                fft_mag = np.abs(np.fft.rfft(peak_segment * win))
                freqs_local = np.fft.rfftfreq(len(peak_segment), 1/self.fs)
                
                # v3: Búsqueda de pico ampliada a [20, 450] Hz (antes [50, 400])
                mask_search = (freqs_local >= self.auto_fmin) & (freqs_local <= min(self.auto_fmax, 450))
                if np.any(mask_search):
                    peak_freq_idx = np.argmax(fft_mag[mask_search])
                    f_dominant = freqs_local[mask_search][peak_freq_idx]
                else:
                    f_dominant = 200.0  # Default
            else:
                f_dominant = 200.0
            
            # SNR del pico temporal
            snr_peak = np.max(np.abs(search_region)) / np.std(search_region)
            
            f_values.append(f_dominant)
            snr_values.append(snr_peak)
            mass_estimates.append(mass_from_frequency(f_dominant))
        
        # Usar detector con mejor SNR
        best_idx = np.argmax(snr_values)
        f_dominant = f_values[best_idx]
        mass_est = mass_estimates[best_idx]
        voting_result = all_voting_results[best_idx]
        
        # Combinar evidencia: frecuencia de pico + bandpower voting
        # Si hay desacuerdo significativo, reducir confianza
        category_from_mass = classify_event(f_dominant)
        category_from_voting = voting_result['category']
        
        # Confianza base del voting
        confidence = voting_result['confidence']
        
        # Penalizar si pico y voting discrepan
        if category_from_mass != category_from_voting:
            confidence *= 0.7  # Reducir confianza en caso de discrepancia
            print(f"   ⚠️  Discrepancia: pico sugiere {category_from_mass.value}, voting sugiere {category_from_voting.value}")
        
        # Decisión final: priorizar voting si confianza es alta, sino usar masa
        if confidence > self.CONFIDENCE_THRESHOLD:
            final_category = category_from_voting
        else:
            # En caso de baja confianza, usar clasificación por masa (más tradicional)
            final_category = category_from_mass
        
        print(f"   Análisis preliminar: f={f_dominant:.0f} Hz, M~{mass_est:.0f} Msun, SNR={snr_values[best_idx]:.1f}")
        print(f"   Bandpower voting: {voting_result['category'].value} (conf={confidence:.2f})")
        
        # Advertencias de confianza
        if confidence < self.CONFIDENCE_MINIMUM:
            print(f"\n   ⛔ CONFIANZA MUY BAJA ({confidence:.2f} < {self.CONFIDENCE_MINIMUM})")
            print(f"       Recomendación: usar --category para especificar manualmente")
            print(f"       Sugerencias: --category stellar | --category imbh | --category heavy")
        elif confidence < self.CONFIDENCE_THRESHOLD:
            print(f"\n   ⚠️  Confianza moderada ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD})")
            print(f"       Considere verificar con --category si los resultados no son satisfactorios")
        
        return final_category, confidence, voting_result
    
    def load_and_preprocess(self):
        """Carga datos y detecta/aplica categoría."""
        print("\n" + "="*60)
        print("🔍 DETECCIÓN DE CATEGORÍA")
        print("="*60)
        
        if self.forced_category:
            self.category = self.forced_category
            self.detection_confidence = None
            print(f"   ✓ Categoría forzada: {self.category.value}")
            print(f"     (usando bounds de categoría directamente, sin data-driven centering)")
        else:
            self.category, self.detection_confidence, self.bandpower_results = self._detect_category()
            print(f"   → Categoría detectada: {self.category.value}")
        
        self.config = CATEGORY_CONFIGS[self.category]
        
        # Override overtones según flag
        if self.overtones == 'on':
            # Forzar overtones activos (crear copia modificada de config)
            self.config = CategoryConfig(
                name=self.config.name,
                description=self.config.description,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                mass_range=self.config.mass_range,
                spin_range=self.config.spin_range,
                pre_peak_ms=self.config.pre_peak_ms,
                post_peak_ms=self.config.post_peak_ms,
                include_overtone=True,  # Forzado
                nwalkers=self.config.nwalkers,
                nsteps=self.config.nsteps,
                burnin=self.config.burnin
            )
            print(f"   ⚡ Overtones forzados: ON")
        elif self.overtones == 'off':
            self.config = CategoryConfig(
                name=self.config.name,
                description=self.config.description,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
                mass_range=self.config.mass_range,
                spin_range=self.config.spin_range,
                pre_peak_ms=self.config.pre_peak_ms,
                post_peak_ms=self.config.post_peak_ms,
                include_overtone=False,  # Forzado
                nwalkers=self.config.nwalkers,
                nsteps=self.config.nsteps,
                burnin=self.config.burnin
            )
            print(f"   ⚡ Overtones forzados: OFF")
        
        print(f"\n📋 {self.config.name}")
        print(f"   {self.config.description}")
        print(f"   Filtrado: [{self.config.fmin}, {self.config.fmax}] Hz")
        print(f"   Masa prior: [{self.config.mass_range[0]}, {self.config.mass_range[1]}] Msun")
        print(f"   Ventana: -{self.config.pre_peak_ms}ms / +{self.config.post_peak_ms}ms")
        print(f"   Overtone: {'Sí' if self.config.include_overtone else 'No'}")
        print(f"   Multiweight: {self.multiweight}")
        
        print("\n" + "="*60)
        print("🔧 PREPROCESAMIENTO")
        print("="*60)
        
        for i, fpath in enumerate(self.data_files):
            det_name = self.detector_names[i]
            print(f"\n   [{det_name}] {fpath}")
            
            raw = np.load(fpath)
            time, data, info = preprocess_data(raw, self.fs, self.config)
            
            self.times.append(time)
            self.data.append(data)
            self.info.append(info)
            
            print(f"       SNR pico: {info['snr_peak']:.1f}")
            print(f"       f dominante: {info['f_dominant']:.0f} Hz")
            print(f"       M estimada: {info['mass_estimate']:.0f} Msun")
    
    def _setup_bounds(self):
        """Configura bounds para MCMC."""
        cfg = self.config
        
        # Si categoría está forzada, usar rangos de la categoría directamente
        if self.forced_category is not None:
            m_min, m_max = cfg.mass_range
            print(f"   ✓ Usando bounds de categoría forzada (sin data-driven centering)")
        else:
            # Ajustar masa según frecuencia observada
            f_obs = np.mean([info['f_dominant'] for info in self.info])
            m_est = mass_from_frequency(f_obs)
            
            # Usar el rango de la categoría pero centrar en estimación
            m_min = max(cfg.mass_range[0], m_est * 0.4)
            m_max = min(cfg.mass_range[1], m_est * 2.0)
            
            # Asegurar que m_min < m_max
            if m_min >= m_max:
                m_min, m_max = cfg.mass_range
        
        # Amplitud según SNR
        snr_max = max([info['snr_peak'] for info in self.info])
        amp_max = max(snr_max * 4, 15)  # Mínimo 15
        
        if cfg.include_overtone:
            # 7 parámetros: mass, spin, A0, phi0, A1, phi1, t0
            self.bounds = np.array([
                [m_min, m_max],
                list(cfg.spin_range),
                [0.5, amp_max],           # A0
                [0, 2*np.pi],             # phi0
                [0.1, amp_max * 0.5],     # A1 (típicamente menor que A0)
                [0, 2*np.pi],             # phi1
                [-0.01, 0.01]             # t0
            ])
            self.param_names = ['mass', 'spin', 'A0', 'phi0', 'A1', 'phi1', 't0']
        else:
            # 5 parámetros
            self.bounds = np.array([
                [m_min, m_max],
                list(cfg.spin_range),
                [0.5, amp_max],
                [0, 2*np.pi],
                [-0.01, 0.01]
            ])
            self.param_names = ['mass', 'spin', 'amplitude', 'phase', 't0']
        
        print(f"\n📊 Priors finales:")
        print(f"   Mass: [{self.bounds[0,0]:.0f}, {self.bounds[0,1]:.0f}] Msun")
        print(f"   Spin: [{self.bounds[1,0]:.2f}, {self.bounds[1,1]:.2f}]")
    
    def _compute_weights(self) -> np.ndarray:
        """
        Calcula pesos para cada detector según configuración multiweight.
        
        Returns:
            Array de pesos normalizados (suma = n_detectores)
        """
        n_det = len(self.info)
        
        if self.multiweight == 'equal':
            return np.ones(n_det)
        
        elif self.multiweight == 'snr':
            # Peso proporcional a SNR²
            snrs = np.array([info['snr_peak'] for info in self.info])
            weights = snrs ** 2
            # Normalizar para que la suma sea n_det (preserva escala de chi2)
            weights = weights / np.mean(weights)
            return weights
        
        elif self.multiweight == 'sigma':
            # Peso inversamente proporcional a sigma²
            sigmas = np.array([info['sigma'] for info in self.info])
            weights = 1.0 / (sigmas ** 2 + 1e-20)
            weights = weights / np.mean(weights)
            return weights
        
        else:
            print(f"   ⚠️  multiweight '{self.multiweight}' no reconocido, usando 'equal'")
            return np.ones(n_det)
    
    def log_prior(self, theta):
        for i, (lo, hi) in enumerate(self.bounds):
            if not (lo <= theta[i] <= hi):
                return -np.inf
        return 0.0
    
    def log_likelihood(self, theta):
        if self.config.include_overtone:
            mass, spin, A0, phi0, A1, phi1, t0 = theta
            model_func = lambda t: ringdown_model_overtone(t, mass, spin, A0, phi0, A1, phi1, t0)
        else:
            mass, spin, amp, phase, t0 = theta
            model_func = lambda t: ringdown_model_single(t, mass, spin, amp, phase, t0)
        
        # v3: Ponderación por detector
        weights = self._compute_weights()
        
        total_chi2 = 0.0
        for i, (time, data) in enumerate(zip(self.times, self.data)):
            model = model_func(time)
            chi2_det = np.sum((data - model)**2)
            total_chi2 += weights[i] * chi2_det
        
        return -0.5 * total_chi2
    
    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf
    
    def find_map(self):
        """Encuentra MAP."""
        print("\n🔍 Buscando MAP...")
        
        def neg_log_prob(theta):
            return -self.log_probability(theta)
        
        bounds_list = [(lo, hi) for lo, hi in self.bounds]
        
        best_result = None
        best_val = np.inf
        
        # Differential evolution
        try:
            result = differential_evolution(neg_log_prob, bounds_list, 
                                           maxiter=300, seed=42, workers=1)
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x
        except:
            pass
        
        # Random starts
        np.random.seed(42)
        for _ in range(30):
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in self.bounds])
            try:
                result = minimize(neg_log_prob, x0, method='Nelder-Mead',
                                options={'maxiter': 2000})
                if result.fun < best_val:
                    best_val = result.fun
                    best_result = result.x
            except:
                pass
        
        if best_result is not None:
            print(f"   MAP: M={best_result[0]:.1f}, a={best_result[1]:.2f}")
        
        return best_result
    
    def run_mcmc(self):
        """Ejecuta MCMC."""
        self._setup_bounds()
        map_estimate = self.find_map()
        
        if map_estimate is None:
            raise ValueError("No se encontró MAP")
        
        cfg = self.config
        ndim = len(self.bounds)
        
        print(f"\n🎲 MCMC: {cfg.nwalkers} walkers, {cfg.nsteps} steps")
        if self.multiweight != 'equal':
            weights = self._compute_weights()
            print(f"   Pesos por detector: {[f'{w:.2f}' for w in weights]}")
        
        # Inicializar walkers
        pos = map_estimate + 1e-3 * np.random.randn(cfg.nwalkers, ndim)
        for i in range(cfg.nwalkers):
            for j in range(ndim):
                pos[i, j] = np.clip(pos[i, j], self.bounds[j, 0] + 1e-6, self.bounds[j, 1] - 1e-6)
        
        sampler = emcee.EnsembleSampler(cfg.nwalkers, ndim, self.log_probability)
        sampler.run_mcmc(pos, cfg.nsteps, progress=True)
        
        samples = sampler.get_chain(discard=cfg.burnin, flat=True)
        
        return {
            'samples': samples,
            'chain': sampler.get_chain(),
            'map': map_estimate,
            'acceptance': np.mean(sampler.acceptance_fraction)
        }
    
    def analyze(self):
        """Análisis completo."""
        self.load_and_preprocess()
        results = self.run_mcmc()
        
        samples = results['samples']
        
        # Extraer estadísticas
        mass_pct = np.percentile(samples[:, 0], [16, 50, 84])
        spin_pct = np.percentile(samples[:, 1], [16, 50, 84])
        
        # t0 está en la última columna
        t0_pct = np.percentile(samples[:, -1], [16, 50, 84])
        
        # Masa fuente
        if self.redshift:
            mass_src = mass_pct / (1 + self.redshift)
        else:
            mass_src = None
        
        # QNM
        f_qnm, tau_qnm = get_kerr_qnm(mass_pct[1], spin_pct[1])
        
        results['summary'] = {
            'category': self.category.value,
            'mass_detector': {'median': mass_pct[1], 'lower': mass_pct[1]-mass_pct[0], 'upper': mass_pct[2]-mass_pct[1]},
            'mass_source': {'median': mass_src[1], 'lower': mass_src[1]-mass_src[0], 'upper': mass_src[2]-mass_src[1]} if mass_src is not None else None,
            'spin': {'median': spin_pct[1], 'lower': spin_pct[1]-spin_pct[0], 'upper': spin_pct[2]-spin_pct[1]},
            't0_ms': {'median': t0_pct[1]*1000, 'lower': (t0_pct[1]-t0_pct[0])*1000, 'upper': (t0_pct[2]-t0_pct[1])*1000},
            'f_qnm_hz': f_qnm,
            'tau_qnm_ms': tau_qnm * 1000,
            'redshift': self.redshift,
            'include_overtone': self.config.include_overtone,
            'detection_confidence': self.detection_confidence,
            'multiweight': self.multiweight
        }
        
        results['config'] = {
            'category': self.category.value,
            'fmin': self.config.fmin,
            'fmax': self.config.fmax,
            'mass_range': self.config.mass_range,
            'detectors': self.detector_names,
            'auto_fmin': self.auto_fmin,
            'auto_fmax': self.auto_fmax,
            'overtones': self.overtones,
            'multiweight': self.multiweight
        }
        
        return results
    
    def print_results(self, results):
        s = results['summary']
        
        print("\n" + "="*60)
        print(f"🏆 RESULTADOS ({s['category'].upper()})")
        print("="*60)
        
        m = s['mass_detector']
        print(f"   Masa (detector): {m['median']:.1f} +{m['upper']:.1f}/-{m['lower']:.1f} Msun")
        
        if s['mass_source']:
            m = s['mass_source']
            print(f"   Masa (fuente):   {m['median']:.1f} +{m['upper']:.1f}/-{m['lower']:.1f} Msun  (z={s['redshift']})")
        
        sp = s['spin']
        print(f"   Spin:            {sp['median']:.3f} +{sp['upper']:.3f}/-{sp['lower']:.3f}")
        
        t = s['t0_ms']
        print(f"   t0:              {t['median']:.2f} +{t['upper']:.2f}/-{t['lower']:.2f} ms")
        
        print(f"\n   f_QNM (220):     {s['f_qnm_hz']:.1f} Hz")
        print(f"   τ_QNM:           {s['tau_qnm_ms']:.2f} ms")
        
        if s['include_overtone']:
            print(f"\n   ⚠️  Modelo incluye overtone (221)")
        
        if s['detection_confidence'] is not None:
            print(f"\n   📊 Confianza detección: {s['detection_confidence']:.2f}")
            if s['detection_confidence'] < self.CONFIDENCE_THRESHOLD:
                print(f"      (considere verificar con --category)")
        
        if s['multiweight'] != 'equal':
            print(f"   🔧 Ponderación: {s['multiweight']}")
    
    def save_results(self, results):
        os.makedirs(self.outdir, exist_ok=True)
        
        print(f"\n📊 Guardando en {self.outdir}/...")
        
        # JSON
        with open(f"{self.outdir}/summary.json", 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)
        
        # Config
        with open(f"{self.outdir}/config.json", 'w') as f:
            json.dump(results['config'], f, indent=2, default=str)
        
        # Samples
        np.save(f"{self.outdir}/samples.npy", results['samples'])
        
        # Corner
        labels = [p.replace('_', ' ') for p in self.param_names]
        fig = corner.corner(results['samples'], labels=labels,
                           quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig(f'{self.outdir}/corner.png', dpi=150)
        plt.close()
        
        # Fit
        n_det = len(self.times)
        fig, axes = plt.subplots(n_det, 2, figsize=(12, 4*n_det), squeeze=False)
        
        map_params = results['map']
        
        for i, (time, data, info) in enumerate(zip(self.times, self.data, self.info)):
            det_name = self.detector_names[i]
            
            if self.config.include_overtone:
                model = ringdown_model_overtone(time, *map_params)
            else:
                model = ringdown_model_single(time, *map_params)
            
            axes[i, 0].plot(time*1000, data, 'gray', alpha=0.6, label=f'{det_name}')
            axes[i, 0].plot(time*1000, model, 'r-', lw=2, label='MAP')
            axes[i, 0].set_xlabel('t [ms]')
            axes[i, 0].set_ylabel('Strain')
            axes[i, 0].legend()
            axes[i, 0].set_title(f'{det_name} | {self.category.value} | SNR={info["snr_peak"]:.1f}')
            
            residual = data - model
            axes[i, 1].plot(time*1000, residual, 'gray')
            axes[i, 1].axhline(0, color='r', ls='--')
            axes[i, 1].set_xlabel('t [ms]')
            axes[i, 1].set_ylabel('Residual')
            axes[i, 1].set_title(f'RMS: {np.std(residual):.2f}')
        
        plt.tight_layout()
        plt.savefig(f'{self.outdir}/fit.png', dpi=150)
        plt.close()
        
        print("   ✅ summary.json, config.json, samples.npy, corner.png, fit.png")


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Ringdown Bayesiano v3 - Clasificación automática robusta',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Detección automática (default)
    python ringdown_bayesian_v3.py --data l1.npy h1.npy --redshift 0.09

    # Forzar categoría (recomendado si confianza baja)
    python ringdown_bayesian_v3.py --data l1.npy --category stellar

    # Multi-detector con ponderación SNR
    python ringdown_bayesian_v3.py --data l1.npy h1.npy --multiweight snr

    # Forzar overtones activos
    python ringdown_bayesian_v3.py --data l1.npy --overtones on

    # Configurar bandas de auto-detección
    python ringdown_bayesian_v3.py --data l1.npy --auto-fmin 15 --auto-fmax 600
        """
    )
    
    parser.add_argument('--data', nargs='+', required=True, help='Archivos .npy')
    parser.add_argument('--detectors', nargs='+', default=None)
    parser.add_argument('--fs', type=float, default=4096.0)
    parser.add_argument('--redshift', type=float, default=None)
    parser.add_argument('--category', choices=['light', 'stellar', 'imbh', 'heavy'], default=None,
                       help='Forzar categoría (si no, auto-detecta)')
    parser.add_argument('--outdir', default='ringdown_output')
    
    # v3: Nuevos flags
    parser.add_argument('--multiweight', choices=['equal', 'snr', 'sigma'], default='equal',
                       help='Ponderación multi-detector: equal (default), snr, sigma')
    parser.add_argument('--overtones', choices=['on', 'off', 'auto'], default='auto',
                       help='Control de overtones: on, off, auto (usa config de categoría)')
    parser.add_argument('--auto-fmin', type=float, default=20.0,
                       help='Frecuencia mínima para auto-detección (default: 20 Hz)')
    parser.add_argument('--auto-fmax', type=float, default=500.0,
                       help='Frecuencia máxima para auto-detección (default: 500 Hz)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RINGDOWN BAYESIANO v3 - CLASIFICACIÓN AUTOMÁTICA ROBUSTA")
    print("="*60)
    print(f"   Auto-detección: [{args.auto_fmin}, {args.auto_fmax}] Hz")
    print(f"   Multiweight: {args.multiweight}")
    print(f"   Overtones: {args.overtones}")
    
    # Mapear categoría
    category_map = {
        'light': EventCategory.LIGHT,
        'stellar': EventCategory.STELLAR,
        'imbh': EventCategory.INTERMEDIATE,
        'heavy': EventCategory.HEAVY
    }
    forced_cat = category_map.get(args.category) if args.category else None
    
    analysis = RingdownAnalysis(
        data_files=args.data,
        detector_names=args.detectors,
        fs=args.fs,
        redshift=args.redshift,
        category=forced_cat,
        outdir=args.outdir,
        multiweight=args.multiweight,
        overtones=args.overtones,
        auto_fmin=args.auto_fmin,
        auto_fmax=args.auto_fmax
    )
    
    results = analysis.analyze()
    analysis.print_results(results)
    analysis.save_results(results)
    
    print("\n✅ Completado")


if __name__ == "__main__":
    main()
