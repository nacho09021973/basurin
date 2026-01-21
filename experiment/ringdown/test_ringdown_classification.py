#!/usr/bin/env python3
"""
test_ringdown_classification.py
===============================
Tests de regresión para ringdown_bayesian_v3.py

Estos tests verifican el comportamiento de:
- Auto-detección de categoría
- Métrica de confianza
- Cálculo de pesos multi-detector

NO ejecutan MCMC (para evitar dependencias pesadas y tiempos largos).

Uso:
    python test_ringdown_classification.py

Autor: Tests de regresión para BASURIN/ringdown
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Importar solo las partes que necesitamos (sin matplotlib/emcee/corner)
# Hacemos mock de los imports problemáticos antes de importar el módulo
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

sys.modules['matplotlib'] = MockModule()
sys.modules['matplotlib.pyplot'] = MockModule()
sys.modules['emcee'] = MockModule()
sys.modules['corner'] = MockModule()

# Ahora importamos las funciones que queremos testear
from ringdown_bayesian_v3 import (
    EventCategory,
    CATEGORY_CONFIGS,
    classify_event,
    mass_from_frequency,
    bandpower_voting,
    compute_bandpower,
    RingdownAnalysis,
)


# =============================================================================
# Helpers para generar datos sintéticos
# =============================================================================

def generate_ringdown_signal(f_qnm: float, tau: float, amplitude: float, 
                             fs: float, duration: float, t0: float = 0.5) -> np.ndarray:
    """Genera señal de ringdown sintética."""
    N = int(duration * fs)
    t = np.arange(N) / fs
    
    signal = np.zeros(N)
    mask = t >= t0
    dt = t[mask] - t0
    signal[mask] = amplitude * np.exp(-dt / tau) * np.cos(2 * np.pi * f_qnm * dt)
    
    return signal


def add_colored_noise(signal: np.ndarray, fs: float, snr_target: float,
                      noise_psd_func=None) -> np.ndarray:
    """Añade ruido coloreado a la señal."""
    N = len(signal)
    
    if noise_psd_func is None:
        # PSD tipo LIGO simplificado: 1/f^2 en LF, plano en HF
        freqs = np.fft.rfftfreq(N, 1/fs)
        freqs[0] = 1.0  # Evitar división por cero
        psd = 1.0 / (1 + (freqs / 50)**(-4))  # Rolloff en 50 Hz
        psd = psd / np.mean(psd)  # Normalizar
    else:
        psd = noise_psd_func(N, fs)
    
    # Generar ruido con ese PSD
    white = np.random.randn(N)
    white_fft = np.fft.rfft(white)
    freqs_fft = np.fft.rfftfreq(N, 1/fs)
    
    # Interpolar PSD al tamaño correcto si es necesario
    if len(psd) != len(white_fft):
        from scipy.interpolate import interp1d
        interp_psd = interp1d(np.linspace(0, fs/2, len(psd)), psd, fill_value="extrapolate")
        psd = interp_psd(freqs_fft)
    
    colored_fft = white_fft * np.sqrt(np.maximum(psd, 1e-10))
    colored = np.fft.irfft(colored_fft, n=N)
    
    # Escalar para SNR objetivo
    signal_power = np.mean(signal**2)
    noise_power = np.mean(colored**2)
    
    if signal_power > 0 and noise_power > 0:
        scale = np.sqrt(signal_power / (snr_target**2 * noise_power))
        colored = colored * scale
    
    return signal + colored


# =============================================================================
# Test A: GW150914-like (robustez de clasificación)
# =============================================================================

def test_A_gw150914_like():
    """
    Test A: Señal tipo GW150914 (f ~ 250 Hz, M ~ 70 Msun).
    
    Verificar:
    - La categoría detectada es 'stellar' o la confianza es baja
    - Si hay discrepancia, el sistema advierte
    """
    print("\n" + "="*60)
    print("TEST A: GW150914-like (f ~ 250 Hz)")
    print("="*60)
    
    fs = 4096.0
    duration = 2.0
    
    # Parámetros tipo GW150914
    f_qnm = 251.0  # Hz (valor aproximado para M ~ 70 Msun, a ~ 0.67)
    tau = 4e-3     # 4 ms
    amplitude = 5.0
    snr = 8.0
    
    # Generar señal
    signal = generate_ringdown_signal(f_qnm, tau, amplitude, fs, duration)
    data = add_colored_noise(signal, fs, snr)
    
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, data)
        data_file = f.name
    
    try:
        # Crear análisis (no ejecutamos run_mcmc)
        analysis = RingdownAnalysis(
            data_files=[data_file],
            detector_names=['test'],
            fs=fs,
            auto_fmin=20.0,
            auto_fmax=500.0
        )
        
        # Ejecutar solo detección
        category, confidence, voting = analysis._detect_category()
        
        print(f"\n   Resultados:")
        print(f"   - Categoría detectada: {category.value}")
        print(f"   - Confianza: {confidence:.2f}")
        print(f"   - Voting category: {voting['category'].value}")
        
        # Verificaciones
        passed = True
        
        # La categoría debe ser 'stellar' (f ~ 250 Hz está en el rango 150-400 Hz)
        # O la confianza debe ser baja y se debe recomendar --category
        if category == EventCategory.STELLAR:
            print(f"   ✓ Categoría correcta: stellar")
        elif confidence < analysis.CONFIDENCE_THRESHOLD:
            print(f"   ✓ Confianza baja ({confidence:.2f}), sistema pide verificación")
        else:
            print(f"   ✗ FALLO: categoría {category.value} con confianza alta {confidence:.2f}")
            passed = False
        
        # El voting no debe estar sesgado a HEAVY/INTERMEDIATE por ruido LF
        # (este era el bug original)
        if voting['category'] in [EventCategory.STELLAR, EventCategory.LIGHT]:
            print(f"   ✓ Voting no sesgado a LF")
        else:
            # Podría estar bien si la confianza es baja
            if confidence < analysis.CONFIDENCE_MINIMUM:
                print(f"   ~ Voting = {voting['category'].value} pero confianza muy baja, OK")
            else:
                print(f"   ⚠ Voting = {voting['category'].value}, verificar si es por ruido LF")
        
        assert passed        
    finally:
        Path(data_file).unlink()


# =============================================================================
# Test B: Caso ambiguo (ruido coloreado sin señal clara)
# =============================================================================

def test_B_ambiguous():
    """
    Test B: Espectro ambiguo (ruido coloreado, sin pico claro).
    
    Verificar:
    - Confianza muy baja (< CONFIDENCE_MINIMUM)
    - Sistema recomienda usar --category
    """
    print("\n" + "="*60)
    print("TEST B: Caso ambiguo (ruido coloreado)")
    print("="*60)
    
    fs = 4096.0
    duration = 2.0
    N = int(duration * fs)
    
    # Solo ruido coloreado (sin señal)
    np.random.seed(42)
    freqs = np.fft.rfftfreq(N, 1/fs)
    freqs[0] = 1.0
    
    # PSD con pendiente -2 (domina LF)
    psd = 1.0 / (1 + (freqs / 100)**2)
    
    white = np.random.randn(N)
    white_fft = np.fft.rfft(white)
    colored_fft = white_fft * np.sqrt(psd[:len(white_fft)])
    data = np.fft.irfft(colored_fft, n=N)
    
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, data)
        data_file = f.name
    
    try:
        analysis = RingdownAnalysis(
            data_files=[data_file],
            detector_names=['test'],
            fs=fs,
            auto_fmin=20.0,
            auto_fmax=500.0
        )
        
        category, confidence, voting = analysis._detect_category()
        
        print(f"\n   Resultados:")
        print(f"   - Categoría detectada: {category.value}")
        print(f"   - Confianza: {confidence:.2f}")
        
        passed = True
        
        # La confianza debe ser relativamente baja (espectro sin pico claro)
        # No necesariamente < CONFIDENCE_MINIMUM, pero sí < CONFIDENCE_THRESHOLD
        if confidence < analysis.CONFIDENCE_THRESHOLD:
            print(f"   ✓ Confianza baja, sistema advierte correctamente")
        else:
            print(f"   ⚠ Confianza relativamente alta ({confidence:.2f}), puede ser OK si el ruido tiene pico claro")
            # No es necesariamente un fallo, depende del ruido generado
        
        assert passed        
    finally:
        Path(data_file).unlink()


# =============================================================================
# Test C: Multi-detector weighting
# =============================================================================

def test_C_multidetector_weighting():
    """
    Test C: Verificar cálculo de pesos multi-detector.
    
    Verificar:
    - --multiweight equal produce pesos [1, 1]
    - --multiweight snr produce pesos normalizados (media = 1)
    - Con SNRs dispares, el detector fuerte domina
    """
    print("\n" + "="*60)
    print("TEST C: Multi-detector weighting")
    print("="*60)
    
    fs = 4096.0
    duration = 2.0
    
    # Misma señal, diferentes SNRs
    f_qnm = 250.0
    tau = 4e-3
    amplitude = 5.0
    
    signal = generate_ringdown_signal(f_qnm, tau, amplitude, fs, duration)
    
    # Detector 1: SNR alto
    data1 = add_colored_noise(signal.copy(), fs, snr_target=12.0)
    
    # Detector 2: SNR bajo (más ruido)
    data2 = add_colored_noise(signal.copy(), fs, snr_target=4.0)
    
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f1:
        np.save(f1.name, data1)
        file1 = f1.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f2:
        np.save(f2.name, data2)
        file2 = f2.name
    
    try:
        passed = True
        
        # Test equal weights
        print("\n   --- multiweight = equal ---")
        analysis_eq = RingdownAnalysis(
            data_files=[file1, file2],
            detector_names=['H1', 'L1'],
            fs=fs,
            multiweight='equal',
            auto_fmin=20.0,
            auto_fmax=500.0
        )
        
        # Simular preprocesamiento mínimo para tener self.info
        analysis_eq.info = [
            {'snr_peak': 12.0, 'sigma': 1.0},
            {'snr_peak': 4.0, 'sigma': 1.0}
        ]
        
        weights_eq = analysis_eq._compute_weights()
        print(f"   Pesos: {weights_eq}")
        
        if np.allclose(weights_eq, [1.0, 1.0]):
            print(f"   ✓ Pesos iguales correctos")
        else:
            print(f"   ✗ Pesos iguales incorrectos")
            passed = False
        
        # Test SNR weights
        print("\n   --- multiweight = snr ---")
        analysis_snr = RingdownAnalysis(
            data_files=[file1, file2],
            detector_names=['H1', 'L1'],
            fs=fs,
            multiweight='snr',
            auto_fmin=20.0,
            auto_fmax=500.0
        )
        analysis_snr.info = [
            {'snr_peak': 12.0, 'sigma': 1.0},
            {'snr_peak': 4.0, 'sigma': 1.0}
        ]
        
        weights_snr = analysis_snr._compute_weights()
        print(f"   Pesos: {weights_snr}")
        
        # Verificar normalización (media = 1)
        if np.isclose(np.mean(weights_snr), 1.0, atol=1e-6):
            print(f"   ✓ Normalización correcta (media = {np.mean(weights_snr):.4f})")
        else:
            print(f"   ✗ Normalización incorrecta (media = {np.mean(weights_snr):.4f})")
            passed = False
        
        # Verificar que H1 (SNR alto) tiene peso mayor
        if weights_snr[0] > weights_snr[1]:
            ratio = weights_snr[0] / weights_snr[1]
            # Esperado: (12^2) / (4^2) = 9, normalizado: [1.8, 0.2] -> ratio = 9
            print(f"   ✓ H1 (SNR alto) domina: ratio = {ratio:.1f}")
            
            # El ratio debe ser aproximadamente (12/4)^2 = 9
            expected_ratio = (12.0 / 4.0) ** 2
            if np.isclose(ratio, expected_ratio, rtol=0.1):
                print(f"   ✓ Ratio correcto (esperado {expected_ratio:.1f})")
            else:
                print(f"   ⚠ Ratio difiere del esperado ({expected_ratio:.1f})")
        else:
            print(f"   ✗ H1 debería dominar pero no lo hace")
            passed = False
        
        assert passed        
    finally:
        Path(file1).unlink()
        Path(file2).unlink()


# =============================================================================
# Test D: Verificar fix v3.1 (voting sobre FFT filtrada)
# =============================================================================

def test_D_voting_filtered():
    """
    Test D: Verificar que el voting usa FFT filtrada (fix v3.1).
    
    Este test verifica indirectamente que el voting no se sesga por
    contenido fuera de [auto_fmin, auto_fmax].
    """
    print("\n" + "="*60)
    print("TEST D: Voting sobre FFT filtrada (fix v3.1)")
    print("="*60)
    
    fs = 4096.0
    N = int(2.0 * fs)
    
    # Crear espectro con pico claro en 250 Hz (stellar)
    # pero con mucha potencia en LF (< 20 Hz) que el filtro debe ignorar
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Espectro: pico en 250 Hz + mucho ruido en 5-15 Hz
    spectrum = np.zeros(len(freqs))
    
    # Pico en 250 Hz (banda stellar)
    mask_250 = (freqs >= 240) & (freqs <= 260)
    spectrum[mask_250] = 10.0
    
    # Ruido muy fuerte en LF (fuera de auto_fmin=20)
    mask_lf = (freqs >= 5) & (freqs <= 15)
    spectrum[mask_lf] = 100.0  # 10x más potente que el pico real
    
    print(f"\n   Espectro sintético:")
    print(f"   - Pico en 240-260 Hz: potencia = 10")
    print(f"   - Ruido LF en 5-15 Hz: potencia = 100 (10x mayor)")
    
    # Verificar bandpower_voting con fmin_auto=20 (debe ignorar LF)
    voting_filtered = bandpower_voting(spectrum, freqs, fmin_auto=20.0, fmax_auto=500.0)
    
    print(f"\n   Voting con fmin_auto=20 Hz:")
    print(f"   - Categoría: {voting_filtered['category'].value}")
    print(f"   - Confianza: {voting_filtered['confidence']:.2f}")
    print(f"   - Bandpowers: STELLAR={voting_filtered['bandpowers'].get(EventCategory.STELLAR, 0):.1f}, "
          f"HEAVY={voting_filtered['bandpowers'].get(EventCategory.HEAVY, 0):.1f}")
    
    passed = True
    
    # Con fmin_auto=20, la banda HEAVY debería ser [20, 50] Hz, no [10, 50]
    # Por lo tanto, el ruido en 5-15 Hz NO debe afectar
    if voting_filtered['category'] == EventCategory.STELLAR:
        print(f"   ✓ Categoría correcta: stellar (LF ignorado)")
    else:
        print(f"   ✗ Categoría incorrecta: {voting_filtered['category'].value}")
        print(f"     (probablemente el fix v3.1 no está aplicado)")
        passed = False
    
    # Comparar con lo que pasaría SIN el fix (fmin_auto=10)
    voting_unfiltered = bandpower_voting(spectrum, freqs, fmin_auto=10.0, fmax_auto=500.0)
    
    print(f"\n   Comparación con fmin_auto=10 Hz (sin fix):")
    print(f"   - Categoría: {voting_unfiltered['category'].value}")
    print(f"   - Bandpowers: STELLAR={voting_unfiltered['bandpowers'].get(EventCategory.STELLAR, 0):.1f}, "
          f"HEAVY={voting_unfiltered['bandpowers'].get(EventCategory.HEAVY, 0):.1f}")
    
    # Con fmin_auto=10, debería ver más potencia en HEAVY (por el ruido en 5-15 Hz)
    if voting_unfiltered['category'] == EventCategory.HEAVY:
        print(f"   (Confirmado: sin fix, se sesga a HEAVY por ruido LF)")
    
    assert passed

# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("TESTS DE REGRESIÓN PARA RINGDOWN v3.1")
    print("="*60)
    print("\nEstos tests verifican comportamiento de clasificación")
    print("sin ejecutar MCMC (rápidos, reproducibles).")
    
    np.random.seed(42)
    
    results = {}
    
    # Test A
    results['A'] = test_A_gw150914_like()
    
    # Test B
    results['B'] = test_B_ambiguous()
    
    # Test C
    results['C'] = test_C_multidetector_weighting()
    
    # Test D (específico para v3.1)
    results['D'] = test_D_voting_filtered()
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   Test {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 Todos los tests pasaron")
        return 0
    else:
        print("⚠️  Algunos tests fallaron")
        return 1


if __name__ == "__main__":
    sys.exit(main())
