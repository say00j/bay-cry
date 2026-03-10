"""
audio_preprocessing.py
=======================
Domain-adaptation audio preprocessing for phone-speaker inference.

The model was trained on clean, studio-quality baby cry recordings.
When the system captures audio that passes through a phone speaker and then
a microphone, the signal has:
  - Narrower frequency response (phone speaker ~200 Hz – 8 kHz)
  - Added reverb / room reflections
  - Codec compression artifacts
  - Different RMS level

This module applies lightweight signal processing steps at INFERENCE time to
make live audio look more like clean training data, BEFORE feature extraction.
It also exports augmentation helpers for retraining with simulated distortion.

No extra pip packages are required beyond what is already installed
(numpy, scipy, librosa).
"""

import numpy as np
from scipy.signal import butter, sosfilt, lfilter
import librosa


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000  # Hz – must match training


# ---------------------------------------------------------------------------
# 1. PRE-EMPHASIS FILTER
#    Boosts high frequencies that phone speakers attenuate.
#    Commonly used in speech/audio preprocessing.
# ---------------------------------------------------------------------------
def apply_pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply a high-frequency pre-emphasis filter."""
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


# ---------------------------------------------------------------------------
# 2. BANDPASS FILTERING
#    Phone speakers typically pass 200 Hz – 8 kHz.
#    We band-limit the signal to that range to suppress rumble & alias noise
#    that was NOT present in training data.
# ---------------------------------------------------------------------------
def apply_bandpass(signal: np.ndarray, sr: int = SAMPLE_RATE,
                   low_hz: float = 150.0, high_hz: float = 7500.0,
                   order: int = 4) -> np.ndarray:
    """Band-pass filter to strip frequencies outside phone speaker range."""
    nyq = sr / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. SPECTRAL EQUALISATION (Wiener-style tilt correction)
#    Estimate the long-term average spectrum of the signal; divide it out;
#    then re-weight toward a "flat" target to undo phone-speaker coloration.
# ---------------------------------------------------------------------------
def apply_spectral_eq(signal: np.ndarray, sr: int = SAMPLE_RATE,
                      n_fft: int = 512, alpha: float = 0.8) -> np.ndarray:
    """
    Flatten the long-term spectrum of the signal.
    alpha=1.0 → full whitening; alpha=0.0 → no change.
    Keep alpha < 1.0 so we don't over-amplify noise.
    """
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft // 2)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # Long-term magnitude envelope (mean across time)
    mean_spectrum = np.mean(magnitude, axis=1, keepdims=True) + 1e-8

    # Partial whitening: divide by envelope raised to power alpha
    magnitude_eq = magnitude / (mean_spectrum ** alpha)

    stft_eq = magnitude_eq * np.exp(1j * phase)
    signal_eq = librosa.istft(stft_eq, hop_length=n_fft // 2, length=len(signal))
    return signal_eq.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. RMS NORMALISATION
#    Normalise loudness to a fixed target to match training data statistics.
# ---------------------------------------------------------------------------
def rms_normalize(signal: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Normalise signal to a fixed RMS level."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-9:
        return signal  # silence – don't divide by zero
    return (signal / rms * target_rms).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. SIMPLE NOISE GATE
#    Suppress very low-energy frames that are just hiss,
#    which confuses the model.
# ---------------------------------------------------------------------------
def apply_noise_gate(signal: np.ndarray, sr: int = SAMPLE_RATE,
                     frame_ms: float = 20.0, threshold_rms: float = 0.005) -> np.ndarray:
    """Zero-out frames below the energy threshold (noise gate)."""
    frame_len = int(sr * frame_ms / 1000)
    output = signal.copy()
    for start in range(0, len(signal), frame_len):
        frame = signal[start:start + frame_len]
        if np.sqrt(np.mean(frame ** 2)) < threshold_rms:
            output[start:start + frame_len] = 0.0
    return output


# ---------------------------------------------------------------------------
# 6. TRIM / PAD SILENCE
#    librosa.effects.trim reduces leading/trailing silence so the model
#    focuses on the actual cry content.
# ---------------------------------------------------------------------------
def trim_silence(signal: np.ndarray, sr: int = SAMPLE_RATE,
                 top_db: float = 30.0) -> np.ndarray:
    """Trim leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    if len(trimmed) < sr * 0.2:
        return signal  # keep original if trim removed too much
    return trimmed.astype(np.float32)


# ---------------------------------------------------------------------------
# 7. MASTER PIPELINE
#    Apply all steps in the recommended order.
# ---------------------------------------------------------------------------
def preprocess_phone_audio(signal: np.ndarray,
                            sr: int = SAMPLE_RATE,
                            use_bandpass: bool = True,
                            use_spectral_eq: bool = True,
                            use_pre_emphasis: bool = True,
                            use_noise_gate: bool = True,
                            use_rms_norm: bool = True,
                            use_trim: bool = False,  # off by default – realtime chunks don't need it
                            target_rms: float = 0.05,
                            eq_alpha: float = 0.6) -> np.ndarray:
    """
    Full domain-adaptation pipeline for phone-speaker audio.

    Call this BEFORE extract_features_from_signal() in realtime_pytorch.py.

    Parameters
    ----------
    signal      : raw float32 audio array
    sr          : sample rate (default 16000)
    use_bandpass: strip rumble & alias noise outside phone speaker range
    use_spectral_eq: flatten long-term spectrum coloring
    use_pre_emphasis: boost high frequencies
    use_noise_gate: zero out frames below energy threshold
    use_rms_norm: normalise loudness to match training statistics
    use_trim    : trim silence (recommended for file-based; avoid for realtime chunks)
    target_rms  : target RMS level after normalisation
    eq_alpha    : strength of spectral equalisation (0 = off, 1 = full whitening)

    Returns
    -------
    Preprocessed float32 audio array (same length as input unless trim=True)
    """
    if use_bandpass:
        signal = apply_bandpass(signal, sr=sr)

    if use_spectral_eq:
        signal = apply_spectral_eq(signal, sr=sr, alpha=eq_alpha)

    if use_pre_emphasis:
        signal = apply_pre_emphasis(signal)

    if use_noise_gate:
        signal = apply_noise_gate(signal, sr=sr)

    if use_trim:
        signal = trim_silence(signal, sr=sr)

    if use_rms_norm:
        signal = rms_normalize(signal, target_rms=target_rms)

    return signal


# ---------------------------------------------------------------------------
# AUGMENTATION HELPERS  (for retraining)
# ---------------------------------------------------------------------------

def simulate_phone_speaker(signal: np.ndarray, sr: int = SAMPLE_RATE,
                            reverb_amount: float = 0.15) -> np.ndarray:
    """
    Simulate the distortion introduced by a phone speaker:
      1. Band-limit to phone speaker range
      2. Add a short synthetic room reverb
      3. Add mild white noise (simulates room noise)

    Use this during training to augment clean recordings with
    phone-speaker-like distortions.
    """
    # 1. Band-limit
    sig = apply_bandpass(signal, sr=sr, low_hz=200.0, high_hz=7500.0)

    # 2. Lightweight reverb via a simple FIR delay line
    reverb_sig = _add_simple_reverb(sig, sr=sr, decay=reverb_amount)

    # 3. Add mild white noise (SNR ~ 30 dB)
    rms_sig = np.sqrt(np.mean(reverb_sig ** 2)) + 1e-9
    noise = np.random.randn(len(reverb_sig)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    snr_linear = 10 ** (28.0 / 20.0)  # ~28 dB SNR
    noise = noise * (rms_sig / (noise_rms * snr_linear))
    reverb_sig = reverb_sig + noise

    return reverb_sig.astype(np.float32)


def _add_simple_reverb(signal: np.ndarray, sr: int = SAMPLE_RATE,
                        decay: float = 0.2) -> np.ndarray:
    """
    Add a simple comb-filter reverb using a few delay taps.
    This cheaply simulates room reflections without an impulse response file.
    """
    output = signal.copy().astype(np.float64)
    delays_ms = [30, 60, 100, 150]  # ms
    gains = [decay * 0.7, decay * 0.5, decay * 0.3, decay * 0.2]

    for delay_ms, gain in zip(delays_ms, gains):
        delay_samples = int(sr * delay_ms / 1000)
        if delay_samples >= len(signal):
            continue
        delayed = np.zeros_like(output)
        delayed[delay_samples:] = signal[:len(signal) - delay_samples]
        output += gain * delayed

    # Normalise to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1e-9:
        output = output / max_val * np.max(np.abs(signal))
    return output.astype(np.float32)


def random_pitch_shift(signal: np.ndarray, sr: int = SAMPLE_RATE,
                        semitones_range: float = 1.5) -> np.ndarray:
    """Randomly pitch-shift by ±semitones_range semitones."""
    n_steps = np.random.uniform(-semitones_range, semitones_range)
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)


def random_time_stretch(signal: np.ndarray,
                         rate_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """Randomly time-stretch by a factor in rate_range."""
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(signal, rate=rate)


def random_volume_scale(signal: np.ndarray,
                         factor_range: tuple = (0.6, 1.4)) -> np.ndarray:
    """Randomly scale amplitude."""
    factor = np.random.uniform(*factor_range)
    return (signal * factor).clip(-1.0, 1.0).astype(np.float32)


def add_random_noise(signal: np.ndarray, snr_db: float = None) -> np.ndarray:
    """Add white Gaussian noise at a random SNR between 20 dB and 40 dB."""
    if snr_db is None:
        snr_db = np.random.uniform(20.0, 40.0)
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-9
    noise = np.random.randn(len(signal)).astype(np.float32)
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    snr_linear = 10 ** (snr_db / 20.0)
    noise = noise * (rms / (noise_rms * snr_linear))
    return (signal + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("audio_preprocessing.py self-test")
    # Generate a short sine wave at 440 Hz (simulate a cry harmonic)
    t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
    test_signal = 0.3 * np.sin(2 * np.pi * 440 * t)
    print(f"  Input  RMS = {np.sqrt(np.mean(test_signal**2)):.4f}")

    processed = preprocess_phone_audio(test_signal)
    print(f"  Output RMS = {np.sqrt(np.mean(processed**2)):.4f}")

    simulated = simulate_phone_speaker(test_signal)
    print(f"  Simulated phone-speaker RMS = {np.sqrt(np.mean(simulated**2)):.4f}")

    print("  ✓ All preprocessing steps ran without error.")
