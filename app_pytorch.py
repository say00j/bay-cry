from flask import Flask, render_template, jsonify, request
import os
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import realtime_pytorch as realtime
from collections import deque

app = Flask(__name__, template_folder='Data/templates', static_folder='Data/static')

# Sliding window of recent raw predictions for majority-vote smoothing
_recent_preds = deque(maxlen=3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    running = realtime.is_monitor_running()
    last = realtime.get_last_detections(1)
    return jsonify({
        'running': running,
        'last_detection': last[0] if last else None
    })

@app.route('/logs')
def logs():
    n = int(request.args.get('n', 10))
    return jsonify(realtime.get_last_detections(n))

@app.route('/start', methods=['POST'])
def start():
    data = request.json or {}
    email_on_detect = bool(data.get('email_on_detect', False))
    throttle = int(data.get('throttle_seconds', 60))
    recipient = data.get('recipient') or os.environ.get('CRY_TO_EMAIL')
    from_email = data.get('from_email') or os.environ.get('CRY_EMAIL')
    app_password = data.get('app_password') or os.environ.get('CRY_APP_PASSWORD')
    started = realtime.start_monitor(email_on_detect=email_on_detect, email_recipient=recipient, from_email=from_email, app_password=app_password, throttle_seconds=throttle)
    return jsonify({'started': started})

@app.route('/stop', methods=['POST'])
def stop():
    stopped = realtime.stop_monitor()
    return jsonify({'stopped': stopped})

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """Receive a raw audio blob from the browser mic, run prediction, return JSON."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_bytes = audio_file.read()

    if len(audio_bytes) < 100:
        return jsonify({'error': 'Audio too short or empty'}), 400

    try:
        # Decode audio (WAV, WebM, OGG, etc.) using soundfile
        audio_io = io.BytesIO(audio_bytes)
        signal, sr = sf.read(audio_io, dtype='float32', always_2d=False)
    except Exception:
        # soundfile may fail on WebM/Opus; try scipy wav fallback
        try:
            from scipy.io import wavfile
            audio_io = io.BytesIO(audio_bytes)
            sr, signal = wavfile.read(audio_io)
            signal = signal.astype(np.float32)
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            # Normalize int16 -> float32 [-1, 1]
            if signal.max() > 1.0:
                signal = signal / 32768.0
        except Exception as e2:
            return jsonify({'error': f'Could not decode audio: {str(e2)}'}), 422

    # Convert stereo -> mono
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    # Resample to 16000 Hz if needed
    target_sr = realtime.SAMPLE_RATE  # 16000
    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), target_sr)
        signal = resample_poly(signal, target_sr // g, int(sr) // g).astype(np.float32)

    # ── HYBRID DETECTION PIPELINE: 5-Stage Pre-filtering ──────────────────
    # Stage 1: Energy check - reject silence
    # Stage 2: Frequency analysis - reject if not in baby cry frequency range
    # Stage 3: Spectral pattern matching - reject if harmonic structure doesn't match
    # Stage 4: Temporal pattern analysis - reject if rhythm doesn't match baby cry
    # Stage 5: Model classification - classify baby cry type
    # ──────────────────────────────────────────────────────────────────
    
    # Stage 1: Energy Check
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 0.002:
        print(f"DEBUG: Stage 1 - Energy gate rejected (RMS={rms:.5f})")
        return jsonify({
            'label': 'no_baby_detected',
            'confidence': 1.0,
            'reason': 'silence or extremely low energy',
            'rms': round(rms, 5),
            'guidance': None,
            'all_probs': {}
        })
    
    # Stage 2: Frequency Analysis - reject if not in baby cry frequency range
    freq_is_baby_cry, energy_ratio, dominant_freq_range = realtime.analyze_frequency_characteristics(signal)
    if not freq_is_baby_cry:
        print(f"DEBUG: Stage 2 - Frequency rejection (energy_ratio={energy_ratio:.2f}, dominant={dominant_freq_range})")
        return jsonify({
            'label': 'no_baby_detected',
            'confidence': 1.0,
            'reason': 'frequency_rejection',
            'energy_ratio': round(energy_ratio, 3),
            'dominant_freq_range': dominant_freq_range,
            'guidance': None,
            'all_probs': {}
        })
    
    # Stage 3: Spectral Pattern Matching - reject if harmonic structure doesn't match
    spectral_is_baby_cry, hnr_db, num_peaks = realtime.check_harmonic_structure(signal)
    if not spectral_is_baby_cry:
        print(f"DEBUG: Stage 3 - Spectral rejection (HNR={hnr_db:.2f} dB, peaks={num_peaks})")
        return jsonify({
            'label': 'no_baby_detected',
            'confidence': 1.0,
            'reason': 'spectral_rejection',
            'hnr_db': round(hnr_db, 2),
            'num_peaks': num_peaks,
            'guidance': None,
            'all_probs': {}
        })
    
    # Stage 4: Temporal Pattern Analysis - reject if rhythm doesn't match baby cry
    temporal_is_baby_cry, burst_durations, valid_burst_count = realtime.detect_cry_rhythm(signal)
    if not temporal_is_baby_cry:
        burst_str = ", ".join([f"{d:.2f}s" for d in burst_durations[:5]])
        if len(burst_durations) > 5:
            burst_str += "..."
        print(f"DEBUG: Stage 4 - Temporal rejection (valid_bursts={valid_burst_count}/{len(burst_durations)})")
        return jsonify({
            'label': 'no_baby_detected',
            'confidence': 1.0,
            'reason': 'temporal_rejection',
            'valid_burst_count': valid_burst_count,
            'total_bursts': len(burst_durations),
            'guidance': None,
            'all_probs': {}
        })
    
    print(f"DEBUG: Pre-filtering passed - proceeding to Stage 5 (Model Classification)")
    print(f"  - Energy: RMS={rms:.4f}")
    print(f"  - Frequency: energy_ratio={energy_ratio:.2f}, dominant={dominant_freq_range}")
    print(f"  - Spectral: HNR={hnr_db:.2f} dB, peaks={num_peaks}")
    print(f"  - Temporal: valid_bursts={valid_burst_count}/{len(burst_durations)}")
    
    # Boost quiet audio (Common for phone recordings)
    if 0.002 <= rms < 0.08:
        gain = 0.12 / (rms + 1e-6)
        signal = signal * gain
        print(f"DEBUG: Signal boosted (Gain: {gain:.2f}x, New RMS: {rms*gain:.4f})")
    # ──────────────────────────────────────────────────────────────────

    # Stage 5: Model Classification - classify baby cry type
    try:
        import torch
        X = realtime.extract_features_from_signal(signal)
        with torch.no_grad():
            outputs = realtime.model(X)
            probs = torch.softmax(outputs, dim=1)
            confidence, idx = torch.max(probs, 1)
            idx = idx.item()
            confidence = round(confidence.item(), 4)
        
        raw_label = str(realtime.labels[idx])
        all_probs = {
            str(realtime.labels[i]): round(float(probs[0, i]), 4)
            for i in range(len(realtime.labels))
        }

        # Confidence threshold for model predictions
        CONFIDENCE_THRESHOLD = 0.25
        
        print(f"DEBUG: Stage 5 - Model prediction: {raw_label}, Conf={confidence:.4f}, Thresh={CONFIDENCE_THRESHOLD}")

        if confidence < CONFIDENCE_THRESHOLD or raw_label in ('silence', 'non_cry'):
            pred_label = 'no_baby_detected'
            guidance = None
            reason = 'low_confidence' if confidence < CONFIDENCE_THRESHOLD else raw_label
            print(f"DEBUG: Result=no_baby_detected (Reason: {reason})")
        else:
            pred_label = raw_label
            guidance = realtime.GUIDANCE.get(pred_label, None)
            print(f"DEBUG: Result={pred_label}")

        return jsonify({
            'label': pred_label,
            'confidence': confidence,
            'all_probs': all_probs,
            'guidance': guidance
        })
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500


@app.route('/send-report', methods=['POST'])
def send_report():
    # Accept JSON if provided, otherwise continue with defaults
    data = request.get_json(silent=True) or {}
    recipient = data.get('recipient') or os.environ.get('CRY_TO_EMAIL')
    from_email = data.get('from_email') or os.environ.get('CRY_EMAIL')
    app_password = data.get('app_password') or os.environ.get('CRY_APP_PASSWORD')

    # Prefer in-memory latest detection (live), otherwise fallback to CSV
    last = realtime.get_last_detections(1)
    if last:
        entry = last[0]
        guidance = entry.get('guidance') or {}
        ok, err = realtime.report_email(entry['label'], guidance.get('advice', ''), guidance.get('doctor', ''), to_email=recipient, from_email=from_email, app_password=app_password)
        if ok:
            return jsonify({'sent': True})
        else:
            return jsonify({'sent': False, 'error': err}), 500
    else:
        # fallback to sending last CSV row
        ok, err = realtime.send_latest_log_as_email(num_lines=1, to_email=recipient, from_email=from_email, app_password=app_password)
        if ok:
            return jsonify({'sent': True, 'via': 'csv'})
        else:
            return jsonify({'sent': False, 'error': err, 'reason': 'no live detection, tried csv'}), 500

if __name__ == '__main__':
    print("="*70)
    print("Baby Cry Detection System - PyTorch Version")
    print("="*70)
    print("Starting Flask web server...")
    print("Access the web interface at: http://localhost:5000")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True)
