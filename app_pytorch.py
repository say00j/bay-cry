from flask import Flask, render_template, jsonify, request
import os
import io
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import realtime_pytorch as realtime

app = Flask(__name__, template_folder='Data/templates', static_folder='Data/static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    running = realtime.is_monitor_running()
    last = realtime.get_last_detections(1)
    return jsonify({'running': running, 'last_detection': last[0] if last else None})


@app.route('/logs')
def logs():
    n = int(request.args.get('n', 10))
    return jsonify(realtime.get_last_detections(n))


@app.route('/start', methods=['POST'])
def start():
    data = request.json or {}
    started = realtime.start_monitor(
        email_on_detect=bool(data.get('email_on_detect', False)),
        email_recipient=data.get('recipient') or os.environ.get('CRY_TO_EMAIL'),
        from_email=data.get('from_email') or os.environ.get('CRY_EMAIL'),
        app_password=data.get('app_password') or os.environ.get('CRY_APP_PASSWORD'),
        throttle_seconds=int(data.get('throttle_seconds', 60))
    )
    return jsonify({'started': started})


@app.route('/stop', methods=['POST'])
def stop():
    return jsonify({'stopped': realtime.stop_monitor()})


def _decode_audio(audio_bytes):
    """Decode audio bytes to (signal, sr). Raises on failure."""
    try:
        signal, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
        return signal, sr
    except Exception:
        from scipy.io import wavfile
        sr, signal = wavfile.read(io.BytesIO(audio_bytes))
        signal = signal.astype(np.float32)
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        if signal.max() > 1.0:
            signal = signal / 32768.0
        return signal, sr


def _to_mono_16k(signal, sr):
    """Convert to mono and resample to 16kHz."""
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != realtime.SAMPLE_RATE:
        from math import gcd
        g = gcd(int(sr), realtime.SAMPLE_RATE)
        signal = resample_poly(signal, realtime.SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return signal


def _run_pipeline(signal):
    """
    Run the 5-stage pre-filter + model on a signal.
    Returns (label, confidence, all_probs) where label is the raw model output
    or None if rejected at any stage.
    """
    import torch

    all_probs = {}
    rms = float(np.sqrt(np.mean(signal ** 2)))

    if rms < 0.002:
        print(f"DEBUG: Stage 1 - Energy rejected (RMS={rms:.5f})")
        return None, None, all_probs

    if not realtime.analyze_frequency_characteristics(signal)[0]:
        print("DEBUG: Stage 2 - Frequency rejected")
        return None, None, all_probs

    if not realtime.check_harmonic_structure(signal)[0]:
        print("DEBUG: Stage 3 - Spectral rejected")
        return None, None, all_probs

    if not realtime.detect_cry_rhythm(signal)[0]:
        print("DEBUG: Stage 4 - Temporal rejected")
        return None, None, all_probs

    print("DEBUG: Pre-filter passed -> Stage 5 (model)")

    # Boost quiet audio
    if 0.002 <= rms < 0.08:
        gain = 0.12 / (rms + 1e-6)
        signal = signal * gain
        print(f"DEBUG: Signal boosted {gain:.2f}x")

    X = realtime.extract_features_from_signal(signal)
    with torch.no_grad():
        outputs = realtime.model(X)
        probs   = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
        idx  = idx.item()
        conf = round(conf.item(), 4)

    raw_label = str(realtime.labels[idx])
    all_probs = {str(realtime.labels[i]): round(float(probs[0, i]), 4)
                 for i in range(len(realtime.labels))}

    print(f"DEBUG: Stage 5 - {raw_label}, conf={conf:.4f}")

    if conf >= realtime.CONFIDENCE_THRESHOLD and raw_label not in ('silence', 'non_cry'):
        return raw_label, conf, all_probs

    reason = 'low_confidence' if conf < realtime.CONFIDENCE_THRESHOLD else raw_label
    print(f"DEBUG: Model rejected ({reason})")
    return None, None, all_probs


@app.route('/predict-file', methods=['POST'])
def predict_file():
    """
    File upload endpoint — instant single-shot prediction, no windowing.
    Used when the user uploads a pre-recorded audio file.
    Runs the full pre-filter + model pipeline once and returns the result immediately.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_bytes = request.files['audio'].read()
    if len(audio_bytes) < 100:
        return jsonify({'error': 'Audio too short or empty'}), 400

    try:
        signal, sr = _decode_audio(audio_bytes)
    except Exception as e:
        return jsonify({'error': f'Could not decode audio: {e}'}), 422

    signal = _to_mono_16k(signal, sr)

    try:
        label, confidence, all_probs = _run_pipeline(signal)
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {e}'}), 500

    if label is None:
        return jsonify({
            'label':      'no_baby_detected',
            'confidence': 0.0,
            'all_probs':  all_probs,
            'guidance':   None
        })

    guidance = realtime.GUIDANCE.get(label)

    from datetime import datetime
    now = datetime.now()
    realtime.log_event_realtime(label, confidence)
    realtime.last_detections.append({
        'date':       now.date().isoformat(),
        'time':       now.strftime('%H:%M:%S'),
        'label':      label,
        'confidence': f"{confidence:.4f}",
        'guidance':   guidance
    })

    print(f"DEBUG: File result -> {label} (conf={confidence:.4f})")
    return jsonify({
        'label':      label,
        'confidence': confidence,
        'all_probs':  all_probs,
        'guidance':   guidance
    })


@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """
    Live mic endpoint. Every call = one window slot (5-window aggregation).

    Window logic (WINDOW_SIZE = 5):
      - Every chunk is always counted as a slot whether cry is detected or not.
      - Confident cry found  -> slot = (label, confidence).
      - Rejected / low-conf  -> slot = None.
      - After 5 slots:
          * All None         -> label='no_baby_detected'
          * At least one cry -> sum confidences per class across slots;
            repeated class detections accumulate (stronger signal wins).
      - Slots 1-4 -> HTTP 204 No Content (frontend does nothing).
      - Slot 5    -> HTTP 200 + JSON result.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_bytes = request.files['audio'].read()
    if len(audio_bytes) < 100:
        return jsonify({'error': 'Audio too short or empty'}), 400

    try:
        signal, sr = _decode_audio(audio_bytes)
    except Exception as e:
        return jsonify({'error': f'Could not decode audio: {e}'}), 422

    signal = _to_mono_16k(signal, sr)

    try:
        slot_label, slot_conf, all_probs = _run_pipeline(signal)
    except Exception as e:
        return jsonify({'error': f'Model inference failed: {e}'}), 500

    # Push slot — EVERY window counts (None or cry)
    result = realtime.push_window_slot(slot_label, slot_conf)

    if result is None:
        # Still collecting slots 1-4 — frontend does nothing
        return ('', 204)

    # 5 slots complete — return aggregated result
    final_label = result['label']
    final_conf  = result['confidence']
    guidance    = realtime.GUIDANCE.get(final_label)

    if final_label != 'no_baby_detected':
        realtime.log_event_realtime(final_label, final_conf)
        from datetime import datetime
        now = datetime.now()
        realtime.last_detections.append({
            'date':       now.date().isoformat(),
            'time':       now.strftime('%H:%M:%S'),
            'label':      final_label,
            'confidence': f"{final_conf:.4f}",
            'guidance':   guidance
        })

    print(f"DEBUG: 5-slot result -> {final_label} (accumulated_conf={final_conf:.4f})")
    return jsonify({
        'label':      final_label,
        'confidence': final_conf,
        'all_probs':  all_probs,
        'guidance':   guidance,
        'counts':     result.get('counts', {})
    })


@app.route('/send-report', methods=['POST'])
def send_report():
    data = request.get_json(silent=True) or {}
    recipient    = data.get('recipient')    or os.environ.get('CRY_TO_EMAIL')
    from_email   = data.get('from_email')   or os.environ.get('CRY_EMAIL')
    app_password = data.get('app_password') or os.environ.get('CRY_APP_PASSWORD')
    last = realtime.get_last_detections(1)
    if last:
        entry    = last[0]
        guidance = entry.get('guidance') or {}
        ok, err  = realtime.report_email(entry['label'],
                                         guidance.get('advice', ''), guidance.get('doctor', ''),
                                         to_email=recipient, from_email=from_email, app_password=app_password)
        return jsonify({'sent': ok} if ok else {'sent': False, 'error': err}), (200 if ok else 500)
    ok, err = realtime.send_latest_log_as_email(num_lines=1, to_email=recipient,
                                                 from_email=from_email, app_password=app_password)
    return jsonify({'sent': ok, 'via': 'csv'} if ok else
                   {'sent': False, 'error': err, 'reason': 'no live detection, tried csv'}), (200 if ok else 500)


if __name__ == '__main__':
    print("=" * 70)
    print("Baby Cry Detection System - PyTorch Version")
    print("=" * 70)
    print("Starting Flask web server...")
    print("Access the web interface at: http://localhost:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)