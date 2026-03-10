from flask import Flask, render_template, jsonify, request
import os
import realtime

app = Flask(__name__, template_folder='templates', static_folder='static')

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
    app.run(host='0.0.0.0', port=5000, debug=True)
