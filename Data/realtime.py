import numpy as np
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
import csv
from datetime import datetime
import os
import smtplib
from email.message import EmailMessage
import threading
import time
from collections import deque
from typing import Optional

# Monitor runtime state
monitor_thread: Optional[threading.Thread] = None
monitor_stop_event: Optional[threading.Event] = None
monitor_lock = threading.Lock()
last_detections = deque(maxlen=200)  # store recent detections
monitor_settings = {
    'last_email_sent': 0.0,
    'last_email_attempt': 0.0,
    'consecutive_failures': 0
}

class AttentionLayer(layers.Layer):
    def call(self, inputs):
        attention = tf.nn.tanh(inputs)
        attention = tf.nn.softmax(attention, axis=1)
        output = tf.reduce_sum(inputs * attention, axis=1)
        return output

MODEL_FILE = "babycry_model_mel.h5"
LABELS_FILE = "label_classes.npy"

model = load_model(MODEL_FILE, compile=False,
                   custom_objects={'AttentionLayer': AttentionLayer})
labels = np.load(LABELS_FILE, allow_pickle=True)

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS = 128

LOG_FILE = "cry_events_log_realtime.csv"

GUIDANCE = {
    "hungry": {
        "advice": (
            "Offer a feed if it is close to or past feeding time. "
            "Look for hunger cues like rooting, sucking on hands, and turning toward the breast or bottle."
        ),
        "doctor": (
            "Consult a doctor if baby refuses feeds repeatedly, has poor weight gain, "
            "vomits forcefully, or seems unusually sleepy or weak."
        )
    },
    "tired": {
        "advice": (
            "Create a calm environment, dim lights, reduce noise, and try gentle rocking or swaddling. "
            "Watch for sleep cues like yawning, rubbing eyes, or looking away."
        ),
        "doctor": (
            "See a doctor if baby is very hard to wake, shows low activity, or has changes in breathing or color."
        )
    },
    "belly_pain": {
        "advice": (
            "Try burping, gentle tummy-down massage, or placing baby tummy-down across your lap and patting the back. "
            "Check for gas, constipation, or hard belly."
        ),
        "doctor": (
            "Seek medical help if belly is very hard or swollen, baby cries in pain when touched, "
            "vomits repeatedly, has blood in stool, or high fever."
        )
    },
    "burping": {
        "advice": (
            "Hold baby upright against your shoulder or sitting on your lap and gently pat or rub the back. "
            "Burp during and after feeds to release trapped air."
        ),
        "doctor": (
            "Consult a doctor if baby always seems uncomfortable after feeds, arches back in pain, "
            "or has frequent choking or coughing with feeds."
        )
    },
    "cold_hot": {
        "advice": (
            "Check baby’s neck or chest with your hand, not just hands and feet. "
            "Add or remove a thin layer of clothing or adjust room temperature."
        ),
        "doctor": (
            "Call a doctor urgently if baby has fever (≥38°C), feels very cold, shivers, has mottled or bluish skin, "
            "or breathing looks abnormal."
        )
    },
    "discomfort": {
        "advice": (
            "Check diaper, clothing tags, tight swaddling, or uncomfortable positions. "
            "Offer cuddling, gentle rocking, or a change of environment."
        ),
        "doctor": (
            "Consult a doctor if baby remains very irritable despite comfort, has rash, swelling, "
            "or any new symptom you are worried about."
        )
    },
    "lonely": {
        "advice": (
            "Hold your baby, offer skin-to-skin contact, talk or sing softly, and maintain eye contact. "
            "Sometimes babies just need closeness and reassurance."
        ),
        "doctor": (
            "If crying becomes unusually intense, prolonged, or baby shows other signs of illness, talk to a doctor."
        )
    },
    "scared": {
        "advice": (
            "Comfort baby quickly, hold them close, speak softly, and reduce sudden loud noises or bright lights. "
            "Stay calm so baby can settle more easily."
        ),
        "doctor": (
            "Seek medical help if baby’s cry is suddenly very high-pitched, they appear in pain, "
            "or you suspect injury or illness."
        )
    },
    "silence": {
        "advice": (
            "Baby seems calm/quiet in this segment. Continue routine care, feeding on schedule, "
            "and safe sleep practices."
        ),
        "doctor": (
            "If baby becomes unusually quiet, floppy, very hard to wake, or breathing changes, seek urgent care."
        )
    }
}

def log_event_realtime(pred_label, confidence):
    now = datetime.now()
    date_str = now.date().isoformat()
    time_str = now.time().strftime("%H:%M:%S")
    hour_str = now.strftime("%H")
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "date", "time", "hour",
                "source", "predicted_label", "confidence"
            ])
        writer.writerow([
            date_str,
            time_str,
            hour_str,
            "microphone",
            pred_label,
            f"{confidence:.4f}"
        ])

def extract_log_mel_spectrogram(signal):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec.T

def extract_features_from_signal(signal):
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        pad = SAMPLES_PER_FILE - len(signal)
        signal = np.pad(signal, (0, pad))

    features = extract_log_mel_spectrogram(signal)
    max_len = model.input_shape[1]
    if features.shape[0] < max_len:
        features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)))
    else:
        features = features[:max_len, :]
    features = features[np.newaxis, :, :, np.newaxis]
    return features


def _read_latest_csv_row(file_path=LOG_FILE):
    """Return the last non-empty row from the log CSV as dict or None if not found."""
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            if any(row.values()):
                last = row
        return last


def _send_email_message(subject: str, body: str, to_email: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None):
    """Send an email and return (True, None) on success or (False, error_message) on failure.

    Falls back to environment variables when args are None.
    """
    if to_email is None:
        to_email = os.environ.get('CRY_TO_EMAIL')
    if from_email is None:
        from_email = os.environ.get('CRY_EMAIL')
    if app_password is None:
        app_password = os.environ.get('CRY_APP_PASSWORD')

    if not all([to_email, from_email, app_password]):
        return False, 'missing credentials or recipient'

    msg = EmailMessage()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, app_password)
            server.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


def send_latest_log_as_email(num_lines=10, to_email: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None):
    """Send the last `num_lines` rows of the CSV as plain text email. Returns (True, None) or (False, error).

    Useful for the manual 'send latest' action.
    """
    if not os.path.isfile(LOG_FILE):
        return False, f'{LOG_FILE} not found'
    header = None
    rows = []
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        try:
            header = next(r)
        except StopIteration:
            return False, 'log file empty'
        dq = deque(maxlen=num_lines)
        for row in r:
            dq.append(row)
        rows = list(dq)

    lines = [', '.join(header)] + [', '.join(r) for r in rows]
    body = 'Latest realtime cry events:\n\n' + '\n'.join(lines)
    subject = 'Latest baby cry events (realtime)'
    return _send_email_message(subject, body, to_email=to_email, from_email=from_email, app_password=app_password)


def report_email(pred_label, advice, doctor, to_email: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None):
    """Build and send a detection report using the safe wrapper.

    Returns (True, None) on success or (False, err) on failure and updates monitor_settings counters.
    """
    label_str = str(pred_label)
    subject = 'Baby Cry Alert'
    body = (
        f"Detected baby cry type: {label_str}\n\n"
        f"Suggested parental guidance (not medical advice):\n- What you can try now: {advice}\n- When to consult a doctor: {doctor}\n\n"
        "This is an automated message from the Baby Cry Detection System."
    )

    # Call the generic sender and update monitor settings
    ok, err = _send_email_message(subject, body, to_email=to_email, from_email=from_email, app_password=app_password)
    now_ts = time.time()
    if ok:
        monitor_settings['last_email_sent'] = now_ts
        monitor_settings['consecutive_failures'] = 0
        monitor_settings['last_email_attempt'] = now_ts
    else:
        # record failed attempt and increment failure count
        monitor_settings['last_email_attempt'] = now_ts
        monitor_settings['consecutive_failures'] = monitor_settings.get('consecutive_failures', 0) + 1
        print(f"Email send failed (attempts={monitor_settings['consecutive_failures']}): {err}")
    return ok, err
    
def run_realtime_monitor(stop_event: Optional[threading.Event]=None, email_on_detect: bool=False, email_recipient: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None, throttle_seconds: int=60):
    """Run the real-time baby cry monitor loop.

    If `stop_event` is provided, the loop exits when stop_event.is_set() is True.
    If `email_on_detect` is True, an email is sent for detections (respecting `throttle_seconds`).
    """
    print("Starting real-time baby cry monitoring...")
    print(f"Sampling from microphone at {SAMPLE_RATE} Hz, window = {DURATION} seconds.")
    print("Press Ctrl+C to stop.\n")

    try:
        last_sent = monitor_settings.get('last_email_sent', 0.0)
        while True:
            if stop_event is not None and stop_event.is_set():
                print("Stop event set — exiting monitor loop.")
                break

            print("Listening...")
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype='float32')
            sd.wait()

            signal = audio.flatten()
            X_new = extract_features_from_signal(signal)

            pred = model.predict(X_new)
            idx = int(np.argmax(pred))
            pred_label = labels[idx]
            confidence = float(pred[0, idx])

            now = datetime.now()
            print(f"\nDetected: {pred_label} (confidence: {confidence:.2f})")

            # Log and keep in memory
            log_event_realtime(str(pred_label), confidence)
            guidance = GUIDANCE.get(str(pred_label), None)
            entry = {
                'date': now.date().isoformat(),
                'time': now.time().strftime('%H:%M:%S'),
                'label': str(pred_label),
                'confidence': f"{confidence:.4f}",
                'guidance': guidance
            }
            last_detections.append(entry)

            # Email-on-detect (with throttle + backoff on failures)
            if email_on_detect and (email_recipient or os.environ.get('CRY_TO_EMAIL')):
                recipient = email_recipient or os.environ.get('CRY_TO_EMAIL')
                now_ts = time.time()

                last_success = monitor_settings.get('last_email_sent', 0.0)
                failures = monitor_settings.get('consecutive_failures', 0)

                # Compute effective throttle: base throttle_seconds multiplied by backoff factor
                backoff_factor = min(2 ** failures, 16)
                effective_throttle = throttle_seconds * backoff_factor

                if now_ts - last_success >= effective_throttle:
                    # send and let report_email update monitor_settings
                    ok, err = report_email(pred_label, guidance['advice'] if guidance else '', guidance['doctor'] if guidance else '', to_email=recipient, from_email=from_email, app_password=app_password)
                    if ok:
                        last_sent = monitor_settings.get('last_email_sent', now_ts)
                        print('✅ Mail sent for latest detection')
                    else:
                        print(f"❌ Failed to send detection email: {err}")
                else:
                    remaining = int(effective_throttle - (now_ts - last_success))
                    print(f"Email suppressed to avoid spamming/backoff; will allow in {remaining}s")

            if guidance:
                print("Suggested parental guidance (not medical advice):")
                print(f"- What you can try now: {guidance['advice']}")
                print(f"- When to consult a doctor: {guidance['doctor']}\n")
            else:
                print("No specific guidance available for this label yet.\n")

    except KeyboardInterrupt:
        print("\nReal-time monitoring stopped by user.")
    finally:
        # Make sure we note monitor state
        monitor_settings['last_email_sent'] = last_sent
        print('Monitor loop exited.')

def start_monitor(email_on_detect: bool=False, email_recipient: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None, throttle_seconds: int=60):
    """Start the monitor in a background thread. Returns True if started, False if already running."""
    global monitor_thread, monitor_stop_event
    with monitor_lock:
        if monitor_thread and monitor_thread.is_alive():
            return False
        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=run_realtime_monitor,
            kwargs={
                'stop_event': monitor_stop_event,
                'email_on_detect': email_on_detect,
                'email_recipient': email_recipient,
                'from_email': from_email,
                'app_password': app_password,
                'throttle_seconds': throttle_seconds
            },
            daemon=True
        )
        monitor_thread.start()
        return True


def stop_monitor(timeout: float=5.0):
    """Signal the monitor to stop and wait up to `timeout` seconds. Returns True if stopped."""
    global monitor_thread, monitor_stop_event
    with monitor_lock:
        if not monitor_thread or not monitor_thread.is_alive():
            return False
        monitor_stop_event.set()
        monitor_thread.join(timeout)
        alive = monitor_thread.is_alive()
        if not alive:
            monitor_thread = None
            monitor_stop_event = None
        return not alive


def is_monitor_running():
    global monitor_thread
    return bool(monitor_thread and monitor_thread.is_alive())


def get_last_detections(n: int = 10):
    return list(last_detections)[-n:]


if __name__ == "__main__":
    import sounddevice as sd
    run_realtime_monitor()
