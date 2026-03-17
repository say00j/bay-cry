import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import csv
from datetime import datetime
import os
import smtplib
from email.message import EmailMessage
import threading
import time
from collections import deque
from typing import Optional

# Domain-adaptation preprocessing for phone-speaker audio
try:
    from audio_preprocessing import preprocess_phone_audio
    PHONE_PREPROCESSING = True
    print("Phone-speaker audio preprocessing loaded (currently DISABLED)")
except ImportError:
    PHONE_PREPROCESSING = False
    print("audio_preprocessing.py not found - skipping domain-adaptation")

# Monitor runtime state
monitor_thread: Optional[threading.Thread] = None
monitor_stop_event: Optional[threading.Event] = None
monitor_lock = threading.Lock()
last_detections = deque(maxlen=200)

# ── 5-window aggregation state ────────────────────────────────────────────────
# Every window slot is always filled — cry class OR None (silence/rejected).
# After 5 slots: accumulate confidence per class, pick highest. All None → no cry.
window_slots: list = []               # list of (label, confidence) or None
window_slots_lock = threading.Lock()
WINDOW_SIZE = 5
CONFIDENCE_THRESHOLD = 0.25
# ─────────────────────────────────────────────────────────────────────────────

monitor_settings = {
    'last_email_sent': 0.0,
    'last_email_attempt': 0.0,
    'consecutive_failures': 0
}


# Attention Layer for LSTM models
class AttentionLayer(nn.Module):
    def forward(self, x):
        attention = torch.tanh(x)
        attention = torch.softmax(attention, dim=1)
        output = torch.sum(x * attention, dim=1)
        return output


# Architecture 1: Deeper CNN with GAP
class CNNBabyCryModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNBabyCryModel, self).__init__()
        self.conv1a = nn.Conv2d(1, 64, 3, padding=1); self.bn1a = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1); self.bn1b = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2); self.dropout1 = nn.Dropout(0.2)
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1); self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1); self.bn2b = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2); self.dropout2 = nn.Dropout(0.2)
        self.conv3a = nn.Conv2d(128, 256, 3, padding=1); self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1); self.bn3b = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2); self.dropout3 = nn.Dropout(0.3)
        self.conv4a = nn.Conv2d(256, 512, 3, padding=1); self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1); self.bn4b = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2); self.dropout4 = nn.Dropout(0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512); self.bn5 = nn.BatchNorm1d(512); self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256); self.bn6 = nn.BatchNorm1d(256); self.dropout6 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 128); self.bn7 = nn.BatchNorm1d(128); self.dropout7 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, num_classes)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1a(self.conv1a(x))); x = self.leaky_relu(self.bn1b(self.conv1b(x))); x = self.pool1(x); x = self.dropout1(x)
        x = self.leaky_relu(self.bn2a(self.conv2a(x))); x = self.leaky_relu(self.bn2b(self.conv2b(x))); x = self.pool2(x); x = self.dropout2(x)
        x = self.leaky_relu(self.bn3a(self.conv3a(x))); x = self.leaky_relu(self.bn3b(self.conv3b(x))); x = self.pool3(x); x = self.dropout3(x)
        x = self.leaky_relu(self.bn4a(self.conv4a(x))); x = self.leaky_relu(self.bn4b(self.conv4b(x))); x = self.pool4(x); x = self.dropout4(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.leaky_relu(self.bn5(self.fc1(x))); x = self.dropout5(x)
        x = self.leaky_relu(self.bn6(self.fc2(x))); x = self.dropout6(x)
        x = self.leaky_relu(self.bn7(self.fc3(x))); x = self.dropout7(x)
        return self.fc4(x)


# Architecture 2: CNN-LSTM with Attention
class LSTMBabyCryModel(nn.Module):
    def __init__(self, num_classes):
        super(LSTMBabyCryModel, self).__init__()
        self.conv1=nn.Conv2d(1,64,3,padding=1);self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,64,3,padding=1);self.bn2=nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(2,2);self.dropout1=nn.Dropout(0.3)
        self.conv3=nn.Conv2d(64,128,3,padding=1);self.bn3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128,128,3,padding=1);self.bn4=nn.BatchNorm2d(128)
        self.pool2=nn.MaxPool2d(2,2);self.dropout2=nn.Dropout(0.3)
        self.conv5=nn.Conv2d(128,256,3,padding=1);self.bn5=nn.BatchNorm2d(256)
        self.pool3=nn.MaxPool2d(2,2);self.dropout3=nn.Dropout(0.4)
        self.lstm1=nn.LSTM(4864, 128, batch_first=True, bidirectional=True)
        self.lstm2=nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.attention=AttentionLayer()
        self.fc1=nn.Linear(128,256);self.bn6=nn.BatchNorm1d(256);self.dropout4=nn.Dropout(0.6)
        self.fc2=nn.Linear(256,128);self.dropout5=nn.Dropout(0.5)
        self.fc3=nn.Linear(128,num_classes);self.relu=nn.ReLU()

    def forward(self, x):
        x=self.relu(self.bn1(self.conv1(x)));x=self.relu(self.bn2(self.conv2(x)));x=self.pool1(x);x=self.dropout1(x)
        x=self.relu(self.bn3(self.conv3(x)));x=self.relu(self.bn4(self.conv4(x)));x=self.pool2(x);x=self.dropout2(x)
        x=self.relu(self.bn5(self.conv5(x)));x=self.pool3(x);x=self.dropout3(x)
        b=x.size(0);x=x.view(b, x.size(1)*x.size(2), x.size(3)).permute(0,2,1)
        x,_=self.lstm1(x);x,_=self.lstm2(x);x=self.attention(x)
        x=self.relu(self.bn6(self.fc1(x)));x=self.dropout4(x);x=self.relu(self.fc2(x));x=self.dropout5(x)
        return self.fc3(x)


# ── Load model and labels ─────────────────────────────────────────────────────
_CANDIDATES = [
    ("best_model_phone_robust_v2.pth",  "label_classes_phone_robust_v2.npy",  "normalization_params_phone_robust_v2.npy"),
    ("best_model_phone_robust.pth",     "label_classes_phone_robust.npy",     "normalization_params_phone_robust.npy"),
    ("best_model_enhanced.pth",         "label_classes_enhanced.npy",         "normalization_params_enhanced.npy"),
    ("best_model_retrained.pth",        "label_classes_retrained.npy",        "normalization_params_retrained.npy"),
    ("best_model_improved_90.pth",      "label_classes_improved_90.npy",      "normalization_params_improved_90.npy"),
]
MODEL_FILE, LABELS_FILE, NORM_PARAMS_FILE = next(
    (m, l, n) for m, l, n in _CANDIDATES if os.path.exists(m)
)
print(f"Using model: {MODEL_FILE}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

labels = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(labels)

if any(k in MODEL_FILE for k in ("improved_90", "full_dataset", "enhanced", "retrained", "phone_robust")):
    print("Detected CNN architecture")
    model = CNNBabyCryModel(num_classes).to(device)
else:
    print("Detected legacy LSTM architecture")
    model = LSTMBabyCryModel(num_classes).to(device)

model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=False))
model.eval()

try:
    norm_params = np.load(NORM_PARAMS_FILE, allow_pickle=True).item()
    GLOBAL_MEAN = norm_params.get('mean', norm_params.get('global_mean'))
    GLOBAL_STD  = norm_params.get('std',  norm_params.get('global_std'))
    IS_ROBUST_NORM = norm_params.get('method') == 'per_sample_robust'
    print(f"Normalization: {'Robust' if IS_ROBUST_NORM else 'Global'} (Mean={GLOBAL_MEAN:.2f})")
except Exception:
    print("Using fallback normalization")
    GLOBAL_MEAN, GLOBAL_STD, IS_ROBUST_NORM = None, None, False

print(f"Model loaded: {num_classes} classes → {labels}")

SAMPLE_RATE      = 16000
DURATION         = 5
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS           = 128
LOG_FILE         = "cry_events_log_realtime.csv"

FREQUENCY_ENERGY_THRESHOLD    = 0.05
HARMONIC_NOISE_RATIO_THRESHOLD = -15.0
BURST_DURATION_MIN = 0.5
BURST_DURATION_MAX = 2.0
PAUSE_DURATION_MIN = 0.2
PAUSE_DURATION_MAX = 1.0

GUIDANCE = {
    "hungry":     {"advice": "Offer a feed if it is close to or past feeding time. Look for hunger cues like rooting, sucking on hands, and turning toward the breast or bottle.", "doctor": "Consult a doctor if baby refuses feeds repeatedly, has poor weight gain, vomits forcefully, or seems unusually sleepy or weak."},
    "tired":      {"advice": "Create a calm environment, dim lights, reduce noise, and try gentle rocking or swaddling. Watch for sleep cues like yawning, rubbing eyes, or looking away.", "doctor": "See a doctor if baby is very hard to wake, shows low activity, or has changes in breathing or color."},
    "belly_pain": {"advice": "Try burping, gentle tummy-down massage, or placing baby tummy-down across your lap and patting the back. Check for gas, constipation, or hard belly.", "doctor": "Seek medical help if belly is very hard or swollen, baby cries in pain when touched, vomits repeatedly, has blood in stool, or high fever."},
    "burping":    {"advice": "Hold baby upright against your shoulder or sitting on your lap and gently pat or rub the back. Burp during and after feeds to release trapped air.", "doctor": "Consult a doctor if baby always seems uncomfortable after feeds, arches back in pain, or has frequent choking or coughing with feeds."},
    "cold_hot":   {"advice": "Check baby's neck or chest with your hand, not just hands and feet. Add or remove a thin layer of clothing or adjust room temperature.", "doctor": "Call a doctor urgently if baby has fever (>=38C), feels very cold, shivers, has mottled or bluish skin, or breathing looks abnormal."},
    "discomfort": {"advice": "Check diaper, clothing tags, tight swaddling, or uncomfortable positions. Offer cuddling, gentle rocking, or a change of environment.", "doctor": "Consult a doctor if baby remains very irritable despite comfort, has rash, swelling, or any new symptom you are worried about."},
    "lonely":     {"advice": "Hold your baby, offer skin-to-skin contact, talk or sing softly, and maintain eye contact. Sometimes babies just need closeness and reassurance.", "doctor": "If crying becomes unusually intense, prolonged, or baby shows other signs of illness, talk to a doctor."},
    "scared":     {"advice": "Comfort baby quickly, hold them close, speak softly, and reduce sudden loud noises or bright lights. Stay calm so baby can settle more easily.", "doctor": "Seek medical help if baby's cry is suddenly very high-pitched, they appear in pain, or you suspect injury or illness."},
    "silence":    {"advice": "Baby seems calm/quiet in this segment. Continue routine care, feeding on schedule, and safe sleep practices.", "doctor": "If baby becomes unusually quiet, floppy, very hard to wake, or breathing changes, seek urgent care."},
}


# ── Pre-filter functions ──────────────────────────────────────────────────────

def analyze_frequency_characteristics(signal):
    fft = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft)
    fft_freq = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    total_energy = np.sum(fft_magnitude ** 2)
    if total_energy == 0:
        return False, 0.0, "0-0 Hz"
    baby_cry_mask = (fft_freq >= 1000) & (fft_freq <= 4000)
    baby_cry_energy = np.sum(fft_magnitude[baby_cry_mask] ** 2)
    energy_ratio = baby_cry_energy / total_energy
    bands = [(0,500,"0-500 Hz"),(500,1000,"500-1000 Hz"),(1000,2000,"1000-2000 Hz"),(2000,4000,"2000-4000 Hz"),(4000,8000,"4000-8000 Hz")]
    max_band_energy, dominant_range = 0, "unknown"
    for low, high, lbl in bands:
        e = np.sum(fft_magnitude[(fft_freq >= low) & (fft_freq < high)] ** 2)
        if e > max_band_energy:
            max_band_energy = e; dominant_range = lbl
    return energy_ratio >= FREQUENCY_ENERGY_THRESHOLD, energy_ratio, dominant_range


def check_harmonic_structure(signal):
    fft = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft)
    fft_freq = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    threshold = np.mean(fft_magnitude) * 2
    peaks = [(fft_freq[i], fft_magnitude[i]) for i in range(1, len(fft_magnitude)-1)
             if fft_magnitude[i] > threshold and fft_magnitude[i] > fft_magnitude[i-1] and fft_magnitude[i] > fft_magnitude[i+1]]
    num_peaks = len(peaks)
    if num_peaks < 2:
        return False, 0.0, num_peaks
    harmonic_energy = sum(m**2 for _, m in peaks)
    noise_energy = np.sum(fft_magnitude**2) - harmonic_energy
    hnr_db = 100.0 if noise_energy <= 0 else 10 * np.log10(harmonic_energy / noise_energy + 1e-10)
    return hnr_db >= HARMONIC_NOISE_RATIO_THRESHOLD, hnr_db, num_peaks


def detect_cry_rhythm(signal):
    frame_length = int(0.05 * SAMPLE_RATE)
    hop_length   = int(0.025 * SAMPLE_RATE)
    rms_frames = [np.sqrt(np.mean(signal[i:i+frame_length]**2))
                  for i in range(0, len(signal)-frame_length, hop_length)]
    rms_envelope = np.array(rms_frames)
    if len(rms_envelope) == 0:
        return False, [], 0
    is_burst = rms_envelope > np.mean(rms_envelope) * 0.5
    transitions = np.diff(is_burst.astype(int))
    burst_starts = np.where(transitions == 1)[0]
    burst_ends   = np.where(transitions == -1)[0]
    if is_burst[0]:  burst_starts = np.concatenate([[0], burst_starts])
    if is_burst[-1]: burst_ends   = np.concatenate([burst_ends, [len(is_burst)-1]])
    burst_durations = [(e - s) * hop_length / SAMPLE_RATE for s, e in zip(burst_starts, burst_ends)]
    if not burst_durations:
        return False, [], 0
    valid = [d for d in burst_durations if BURST_DURATION_MIN <= d <= BURST_DURATION_MAX]
    return len(valid) > 0, burst_durations, len(valid)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_signal(signal, is_phone_audio: bool = False):
    try:
        signal, _ = librosa.effects.trim(signal, top_db=20)
        if len(signal) > 0:
            signal = librosa.util.normalize(signal)
    except Exception as e:
        print(f"DEBUG: Signal preprocessing error: {e}")

    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)), mode='constant')

    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
        n_mels=N_MELS, fmin=20, fmax=8000, power=2.0
    )
    feats_mels_time = librosa.power_to_db(mel_spec, ref=np.max)

    if isinstance(model, CNNBabyCryModel):
        feats_time_mels = feats_mels_time.T
        median = np.median(feats_time_mels)
        mad    = np.median(np.abs(feats_time_mels - median))
        if mad > 1e-6:
            feats_time_mels = (feats_time_mels - median) / (1.4826 * mad)
        else:
            feats_time_mels = (feats_time_mels - np.mean(feats_time_mels)) / (np.std(feats_time_mels) + 1e-6)
        feats = feats_time_mels.T
    else:
        feats = feats_mels_time.T
        if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
            feats = (feats - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
        else:
            feats = (feats - np.mean(feats)) / (np.std(feats) + 1e-6)

    return torch.FloatTensor(feats).unsqueeze(0).unsqueeze(0).to(device)


# ── Window aggregation ────────────────────────────────────────────────────────

def _aggregate_slots(slots):
    """
    Given a list of WINDOW_SIZE entries — each either (label, confidence) or None —
    compute the final result.

    Rules:
      1. Every slot counts (None slots = no cry detected in that window).
      2. If ALL slots are None → return {'label': 'no_baby_detected', 'confidence': 0.0}.
      3. Otherwise, sum confidences per cry class across all slots.
         The class with the highest accumulated confidence wins.
      4. The returned confidence is the accumulated sum (can exceed 1.0 if a class
         appears multiple times — this is intentional so repetition boosts the winner).

    Returns dict: {'label': str, 'confidence': float, 'counts': dict}
    """
    class_scores = {}   # label -> accumulated confidence
    class_counts = {}   # label -> how many windows detected it

    for slot in slots:
        if slot is None:
            continue
        label, conf = slot
        class_scores[label] = class_scores.get(label, 0.0) + conf
        class_counts[label] = class_counts.get(label, 0) + 1

    print("\n===== WINDOW SUMMARY =====")
    for i, slot in enumerate(slots, 1):
        if slot is None:
            print(f"  {i}. [no cry detected]")
        else:
            print(f"  {i}. {slot[0]:<14} {slot[1]:.4f}")

    if not class_scores:
        print("  -> FINAL: no_baby_detected")
        print("==========================\n")
        return {'label': 'no_baby_detected', 'confidence': 0.0, 'counts': {}}

    best_label = max(class_scores, key=class_scores.__getitem__)
    best_score = class_scores[best_label]

    print(f"  Accumulated scores: { {k: round(v,4) for k,v in class_scores.items()} }")
    print(f"  -> FINAL: {best_label} (accumulated_conf={best_score:.4f}, "
          f"seen in {class_counts[best_label]}/{WINDOW_SIZE} windows)")
    print("==========================\n")

    return {
        'label':      best_label,
        'confidence': round(best_score, 4),
        'counts':     class_counts,
    }


def push_window_slot(label_or_none, confidence_or_none):
    """
    Record one window's outcome into the shared slot list.

    - Pass (label, confidence) for a valid cry detection.
    - Pass (None, None) for a window with no cry (silence, rejected, low-conf).

    Returns the aggregated result dict once WINDOW_SIZE slots are filled,
    otherwise returns None (still collecting).
    """
    with window_slots_lock:
        if label_or_none is None:
            window_slots.append(None)
        else:
            window_slots.append((label_or_none, confidence_or_none))

        collected = len(window_slots)
        print(f"DEBUG: Slot {collected}/{WINDOW_SIZE} → "
              f"{label_or_none if label_or_none else 'no_cry'}"
              f"{f' ({confidence_or_none:.4f})' if confidence_or_none else ''}")

        if collected < WINDOW_SIZE:
            return None

        # All 5 slots filled — aggregate and reset
        result = _aggregate_slots(list(window_slots))
        window_slots.clear()
        return result


# ── Logging ───────────────────────────────────────────────────────────────────

def log_event_realtime(pred_label, confidence):
    now = datetime.now()
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date","time","hour","source","predicted_label","confidence"])
        writer.writerow([now.date().isoformat(), now.strftime("%H:%M:%S"),
                         now.strftime("%H"), "microphone", pred_label, f"{confidence:.4f}"])


# ── Email helpers ─────────────────────────────────────────────────────────────

def _read_latest_csv_row(file_path=LOG_FILE):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            if any(row.values()):
                last = row
        return last


def _send_email_message(subject, body, to_email=None, from_email=None, app_password=None):
    to_email     = to_email     or os.environ.get('CRY_TO_EMAIL')
    from_email   = from_email   or os.environ.get('CRY_EMAIL')
    app_password = app_password or os.environ.get('CRY_APP_PASSWORD')
    if not all([to_email, from_email, app_password]):
        return False, 'missing credentials or recipient'
    msg = EmailMessage()
    msg['From'] = from_email; msg['To'] = to_email; msg['Subject'] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(from_email, app_password); s.send_message(msg)
        return True, None
    except Exception as e:
        return False, str(e)


def send_latest_log_as_email(num_lines=10, to_email=None, from_email=None, app_password=None):
    if not os.path.isfile(LOG_FILE):
        return False, f'{LOG_FILE} not found'
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        try: header = next(r)
        except StopIteration: return False, 'log file empty'
        dq = deque(maxlen=num_lines)
        for row in r: dq.append(row)
    lines = [', '.join(header)] + [', '.join(r) for r in dq]
    return _send_email_message('Latest baby cry events (realtime)',
                               'Latest realtime cry events:\n\n' + '\n'.join(lines),
                               to_email=to_email, from_email=from_email, app_password=app_password)


def report_email(pred_label, advice, doctor, to_email=None, from_email=None, app_password=None):
    body = (f"Detected baby cry type: {pred_label}\n\n"
            f"Suggested parental guidance (not medical advice):\n"
            f"- What you can try now: {advice}\n"
            f"- When to consult a doctor: {doctor}\n\n"
            "This is an automated message from the Baby Cry Detection System.")
    ok, err = _send_email_message('Baby Cry Alert', body,
                                  to_email=to_email, from_email=from_email, app_password=app_password)
    now_ts = time.time()
    if ok:
        monitor_settings['last_email_sent'] = now_ts
        monitor_settings['consecutive_failures'] = 0
    else:
        monitor_settings['consecutive_failures'] = monitor_settings.get('consecutive_failures', 0) + 1
        print(f"Email failed (attempt {monitor_settings['consecutive_failures']}): {err}")
    monitor_settings['last_email_attempt'] = now_ts
    return ok, err


# ── Realtime monitor (microphone) ─────────────────────────────────────────────

def run_realtime_monitor(stop_event=None, email_on_detect=False,
                         email_recipient=None, from_email=None,
                         app_password=None, throttle_seconds=60):

    print("Starting real-time baby cry monitoring...")
    print(f"Sampling at {SAMPLE_RATE} Hz | Window={DURATION}s | Step=1s | Slots={WINDOW_SIZE}")
    print(f"Using device: {device}\nPress Ctrl+C to stop.\n")

    BUFFER_SIZE = SAMPLE_RATE * DURATION
    STEP_SIZE   = SAMPLE_RATE * 1   # 1-second step

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buffer_filled = 0

    with window_slots_lock:
        window_slots.clear()

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                print("Stop event — exiting monitor.")
                break

            audio = sd.rec(STEP_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()

            audio_buffer = np.roll(audio_buffer, -STEP_SIZE)
            audio_buffer[-STEP_SIZE:] = audio.flatten()
            buffer_filled = min(buffer_filled + STEP_SIZE, BUFFER_SIZE)

            if buffer_filled < BUFFER_SIZE:
                print("Buffer warming up...")
                continue

            signal = audio_buffer.copy()

            # Determine this window's outcome
            slot_label, slot_conf = None, None

            audio_energy = np.sqrt(np.mean(signal**2))
            if audio_energy < 0.003:
                print(f"Stage 1 - Low energy ({audio_energy:.4f})")
            elif not analyze_frequency_characteristics(signal)[0]:
                print("Rejected: Frequency")
            elif not check_harmonic_structure(signal)[0]:
                print("Rejected: Spectral")
            elif not detect_cry_rhythm(signal)[0]:
                print("Rejected: Temporal")
            else:
                print("Pre-filter PASS → model...")
                X_new = extract_features_from_signal(signal)
                with torch.no_grad():
                    outputs = model(X_new)
                    probs = torch.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                    conf = conf.item(); idx = idx.item()
                pred = str(labels[idx])
                if pred not in ('silence', 'non_cry') and conf >= CONFIDENCE_THRESHOLD:
                    slot_label, slot_conf = pred, round(conf, 4)
                    print(f"Model: {pred} ({conf:.4f})")
                else:
                    print(f"Model rejected: {pred} @ {conf:.4f}")

            # Push slot (cry or None) — every window counts
            result = push_window_slot(slot_label, slot_conf)
            if result is None:
                continue   # still collecting slots

            # 5 slots done — process result
            final_label = result['label']
            final_conf  = result['confidence']
            now = datetime.now()

            if final_label == 'no_baby_detected':
                print("No baby cry detected across 5 windows.")
                continue

            print(f"FINAL DETECTED: {final_label} | Accumulated conf={final_conf:.4f}")
            log_event_realtime(final_label, final_conf)

            guidance = GUIDANCE.get(final_label)
            last_detections.append({
                'date': now.date().isoformat(),
                'time': now.strftime('%H:%M:%S'),
                'label': final_label,
                'confidence': f"{final_conf:.4f}",
                'guidance': guidance
            })

            if email_on_detect and (email_recipient or os.environ.get('CRY_TO_EMAIL')):
                recipient = email_recipient or os.environ.get('CRY_TO_EMAIL')
                last_ok = monitor_settings.get('last_email_sent', 0.0)
                failures = monitor_settings.get('consecutive_failures', 0)
                if time.time() - last_ok >= throttle_seconds * min(2**failures, 16):
                    ok, err = report_email(final_label,
                                           guidance['advice'] if guidance else '',
                                           guidance['doctor'] if guidance else '',
                                           to_email=recipient, from_email=from_email, app_password=app_password)
                    print("Email sent" if ok else f"Email failed: {err}")

            if guidance:
                print(f"Guidance:\n- Try: {guidance['advice']}\n- Doctor: {guidance['doctor']}\n")

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        print("Monitor loop exited.")


def start_monitor(email_on_detect=False, email_recipient=None, from_email=None,
                  app_password=None, throttle_seconds=60):
    global monitor_thread, monitor_stop_event
    with monitor_lock:
        if monitor_thread and monitor_thread.is_alive():
            return False
        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=run_realtime_monitor,
            kwargs=dict(stop_event=monitor_stop_event, email_on_detect=email_on_detect,
                        email_recipient=email_recipient, from_email=from_email,
                        app_password=app_password, throttle_seconds=throttle_seconds),
            daemon=True
        )
        monitor_thread.start()
        return True


def stop_monitor(timeout=5.0):
    global monitor_thread, monitor_stop_event
    with monitor_lock:
        if not monitor_thread or not monitor_thread.is_alive():
            return False
        monitor_stop_event.set()
        monitor_thread.join(timeout)
        if not monitor_thread.is_alive():
            monitor_thread = None; monitor_stop_event = None
            return True
        return False


def is_monitor_running():
    return bool(monitor_thread and monitor_thread.is_alive())


def get_last_detections(n=10):
    return list(last_detections)[-n:]


if __name__ == "__main__":
    run_realtime_monitor()