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
    PHONE_PREPROCESSING = True # Force-disabled to prevent feature distortion
    print("Phone-speaker audio preprocessing loaded (currently DISABLED)")
except ImportError:
    PHONE_PREPROCESSING = False
    print("audio_preprocessing.py not found - skipping domain-adaptation")

# Monitor runtime state
monitor_thread: Optional[threading.Thread] = None
monitor_stop_event: Optional[threading.Event] = None
monitor_lock = threading.Lock()
last_detections = deque(maxlen=200)  # store recent detections
window_results = deque(maxlen=5)  # store (label, confidence) for last 5 valid windows
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

# Architecture 1: Deeper CNN with GAP (Used in best_model_improved_90.pth)
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

# Architecture 2: CNN-LSTM with Attention (Original format)
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


# Load model and labels
# Priority Model Candidates
_CANDIDATES = [
    ("best_model_phone_robust_v2.pth",    "label_classes_phone_robust_v2.npy",      "normalization_params_phone_robust_v2.npy"),
    ("best_model_phone_robust.pth",       "label_classes_phone_robust.npy",         "normalization_params_phone_robust.npy"),
    ("best_model_enhanced.pth",           "label_classes_enhanced.npy",             "normalization_params_enhanced.npy"),
    ("best_model_retrained.pth",          "label_classes_retrained.npy",            "normalization_params_retrained.npy"),
    ("best_model_improved_90.pth",        "label_classes_improved_90.npy",          "normalization_params_improved_90.npy"),
]
MODEL_FILE, LABELS_FILE, NORM_PARAMS_FILE = next(
    (m, l, n) for m, l, n in _CANDIDATES if os.path.exists(m)
)
print(f"Using model: {MODEL_FILE}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

labels = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(labels)

# Instantiate the correct architecture
if "improved_90" in MODEL_FILE or "full_dataset" in MODEL_FILE or "enhanced" in MODEL_FILE or "retrained" in MODEL_FILE or "phone_robust" in MODEL_FILE:
    print("Detected CNN architecture for high-accuracy model")
    model = CNNBabyCryModel(num_classes).to(device)
else:
    print("Detected legacy LSTM architecture")
    model = LSTMBabyCryModel(num_classes).to(device)

# Load state dict
model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=False))
model.eval()

# Load normalization parameters
try:
    norm_params = np.load(NORM_PARAMS_FILE, allow_pickle=True).item()
    GLOBAL_MEAN = norm_params.get('mean', norm_params.get('global_mean'))
    GLOBAL_STD = norm_params.get('std', norm_params.get('global_std'))
    IS_ROBUST_NORM = norm_params.get('method') == 'per_sample_robust'
    print(f"Normalization: {'Robust' if IS_ROBUST_NORM else 'Global'} (Mean={GLOBAL_MEAN:.2f})")
except Exception:
    print("Using fallback normalization")
    GLOBAL_MEAN, GLOBAL_STD, IS_ROBUST_NORM = None, None, False

print(f"Model loaded successfully with {num_classes} classes")
print(f"Classes: {labels}")

SAMPLE_RATE = 16000
DURATION = 5
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS = 128

LOG_FILE = "cry_events_log_realtime.csv"

# Hybrid Detection Pipeline Thresholds
# Tuned for hard rejection: ALL stages must pass for sound to reach model
# Values based on empirical testing with cat/dog sounds vs baby cries
# CRITICAL: Thresholds must be permissive enough to avoid false negatives (rejecting baby cries)
# Strategy: Use very permissive thresholds to catch extreme non-baby sounds only
FREQUENCY_ENERGY_THRESHOLD = 0.05  # 5% energy in 1000-4000 Hz (very permissive)
                                    # Baby cries: 7-99% (wide range, some very low-frequency)
                                    # Cat sounds: 1-100% (overlap)
                                    # Dog sounds: 0-78% (some at 0%)
                                    # Set to 5% to catch only extreme low-frequency sounds
                                   
HARMONIC_NOISE_RATIO_THRESHOLD = -15.0  # dB - very permissive, catches only very noisy sounds
                                         # Baby cries: -10 to +1 dB (some very noisy)
                                         # Cat sounds: -11 to +1 dB (overlap)
                                         # Dog sounds: -7 to +1 dB (overlap)
                                         # Set to -15.0 to catch only extremely noisy sounds
                                       
BURST_DURATION_MIN = 0.5  # seconds - minimum burst duration for baby cries
BURST_DURATION_MAX = 2.0  # seconds - maximum burst duration for baby cries
                          # Cat sounds: varied, often have valid bursts
                          # Dog sounds: 0.1-0.3s (too short - this is the discriminative feature)
                          # Baby cries: 0.5-2.0s bursts with pauses

PAUSE_DURATION_MIN = 0.2  # seconds - minimum pause duration between bursts
PAUSE_DURATION_MAX = 1.0  # seconds - maximum pause duration between bursts

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
            "Check baby's neck or chest with your hand, not just hands and feet. "
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
            "Seek medical help if baby's cry is suddenly very high-pitched, they appear in pain, "
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

def analyze_frequency_characteristics(signal):
    """
    Analyze frequency characteristics to detect baby cry patterns.
    Baby cries have dominant energy in 1000-4000 Hz range.
    
    Args:
        signal: Audio signal (numpy array)
    
    Returns:
        tuple: (bool, float, str) - (is_baby_cry, energy_ratio, dominant_freq_range)
            - is_baby_cry: True if baby cry frequency pattern detected, False otherwise
            - energy_ratio: Percentage of energy in baby cry frequency range (0.0-1.0)
            - dominant_freq_range: String describing the dominant frequency range
    """
    # Compute FFT
    fft = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft)
    fft_freq = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    
    # Calculate total energy
    total_energy = np.sum(fft_magnitude ** 2)
    
    if total_energy == 0:
        return False, 0.0, "0-0 Hz"
    
    # Calculate energy in baby cry frequency range (1000-4000 Hz)
    baby_cry_mask = (fft_freq >= 1000) & (fft_freq <= 4000)
    baby_cry_energy = np.sum(fft_magnitude[baby_cry_mask] ** 2)
    
    # Calculate percentage of energy in baby cry range
    energy_ratio = baby_cry_energy / total_energy
    
    # Find dominant frequency range
    # Divide spectrum into bands and find which has most energy
    bands = [
        (0, 500, "0-500 Hz"),
        (500, 1000, "500-1000 Hz"),
        (1000, 2000, "1000-2000 Hz"),
        (2000, 4000, "2000-4000 Hz"),
        (4000, 8000, "4000-8000 Hz")
    ]
    
    max_band_energy = 0
    dominant_range = "unknown"
    for low, high, label in bands:
        band_mask = (fft_freq >= low) & (fft_freq < high)
        band_energy = np.sum(fft_magnitude[band_mask] ** 2)
        if band_energy > max_band_energy:
            max_band_energy = band_energy
            dominant_range = label
    
    # Check if at least 60% of energy is in baby cry frequency range
    is_baby_cry = energy_ratio >= FREQUENCY_ENERGY_THRESHOLD
    return is_baby_cry, energy_ratio, dominant_range

def check_harmonic_structure(signal):
    """
    Check harmonic structure to distinguish baby cries from other sounds.
    Baby cries have clear harmonic structure with high harmonic-to-noise ratio.
    
    Args:
        signal: Audio signal (numpy array)
    
    Returns:
        tuple: (bool, float, int) - (is_baby_cry, hnr_db, num_peaks)
            - is_baby_cry: True if baby cry harmonic pattern detected, False otherwise
            - hnr_db: Harmonic-to-noise ratio in decibels
            - num_peaks: Number of harmonic peaks detected
    """
    # Compute FFT
    fft = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft)
    fft_freq = np.fft.rfftfreq(len(signal), 1.0 / SAMPLE_RATE)
    
    # Find peaks in the spectrum (potential harmonics)
    # Use a simple peak detection: local maxima above mean
    mean_magnitude = np.mean(fft_magnitude)
    threshold = mean_magnitude * 2  # Peaks should be at least 2x mean
    
    peaks = []
    for i in range(1, len(fft_magnitude) - 1):
        if fft_magnitude[i] > threshold and \
           fft_magnitude[i] > fft_magnitude[i-1] and \
           fft_magnitude[i] > fft_magnitude[i+1]:
            peaks.append((fft_freq[i], fft_magnitude[i]))
    
    num_peaks = len(peaks)
    
    if num_peaks < 2:
        # Not enough harmonic structure
        return False, 0.0, num_peaks
    
    # Calculate harmonic-to-noise ratio (HNR)
    # HNR = ratio of harmonic energy to noise energy
    harmonic_energy = sum(mag ** 2 for _, mag in peaks)
    total_energy = np.sum(fft_magnitude ** 2)
    noise_energy = total_energy - harmonic_energy
    
    if noise_energy <= 0:
        hnr_db = 100.0  # Very high HNR
    else:
        hnr = harmonic_energy / noise_energy
        hnr_db = 10 * np.log10(hnr + 1e-10)
    
    # Baby cries should have HNR > 10 dB
    is_baby_cry = hnr_db >= HARMONIC_NOISE_RATIO_THRESHOLD
    return is_baby_cry, hnr_db, num_peaks

def detect_cry_rhythm(signal):
    """
    Detect temporal patterns characteristic of baby cries.
    Baby cries have burst-pause-burst patterns with specific durations.
    
    Args:
        signal: Audio signal (numpy array)
    
    Returns:
        tuple: (bool, list, int) - (is_baby_cry, burst_durations, valid_burst_count)
            - is_baby_cry: True if baby cry temporal pattern detected, False otherwise
            - burst_durations: List of burst durations in seconds
            - valid_burst_count: Number of bursts matching baby cry pattern (0.5-2s)
    """
    # Calculate RMS energy envelope over time
    frame_length = int(0.05 * SAMPLE_RATE)  # 50ms frames
    hop_length = int(0.025 * SAMPLE_RATE)   # 25ms hop
    
    # Calculate RMS for each frame
    rms_frames = []
    for i in range(0, len(signal) - frame_length, hop_length):
        frame = signal[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_frames.append(rms)
    
    rms_envelope = np.array(rms_frames)
    
    if len(rms_envelope) == 0:
        return False, [], 0
    
    # Detect bursts and pauses using threshold
    mean_rms = np.mean(rms_envelope)
    threshold = mean_rms * 0.5  # Bursts are above 50% of mean
    
    # Find burst and pause segments
    is_burst = rms_envelope > threshold
    
    # Find transitions
    transitions = np.diff(is_burst.astype(int))
    burst_starts = np.where(transitions == 1)[0]
    burst_ends = np.where(transitions == -1)[0]
    
    # Handle edge cases
    if len(is_burst) > 0 and is_burst[0]:
        burst_starts = np.concatenate([[0], burst_starts])
    if len(is_burst) > 0 and is_burst[-1]:
        burst_ends = np.concatenate([burst_ends, [len(is_burst) - 1]])
    
    # Calculate burst durations
    burst_durations = []
    for start, end in zip(burst_starts, burst_ends):
        duration_seconds = (end - start) * hop_length / SAMPLE_RATE
        burst_durations.append(duration_seconds)
    
    if len(burst_durations) == 0:
        return False, [], 0
    
    # Check if any burst matches baby cry pattern (0.5-2 seconds)
    valid_bursts = [d for d in burst_durations 
                    if BURST_DURATION_MIN <= d <= BURST_DURATION_MAX]
    
    valid_burst_count = len(valid_bursts)
    
    # Baby cries should have at least one valid burst
    is_baby_cry = valid_burst_count > 0
    return is_baby_cry, burst_durations, valid_burst_count

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

def extract_features_from_signal(signal, is_phone_audio: bool = False):
    """
    Extract log-mel spectrogram features with correct normalization.
    Matches EXACTLY what preprocess_smart_balanced.py did during training.
    """
    # Step 1: Trim and Normalize (Matches preprocess_smart_balanced.py)
    try:
        signal, _ = librosa.effects.trim(signal, top_db=20)
        if len(signal) > 0:
            signal = librosa.util.normalize(signal)
    except Exception as e_proc:
        print(f"DEBUG: Signal preprocessing error: {e_proc}")

    # Step 2: Pad or Truncate to exact length (16000 * 5)
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        pad = SAMPLES_PER_FILE - len(signal)
        signal = np.pad(signal, (0, pad), mode='constant')

    # Step 3: Extract Mel Spectrogram with specific parameters from training
    mel_spec = librosa.feature.melspectrogram(
        y=signal, 
        sr=SAMPLE_RATE, 
        n_fft=2048,
        hop_length=512,
        n_mels=N_MELS,
        fmin=20,
        fmax=8000,
        power=2.0
    )
    feats_mels_time = librosa.power_to_db(mel_spec, ref=np.max)  # (128, 157)

    if isinstance(model, CNNBabyCryModel):
        # Training did:
        #   1. Data extracted -> (128, 157)
        #   2. Transposed -> (157, 128) for storage
        #   3. Robust normalized on (157, 128)
        #   4. Transposed -> (128, 157) for CNN
        feats_time_mels = feats_mels_time.T          
        median = np.median(feats_time_mels)
        mad = np.median(np.abs(feats_time_mels - median))
        if mad > 1e-6:
            feats_time_mels = (feats_time_mels - median) / (1.4826 * mad)
        else:
            feats_time_mels = (feats_time_mels - np.mean(feats_time_mels)) / (np.std(feats_time_mels) + 1e-6)
        feats = feats_time_mels.T                    # (128, 157)
    else:
        # LSTM Model
        feats = feats_mels_time.T
        if GLOBAL_MEAN is not None and GLOBAL_STD is not None:
            feats = (feats - GLOBAL_MEAN) / (GLOBAL_STD + 1e-6)
        else:
            feats = (feats - np.mean(feats)) / (np.std(feats) + 1e-6)

    # Final dimension for PyTorch: (Batch, Channel, Mels, Time) or (Batch, Channel, Time, Mels)
    features = torch.FloatTensor(feats).unsqueeze(0).unsqueeze(0).to(device)
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
    """Send an email and return (True, None) on success or (False, error_message) on failure."""
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
    """Send the last `num_lines` rows of the CSV as plain text email."""
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
    """Build and send a detection report."""
    label_str = str(pred_label)
    subject = 'Baby Cry Alert'
    body = (
        f"Detected baby cry type: {label_str}\n\n"
        f"Suggested parental guidance (not medical advice):\n- What you can try now: {advice}\n- When to consult a doctor: {doctor}\n\n"
        "This is an automated message from the Baby Cry Detection System."
    )

    ok, err = _send_email_message(subject, body, to_email=to_email, from_email=from_email, app_password=app_password)
    now_ts = time.time()
    if ok:
        monitor_settings['last_email_sent'] = now_ts
        monitor_settings['consecutive_failures'] = 0
        monitor_settings['last_email_attempt'] = now_ts
    else:
        monitor_settings['last_email_attempt'] = now_ts
        monitor_settings['consecutive_failures'] = monitor_settings.get('consecutive_failures', 0) + 1
        print(f"Email send failed (attempts={monitor_settings['consecutive_failures']}): {err}")
    return ok, err
    
def run_realtime_monitor(stop_event: Optional[threading.Event]=None,
                         email_on_detect: bool=False,
                         email_recipient: Optional[str]=None,
                         from_email: Optional[str]=None,
                         app_password: Optional[str]=None,
                         throttle_seconds: int=60):

    print("Starting real-time baby cry monitoring (Sliding Window + Max Confidence)...")
    print(f"Sampling at {SAMPLE_RATE} Hz | Window={DURATION}s | Step=1s")
    print(f"Using device: {device}")
    print("Press Ctrl+C to stop.\n")

    # ───── Sliding Window Setup ─────
    BUFFER_SECONDS = DURATION
    STEP_SECONDS = 1

    BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS
    STEP_SIZE = SAMPLE_RATE * STEP_SECONDS

    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buffer_filled = 0

    try:
        last_sent = monitor_settings.get('last_email_sent', 0.0)

        while True:
            if stop_event is not None and stop_event.is_set():
                print("Stop event set — exiting monitor loop.")
                break

            print("Listening (1 sec chunk)...")

            audio = sd.rec(STEP_SIZE,
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype='float32')
            sd.wait()

            new_chunk = audio.flatten()

            # Update circular buffer
            audio_buffer = np.roll(audio_buffer, -STEP_SIZE)
            audio_buffer[-STEP_SIZE:] = new_chunk

            buffer_filled = min(buffer_filled + STEP_SIZE, BUFFER_SIZE)

            if buffer_filled < BUFFER_SIZE:
                print("Buffer warming up...")
                continue

            signal = audio_buffer.copy()

            # ───── Stage 1: Energy Check ─────
            audio_energy = np.sqrt(np.mean(signal**2))
            ENERGY_THRESHOLD = 0.003  # Very permissive threshold to catch all potential baby cries, even very quiet ones

            if audio_energy < ENERGY_THRESHOLD:
                print(f"Stage 1 - Low energy ({audio_energy:.4f}) - skipping")
                continue

            print(f"Energy: {audio_energy:.4f}")

            freq_is_baby_cry, energy_ratio, dominant_freq_range = analyze_frequency_characteristics(signal)
            if not freq_is_baby_cry:
                print(f"Rejected: Frequency (ratio={energy_ratio:.2f}, dominant={dominant_freq_range})")
                continue

            spectral_is_baby_cry, hnr_db, num_peaks = check_harmonic_structure(signal)
            if not spectral_is_baby_cry:
                print(f"Rejected: Spectral (HNR={hnr_db:.2f}, peaks={num_peaks})")
                continue

            temporal_is_baby_cry, burst_durations, valid_burst_count = detect_cry_rhythm(signal)
            if not temporal_is_baby_cry:
                print(f"Rejected: Temporal (valid_bursts={valid_burst_count})")
                continue

            print("Pre-filter PASS")

            # ───── Stage 5: Model Prediction ─────
            X_new = extract_features_from_signal(signal)

            with torch.no_grad():
                outputs = model(X_new)
                probs = torch.softmax(outputs, dim=1)
                confidence, idx = torch.max(probs, 1)
                idx = idx.item()
                confidence = confidence.item()

            pred_label = labels[idx]

            CONFIDENCE_THRESHOLD = 0.24

            if pred_label == 'silence' or confidence < CONFIDENCE_THRESHOLD:
                continue

            # 🧠 Store result for decision
           # 🧠 Store result for decision
            window_results.append((pred_label, confidence))

            window_num = len(window_results)

            print("\n──────── WINDOW ANALYSIS ────────")
            print(f"Window {window_num}/5")
            print(f"Prediction : {pred_label}")
            print(f"Confidence : {confidence:.4f}")
            print("────────────────────────────────")

            # ───── Wait until 5 windows collected ─────
            if len(window_results) < 5:
                continue
            print("\n===== WINDOW SUMMARY (Last 5) =====")
            for i, (lbl, conf) in enumerate(window_results, 1):
                print(f"{i}. {lbl:<12}  {conf:.4f}")
            print("===================================")
            # 🏆 Pick highest-confidence prediction
            best_label, best_conf = max(window_results, key=lambda x: x[1])

            # Clear for next cycle
            window_results.clear()

            pred_label = best_label
            confidence = best_conf

            now = datetime.now()

            print(f"\n🔥 FINAL DETECTED: {pred_label} | Confidence={confidence:.2f}")

            # ───── Log + Memory ─────
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

            # ───── Email Alert ─────
            if email_on_detect and (email_recipient or os.environ.get('CRY_TO_EMAIL')):
                recipient = email_recipient or os.environ.get('CRY_TO_EMAIL')
                now_ts = time.time()

                last_success = monitor_settings.get('last_email_sent', 0.0)
                failures = monitor_settings.get('consecutive_failures', 0)

                backoff_factor = min(2 ** failures, 16)
                effective_throttle = throttle_seconds * backoff_factor

                if now_ts - last_success >= effective_throttle:
                    ok, err = report_email(pred_label,
                                           guidance['advice'] if guidance else '',
                                           guidance['doctor'] if guidance else '',
                                           to_email=recipient,
                                           from_email=from_email,
                                           app_password=app_password)

                    if ok:
                        print("✅ Email sent")
                    else:
                        print(f"❌ Email failed: {err}")

            if guidance:
                print("Guidance:")
                print(f"- Try now: {guidance['advice']}")
                print(f"- Doctor: {guidance['doctor']}\n")

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    finally:
        monitor_settings['last_email_sent'] = last_sent
        print("Monitor loop exited.")

def start_monitor(email_on_detect: bool=False, email_recipient: Optional[str]=None, from_email: Optional[str]=None, app_password: Optional[str]=None, throttle_seconds: int=60):
    """Start the monitor in a background thread."""
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
    """Signal the monitor to stop and wait up to `timeout` seconds."""
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
    run_realtime_monitor()
