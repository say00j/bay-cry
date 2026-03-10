"""
Simplified real-time detection WITHOUT hybrid pipeline
For testing if the pipeline is blocking detections
"""
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn as nn
import os
from datetime import datetime

# Model architecture
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

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

labels = np.load("label_classes_phone_robust_v2.npy", allow_pickle=True)
model = CNNBabyCryModel(len(labels)).to(device)
model.load_state_dict(torch.load("best_model_phone_robust_v2.pth", map_location=device, weights_only=True))
model.eval()

print(f"Model loaded: best_model_phone_robust_v2.pth (improved phone-robust model)")
print(f"Classes: {labels}")

# Audio settings
SAMPLE_RATE = 16000
DURATION = 5
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS = 128

def extract_features(signal):
    """Extract features from audio signal"""
    # Trim and normalize
    try:
        signal, _ = librosa.effects.trim(signal, top_db=20)
        if len(signal) > 0:
            signal = librosa.util.normalize(signal)
    except:
        pass
    
    # Pad or truncate
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)))
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
        n_mels=N_MELS, fmin=20, fmax=8000, power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    log_mel = log_mel.T
    median = np.median(log_mel)
    mad = np.median(np.abs(log_mel - median))
    if mad > 1e-6:
        log_mel = (log_mel - median) / (1.4826 * mad)
    else:
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    log_mel = log_mel.T
    
    return torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0).to(device)

print("\n" + "="*70)
print("SIMPLIFIED REAL-TIME BABY CRY DETECTION")
print("="*70)
print("NO HYBRID PIPELINE - Direct model prediction")
print(f"Listening for {DURATION} seconds per window")
print("Press Ctrl+C to stop")
print("="*70)
print()

try:
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Listening...")
        
        # Record audio
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        signal = audio.flatten()
        
        # Check energy
        energy = np.sqrt(np.mean(signal**2))
        
        if energy < 0.01:
            print(f"  Low energy ({energy:.4f}) - likely silence\n")
            continue
        
        # Extract features and predict
        features = extract_features(signal)
        
        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            pred_class = labels[pred_idx.item()]
            confidence = confidence.item()
        
        # Show all predictions
        print(f"  Energy: {energy:.4f}")
        print(f"  Prediction: {pred_class} ({confidence:.3f})")
        print(f"  All probabilities:")
        probs_np = probs.cpu().numpy()[0]
        for i, (cls, prob) in enumerate(zip(labels, probs_np)):
            print(f"    {cls:12s}: {prob:.3f}")
        print()

except KeyboardInterrupt:
    print("\nStopped by user")
