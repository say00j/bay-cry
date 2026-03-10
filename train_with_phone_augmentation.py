"""
Train model with phone-speaker simulation augmentation
This will make the model robust to phone-to-mic audio quality
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from datetime import datetime
from scipy import signal as scipy_signal

# Configuration
DATA_DIR = "Data"
SAMPLE_RATE = 16000
DURATION = 5
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
RANDOM_SEED = 42

MODEL_OUT = "best_model_phone_robust.pth"
LABELS_OUT = "label_classes_phone_robust.npy"
NORM_OUT = "normalization_params_phone_robust.npy"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print("="*70)
print("PHONE-ROBUST MODEL TRAINING")
print("="*70)
print("Simulating phone-speaker degradation for real-world robustness")
print()

# ============================================================================
# PHONE-SPEAKER AUGMENTATION
# ============================================================================
def simulate_phone_speaker(audio, sr=16000):
    """
    Simulate phone speaker characteristics:
    - Bandpass filter (300-3400 Hz typical phone speaker range)
    - Add slight distortion
    - Reduce dynamic range (compression)
    """
    # Bandpass filter (phone speaker frequency response)
    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, audio)
    
    # Slight compression (reduce dynamic range)
    compressed = np.tanh(filtered * 1.5) / 1.5
    
    # Add tiny bit of clipping distortion
    compressed = np.clip(compressed, -0.95, 0.95)
    
    return compressed.astype(np.float32)

def add_room_reverb(audio, sr=16000):
    """Add simple room reverb effect"""
    # Simple reverb using delayed copies
    delay_samples = int(0.05 * sr)  # 50ms delay
    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = audio[:-delay_samples] * 0.3
    return audio + reverb

def add_background_noise(audio, noise_level=0.005):
    """Add background noise"""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def reduce_volume(audio, factor=0.5):
    """Reduce volume to simulate distance"""
    return audio * factor

def apply_phone_augmentation(signal):
    """
    Apply random phone-to-mic degradation
    50% chance to apply phone simulation
    """
    if np.random.rand() < 0.5:
        # Apply phone speaker simulation
        signal = simulate_phone_speaker(signal)
        
        # 50% chance to add reverb
        if np.random.rand() < 0.5:
            signal = add_room_reverb(signal)
        
        # 50% chance to add noise
        if np.random.rand() < 0.5:
            signal = add_background_noise(signal, noise_level=np.random.uniform(0.002, 0.008))
        
        # 50% chance to reduce volume
        if np.random.rand() < 0.5:
            signal = reduce_volume(signal, factor=np.random.uniform(0.3, 0.7))
    
    return signal

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
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

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def load_audio(path):
    signal, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    if len(signal) > 0:
        signal = librosa.util.normalize(signal)
    if len(signal) < SAMPLES_PER_FILE:
        signal = np.pad(signal, (0, SAMPLES_PER_FILE - len(signal)))
    else:
        signal = signal[:SAMPLES_PER_FILE]
    return signal.astype(np.float32)

def extract_mel_spectrogram(signal):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = log_mel.T
    median = np.median(log_mel)
    mad = np.median(np.abs(log_mel - median))
    if mad > 1e-6:
        log_mel = (log_mel - median) / (1.4826 * mad)
    else:
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    log_mel = log_mel.T
    return log_mel

# ============================================================================
# DATASET WITH PHONE AUGMENTATION
# ============================================================================
class BabyCryDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        signal = load_audio(self.file_paths[idx])
        
        # Apply phone augmentation during training
        if self.augment:
            signal = apply_phone_augmentation(signal)
        
        features = extract_mel_spectrogram(signal)
        features = torch.FloatTensor(features).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return features, label

print("Loading dataset...")
classes = ['belly_pain', 'burping', 'cold_hot', 'discomfort', 'hungry', 
           'lonely', 'scared', 'silence', 'tired']

all_files = []
all_labels = []

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    for wav_file in wav_files:
        all_files.append(os.path.join(class_dir, wav_file))
        all_labels.append(class_idx)
    print(f"  {class_name:12s}: {len(wav_files):4d} files")

all_files = np.array(all_files)
all_labels = np.array(all_labels)
print(f"Total: {len(all_files)} samples\n")

# Split dataset
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_labels
)

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}\n")

# Create datasets with phone augmentation
train_dataset = BabyCryDataset(train_files, train_labels, augment=True)
val_dataset = BabyCryDataset(val_files, val_labels, augment=False)
test_dataset = BabyCryDataset(test_files, test_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model setup
model = CNNBabyCryModel(len(classes)).to(device)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print("Training with phone-speaker augmentation...")
print("="*70)

best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)
    
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  -> Best model saved! Val acc: {val_acc:.4f}")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break
    
    scheduler.step()

print(f"\nTraining complete! Best val acc: {best_val_acc:.4f}")
print(f"Model saved to: {MODEL_OUT}")

# Save artifacts
np.save(LABELS_OUT, np.array(classes))
norm_params = {'method': 'per_sample_robust', 'n_mels': N_MELS, 'sample_rate': SAMPLE_RATE,
               'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'fmin': FMIN, 'fmax': FMAX}
np.save(NORM_OUT, norm_params)

print(f"\nModel is now robust to phone-speaker audio!")
print("Test with: python realtime_simple.py (update model file name)")
