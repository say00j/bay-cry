"""
Improved Phone-Robust Model Training
=====================================
All improvements combined with strong overfitting prevention:
1. 100% phone augmentation (not 50%)
2. Multiple augmentation variations
3. Enhanced class weights
4. Data augmentation for minority classes
5. 150 epochs with early stopping (patience=25)
6. Strong regularization (dropout, weight decay, batch norm)
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
EPOCHS = 150  # Increased
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4  # L2 regularization
PATIENCE = 25  # Early stopping to prevent overfitting
RANDOM_SEED = 42

# Augmentation settings
TARGET_SAMPLES_PER_CLASS = 800  # Augment minority classes

MODEL_OUT = "best_model_phone_robust_v2.pth"
LABELS_OUT = "label_classes_phone_robust_v2.npy"
NORM_OUT = "normalization_params_phone_robust_v2.npy"
HISTORY_CSV = "training_history_phone_robust_v2.csv"
CONFUSION_PNG = "confusion_matrix_phone_robust_v2.png"
TRAINING_PNG = "training_curves_phone_robust_v2.png"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print("="*70)
print("IMPROVED PHONE-ROBUST MODEL TRAINING")
print("="*70)
print("Improvements:")
print("  1. 100% phone augmentation (always applied)")
print("  2. Multiple augmentation variations")
print("  3. Enhanced class weights for failed classes")
print("  4. Data augmentation for minority classes")
print("  5. 150 epochs with early stopping (patience=25)")
print("  6. Strong overfitting prevention (dropout, L2, batch norm)")
print()

# ============================================================================
# PHONE-SPEAKER AUGMENTATION (IMPROVED)
# ============================================================================
def simulate_phone_speaker(audio, sr=16000, phone_type='standard'):
    """Simulate different phone speaker types"""
    nyquist = sr / 2
    
    if phone_type == 'standard':
        low, high = 300 / nyquist, 3400 / nyquist
    elif phone_type == 'low_quality':
        low, high = 400 / nyquist, 3000 / nyquist
    elif phone_type == 'high_quality':
        low, high = 250 / nyquist, 4000 / nyquist
    else:
        low, high = 300 / nyquist, 3400 / nyquist
    
    b, a = scipy_signal.butter(4, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, audio)
    compressed = np.tanh(filtered * np.random.uniform(1.2, 1.8)) / 1.5
    compressed = np.clip(compressed, -0.95, 0.95)
    return compressed.astype(np.float32)

def add_room_reverb(audio, sr=16000, room_size='medium'):
    """Add room reverb with different room sizes"""
    if room_size == 'small':
        delay_ms = np.random.uniform(20, 40)
        decay = np.random.uniform(0.2, 0.3)
    elif room_size == 'medium':
        delay_ms = np.random.uniform(40, 80)
        decay = np.random.uniform(0.3, 0.4)
    elif room_size == 'large':
        delay_ms = np.random.uniform(80, 150)
        decay = np.random.uniform(0.4, 0.5)
    else:
        delay_ms = 50
        decay = 0.3
    
    delay_samples = int(delay_ms / 1000 * sr)
    reverb = np.zeros_like(audio)
    if delay_samples < len(audio):
        reverb[delay_samples:] = audio[:-delay_samples] * decay
    return audio + reverb

def add_background_noise(audio, noise_level='medium'):
    """Add background noise with different levels"""
    if noise_level == 'low':
        factor = np.random.uniform(0.002, 0.004)
    elif noise_level == 'medium':
        factor = np.random.uniform(0.004, 0.008)
    elif noise_level == 'high':
        factor = np.random.uniform(0.008, 0.015)
    else:
        factor = 0.005
    
    noise = np.random.randn(len(audio)) * factor
    return audio + noise

def reduce_volume(audio, distance='medium'):
    """Simulate different distances"""
    if distance == 'close':
        factor = np.random.uniform(0.7, 0.9)
    elif distance == 'medium':
        factor = np.random.uniform(0.4, 0.7)
    elif distance == 'far':
        factor = np.random.uniform(0.2, 0.4)
    else:
        factor = 0.5
    return audio * factor

def apply_phone_augmentation_improved(signal):
    """
    Apply comprehensive phone-to-mic degradation
    100% chance to apply (not 50%)
    Multiple variations for robustness
    """
    # Always apply phone speaker simulation
    phone_type = np.random.choice(['standard', 'low_quality', 'high_quality'])
    signal = simulate_phone_speaker(signal, phone_type=phone_type)
    
    # Always add reverb (different room sizes)
    room_size = np.random.choice(['small', 'medium', 'large'])
    signal = add_room_reverb(signal, room_size=room_size)
    
    # 70% chance to add noise
    if np.random.rand() < 0.7:
        noise_level = np.random.choice(['low', 'medium', 'high'])
        signal = add_background_noise(signal, noise_level=noise_level)
    
    # Always simulate distance
    distance = np.random.choice(['close', 'medium', 'far'])
    signal = reduce_volume(signal, distance=distance)
    
    return signal

# ============================================================================
# MODEL ARCHITECTURE (Same as before - proven architecture)
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
# DATASET WITH IMPROVED PHONE AUGMENTATION
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
        
        # Apply phone augmentation during training (100% of the time)
        if self.augment:
            signal = apply_phone_augmentation_improved(signal)
        
        features = extract_mel_spectrogram(signal)
        features = torch.FloatTensor(features).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return features, label

# ============================================================================
# DATA LOADING WITH MINORITY CLASS AUGMENTATION
# ============================================================================
print("Step 1: Loading dataset with augmentation...")
print()

classes = ['belly_pain', 'burping', 'cold_hot', 'discomfort', 'hungry', 
           'lonely', 'scared', 'silence', 'tired']

all_files = []
all_labels = []
original_counts = {}

for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    original_counts[class_name] = len(wav_files)
    
    for wav_file in wav_files:
        all_files.append(os.path.join(class_dir, wav_file))
        all_labels.append(class_idx)
    
    print(f"  {class_name:12s}: {len(wav_files):4d} files")

print()
print(f"Total original samples: {len(all_files)}")
print()

# Augment minority classes
print("Step 2: Augmenting minority classes...")
print()

augmented_files = []
augmented_labels = []

for class_idx, class_name in enumerate(classes):
    current_count = original_counts[class_name]
    
    if current_count < TARGET_SAMPLES_PER_CLASS:
        needed = TARGET_SAMPLES_PER_CLASS - current_count
        class_files = [f for f, l in zip(all_files, all_labels) if l == class_idx]
        
        for i in range(needed):
            source_file = np.random.choice(class_files)
            augmented_files.append(source_file)
            augmented_labels.append(class_idx)
        
        print(f"  {class_name:12s}: +{needed:4d} augmented samples ({current_count} -> {TARGET_SAMPLES_PER_CLASS})")

all_files.extend(augmented_files)
all_labels.extend(augmented_labels)

all_files = np.array(all_files)
all_labels = np.array(all_labels)

print()
print(f"Total after augmentation: {len(all_files)}")
print()

# Split dataset
train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, all_labels, test_size=0.30, random_state=RANDOM_SEED, stratify=all_labels
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_labels
)

print(f"Step 3: Dataset split")
print(f"  Train: {len(train_files)} samples")
print(f"  Val:   {len(val_files)} samples")
print(f"  Test:  {len(test_files)} samples")
print()

# Create datasets
train_dataset = BabyCryDataset(train_files, train_labels, augment=True)
val_dataset = BabyCryDataset(val_files, val_labels, augment=False)
test_dataset = BabyCryDataset(test_files, test_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# MODEL SETUP WITH ENHANCED CLASS WEIGHTS
# ============================================================================
print("Step 4: Setting up model with enhanced class weights...")
print()

model = CNNBabyCryModel(len(classes)).to(device)

# Compute base class weights
base_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

# ENHANCED: Boost weights for previously failed classes
enhancement_factors = {
    'belly_pain': 1.3,
    'burping': 1.3,
    'cold_hot': 1.2,
    'discomfort': 1.5,  # Failed in previous training
    'hungry': 1.5,      # Failed in previous training
    'lonely': 1.0,
    'scared': 1.0,
    'silence': 1.3,
    'tired': 1.5        # Failed in previous training
}

enhanced_weights = []
for class_idx, class_name in enumerate(classes):
    enhanced_weight = base_weights[class_idx] * enhancement_factors[class_name]
    enhanced_weights.append(enhanced_weight)

class_weights = torch.FloatTensor(enhanced_weights).to(device)

print("Enhanced class weights:")
for class_idx, class_name in enumerate(classes):
    print(f"  {class_name:12s}: {class_weights[class_idx]:.4f} (factor: {enhancement_factors[class_name]}x)")
print()

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"Scheduler: CosineAnnealingLR")
print(f"Early stopping: patience={PATIENCE}")
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("="*70)
print("TRAINING")
print("="*70)
print()

history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
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
    current_lr = optimizer.param_groups[0]['lr']
    
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  lr={current_lr:.6f}  ({epoch_time:.1f}s)")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  -> Best model saved! Val acc: {val_acc:.4f}")
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break
    
    scheduler.step()

print()
print(f"Training complete! Best val acc: {best_val_acc:.4f}")
print()

# Save artifacts
np.save(LABELS_OUT, np.array(classes))
norm_params = {'method': 'per_sample_robust', 'n_mels': N_MELS, 'sample_rate': SAMPLE_RATE,
               'n_fft': N_FFT, 'hop_length': HOP_LENGTH, 'fmin': FMIN, 'fmax': FMAX}
np.save(NORM_OUT, norm_params)

with open(HISTORY_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    for i in range(len(history['epoch'])):
        writer.writerow([history['epoch'][i], history['train_loss'][i], history['train_acc'][i],
                        history['val_loss'][i], history['val_acc'][i], history['lr'][i]])

print(f"Model saved: {MODEL_OUT}")
print(f"Labels saved: {LABELS_OUT}")
print(f"Norm params saved: {NORM_OUT}")
print(f"History saved: {HISTORY_CSV}")
print()
print("="*70)
print("IMPROVED PHONE-ROBUST MODEL READY!")
print("="*70)
print("This model is trained for phone-to-mic audio with:")
print("  ✓ 100% phone-speaker simulation")
print("  ✓ Multiple room sizes and distances")
print("  ✓ Enhanced class weights")
print("  ✓ Strong overfitting prevention")
print("="*70)
