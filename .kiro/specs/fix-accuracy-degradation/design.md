# Design: Achieve 90%+ Accuracy and Stop Degradation

## 1. Overview

This design provides a systematic approach to:
1. **Stop accuracy degradation immediately** - Prevent the model from getting worse
2. **Diagnose the root cause** - Understand why accuracy is dropping
3. **Achieve 90%+ accuracy target** - Implement the documented improvements to reach the goal

**Philosophy:** We diagnose first, then apply targeted fixes that improve accuracy, not just stabilize it at a lower level.

## 2. Diagnostic Framework

### 2.1 Training Metrics Monitor

```python
class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'grad_norm': [],
            'weight_norm': []
        }
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, grad_norm, weight_norm):
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['lr'].append(lr)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['weight_norm'].append(weight_norm)
        
        # Check for accuracy degradation
        if len(self.metrics['val_acc']) > 1:
            acc_drop = self.metrics['val_acc'][-2] - self.metrics['val_acc'][-1]
            if acc_drop > 5.0:
                print(f"⚠️  WARNING: Validation accuracy dropped by {acc_drop:.2f}%")
            
            # Check for overfitting
            gap = train_acc - val_acc
            if gap > 15.0:
                print(f"⚠️  WARNING: Overfitting detected (gap: {gap:.2f}%)")
        
        # Check for gradient issues
        if grad_norm > 10.0:
            print(f"⚠️  WARNING: Large gradient norm: {grad_norm:.4f}")
        elif grad_norm < 0.0001:
            print(f"⚠️  WARNING: Very small gradient norm: {grad_norm:.4f}")
    
    def plot_metrics(self):
        """Plot all metrics for visual inspection"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['train_acc'], label='Train')
        axes[0, 0].plot(self.metrics['epoch'], self.metrics['val_acc'], label='Val')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train')
        axes[0, 1].plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[0, 2].plot(self.metrics['epoch'], self.metrics['lr'])
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].grid(True)
        
        # Gradient Norm
        axes[1, 0].plot(self.metrics['epoch'], self.metrics['grad_norm'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Weight Norm
        axes[1, 1].plot(self.metrics['epoch'], self.metrics['weight_norm'])
        axes[1, 1].set_title('Weight Norm')
        axes[1, 1].grid(True)
        
        # Overfitting Gap
        gap = [t - v for t, v in zip(self.metrics['train_acc'], self.metrics['val_acc'])]
        axes[1, 2].plot(self.metrics['epoch'], gap)
        axes[1, 2].set_title('Train-Val Gap')
        axes[1, 2].axhline(y=10, color='r', linestyle='--', label='Warning threshold')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_diagnostics.png', dpi=300)
        print("Training diagnostics saved to training_diagnostics.png")
```

### 2.2 Gradient Monitoring

```python
def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def compute_weight_norm(model):
    """Compute total weight norm across all parameters"""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def check_gradient_health(model):
    """Check for gradient issues"""
    issues = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            
            # Check for NaN or Inf
            if torch.isnan(grad).any():
                issues.append(f"NaN gradient in {name}")
            if torch.isinf(grad).any():
                issues.append(f"Inf gradient in {name}")
            
            # Check for zero gradients
            if grad.abs().max() < 1e-10:
                issues.append(f"Zero gradient in {name}")
    
    return issues
```

## 3. Common Causes and Fixes

### 3.1 Learning Rate Too High

**Symptoms:**
- Loss oscillates or increases
- Accuracy jumps around erratically
- Large gradient norms

**Fix:**
```python
# Reduce initial learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced from 0.001

# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use more aggressive learning rate scheduling
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)
```

### 3.2 Overfitting

**Symptoms:**
- Training accuracy high, validation accuracy dropping
- Training loss decreasing, validation loss increasing
- Large train-val gap

**Fix:**
```python
# Increase dropout rates
self.dropout1 = nn.Dropout(0.4)  # Increased from 0.25
self.dropout2 = nn.Dropout(0.4)  # Increased from 0.25
self.dropout3 = nn.Dropout(0.5)  # Increased from 0.3
self.dropout4 = nn.Dropout(0.6)  # Increased from 0.5
self.dropout5 = nn.Dropout(0.5)  # Increased from 0.4

# Add L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Reduce model capacity
# Option: Use fewer filters or layers
```

### 3.3 Vanishing Gradients

**Symptoms:**
- Very small gradient norms (< 0.0001)
- Accuracy plateaus early
- Early layers have zero gradients

**Fix:**
```python
# Use different activation functions
self.relu = nn.LeakyReLU(0.1)  # Instead of ReLU

# Add residual connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        x = F.relu(x)
        return x

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

### 3.4 Exploding Gradients

**Symptoms:**
- Very large gradient norms (> 10)
- Loss becomes NaN or Inf
- Accuracy drops to random chance

**Fix:**
```python
# Add gradient clipping (CRITICAL)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Reduce learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Use batch normalization (already in model)
# Ensure it's applied correctly
```

### 3.5 Data Normalization Issues

**Symptoms:**
- Inconsistent performance across batches
- High variance in loss values
- Model performs differently on train vs val

**Fix:**
```python
# Use per-sample normalization instead of global
def normalize_per_sample(X):
    """Normalize each sample independently"""
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        sample = X[i]
        mean = np.mean(sample)
        std = np.std(sample)
        X_normalized[i] = (sample - mean) / (std + 1e-6)
    return X_normalized

# Apply before training
X_train = normalize_per_sample(X_train)
X_test = normalize_per_sample(X_test)

# Or use batch normalization in model (already present)
```

## 4. Implementation Strategy

### Phase 1: Add Monitoring (Immediate)
1. Implement `TrainingMonitor` class
2. Add gradient norm computation
3. Add weight norm computation
4. Log all metrics every epoch
5. Generate diagnostic plots

### Phase 2: Quick Fixes (If issues detected)
1. Add gradient clipping (always safe)
2. Reduce learning rate by 10x
3. Increase dropout rates
4. Add weight decay

### Phase 3: Targeted Fixes (Based on diagnosis)
- If overfitting → Increase regularization
- If exploding gradients → Clip gradients, reduce LR
- If vanishing gradients → Change activations, add skip connections
- If data issues → Fix normalization

## 5. Correctness Properties

### Property 1: Monotonic Loss Decrease
**Validates: Requirements 3.1**

For training loss over epochs:
- `train_loss[i+1] <= train_loss[i] + tolerance` where tolerance = 0.1
- No sudden jumps (> 2x previous value)
- Eventually converges to stable value

### Property 2: Bounded Gradients
**Validates: Requirements 3.2**

For all training steps:
- `0.0001 <= gradient_norm <= 10.0`
- No NaN or Inf values in gradients
- All layers receive non-zero gradients

### Property 3: Controlled Overfitting
**Validates: Requirements 3.3**

For all epochs after epoch 5:
- `train_acc - val_acc <= 15%`
- If gap increases for 3 consecutive epochs, early stopping triggers
- Validation loss does not increase for more than 5 consecutive epochs

## 6. Testing Strategy

### Diagnostic Run (5 epochs)
```python
# Run with full monitoring
monitor = TrainingMonitor()
for epoch in range(5):
    train_loss, train_acc = train_epoch(...)
    val_loss, val_acc = validate(...)
    grad_norm = compute_gradient_norm(model)
    weight_norm = compute_weight_norm(model)
    
    monitor.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, 
                     optimizer.param_groups[0]['lr'], grad_norm, weight_norm)
    
    # Check for issues
    issues = check_gradient_health(model)
    if issues:
        print("Gradient issues detected:")
        for issue in issues:
            print(f"  - {issue}")

monitor.plot_metrics()
```

### Validation Tests
- Test with reduced learning rate (0.0001)
- Test with gradient clipping (max_norm=1.0)
- Test with increased dropout (0.5)
- Compare results to identify best fix

## 7. Emergency Fixes

If accuracy is dropping catastrophically:

```python
# EMERGENCY FIX 1: Reduce learning rate immediately
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.00001

# EMERGENCY FIX 2: Add aggressive gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# EMERGENCY FIX 3: Increase dropout to maximum
# Modify model dropout layers to 0.7

# EMERGENCY FIX 4: Stop and reload best model
if val_acc < best_acc - 10:
    print("Catastrophic accuracy drop detected. Reloading best model.")
    model.load_state_dict(torch.load('best_model_pytorch.pth'))
    break
```

## 8. Path to 90%+ Accuracy

Based on your documentation (COMPLETE_SOLUTION_SUMMARY.md, BALANCING_STRATEGY.md), here's what's needed:

### 8.1 Use Full Dataset (6,520 samples)
```python
# Verify you're using the right dataset file
DATA_FILE = "babycry_features_full_dataset.npz"  # NOT babycry_features_dataset_mel.npz
# OR
DATA_FILE = "babycry_features_smart_balanced.npz"  # With smart balancing

# Check sample count
print(f"Dataset samples: {X.shape[0]}")
assert X.shape[0] >= 6520, "Not using full dataset!"
```

### 8.2 Per-Sample Normalization (Critical!)
```python
# Instead of global normalization
# X = (X - np.mean(X)) / (np.std(X) + 1e-6)  # WRONG

# Use per-sample normalization
def normalize_per_sample(X):
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        sample = X[i]
        mean = np.mean(sample)
        std = np.std(sample)
        X_normalized[i] = (sample - mean) / (std + 1e-6)
    return X_normalized

X = normalize_per_sample(X)

# Save normalization method for realtime use
normalization_params = {
    'method': 'per_sample',
    'global_mean': None,  # Not used in per-sample
    'global_std': None,   # Not used in per-sample
}
np.save("normalization_params.npy", normalization_params)
```

### 8.3 Proper Class Weighting
```python
# Already implemented, but verify it's working
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
print("Class weights:", class_weights)
# Should show higher weights for minority classes (burping, silence)
```

### 8.4 Optimal Training Configuration
```python
# Learning rate: Start higher, reduce gradually
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Scheduler: More patient
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=7, verbose=True
)

# Early stopping: More patient to reach 90%
max_patience = 20  # Increased from 15

# Gradient clipping: Prevent instability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 8.5 Training Duration
```python
# Train for more epochs if needed
for epoch in range(150):  # Increased from 100
    # ... training code ...
    
    # Only stop if we've reached target
    if val_acc >= 90.0:
        print(f"🎉 Target accuracy reached: {val_acc:.2f}%")
        # Continue for a few more epochs to ensure stability
        if epoch > 30:  # At least 30 epochs
            break
```

## 9. Diagnostic Checklist

Before applying fixes, check:

1. **Dataset Size**
   ```python
   print(f"Training samples: {len(X_train)}")  # Should be ~5,542 (85% of 6,520)
   print(f"Test samples: {len(X_test)}")       # Should be ~978 (15% of 6,520)
   ```

2. **Normalization Method**
   ```python
   # Check if using per-sample or global
   # Per-sample is required for 90%+ accuracy
   ```

3. **Class Distribution**
   ```python
   unique, counts = np.unique(y_train, return_counts=True)
   for cls, count in zip(encoder.classes_[unique], counts):
       print(f"{cls}: {count} samples")
   # Check for severe imbalance
   ```

4. **Model Capacity**
   ```python
   total_params = sum(p.numel() for p in model.parameters())
   print(f"Total parameters: {total_params:,}")
   # Should be ~5M parameters
   ```

## 10. Success Criteria

- [ ] Diagnostic plots show clear issue identification
- [ ] Applied fix stabilizes training AND improves accuracy
- [ ] Validation accuracy reaches 90%+ 
- [ ] No NaN or Inf values in training
- [ ] Train-val gap < 10%
- [ ] All classes have > 70% recall
