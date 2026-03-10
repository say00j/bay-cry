# Tasks: Fix Accuracy Degradation During Training

# Tasks: Achieve 90%+ Accuracy and Stop Degradation

## 0. CRITICAL: Verify Current Setup (DO THIS FIRST)

### 0.1 Check which dataset is being used
- [ ] Open `train_pytorch.py` and check `DATA_FILE` variable
- [ ] Verify it's using full dataset (6,520 samples), not partial dataset
- [ ] Expected files:
  - `babycry_features_full_dataset.npz` (6,520 samples) ✓
  - `babycry_features_smart_balanced.npz` (7,329 samples) ✓
  - `babycry_features_dataset_mel.npz` (13,040 samples - might be duplicated) ⚠️
- [ ] Print actual sample count and verify

### 0.2 Check normalization method
- [ ] Check if using global normalization (WRONG for 90%+)
- [ ] Should use per-sample normalization (CORRECT)
- [ ] Verify normalization code in training script

### 0.3 Quick diagnostic run
- [ ] Run training for 3 epochs with current setup
- [ ] Note starting accuracy and trend (improving or degrading?)
- [ ] Check for error messages or warnings
- [ ] Document findings before making changes

## 1. Emergency Diagnostic Phase (URGENT)

### 1.1 Add immediate monitoring
- [ ] Create `TrainingMonitor` class with metrics tracking
- [ ] Add `compute_gradient_norm()` function
- [ ] Add `compute_weight_norm()` function
- [ ] Add `check_gradient_health()` function
- [ ] Integrate monitoring into training loop

### 1.2 Run 5-epoch diagnostic
- [ ] Start training with full monitoring enabled
- [ ] Log metrics after each epoch
- [ ] Check for gradient issues (NaN, Inf, zero gradients)
- [ ] Generate diagnostic plots
- [ ] Identify specific issue (overfitting, exploding/vanishing gradients, etc.)

## 2. Apply Emergency Fixes (IMMEDIATE)

### 2.1 Add gradient clipping (ALWAYS SAFE)
- [ ] Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after backward pass
- [ ] Test training for 2 epochs
- [ ] Verify gradients stay bounded

### 2.2 Reduce learning rate
- [ ] Change initial learning rate from 0.001 to 0.0001
- [ ] Update optimizer initialization
- [ ] Test training for 2 epochs
- [ ] Check if accuracy stabilizes

### 2.3 Add catastrophic failure detection
- [ ] Add check for accuracy drop > 10% in one epoch
- [ ] Reload best model if catastrophic drop detected
- [ ] Log warning and stop training
- [ ] Implement in training loop

## 3. Path to 90%+ Accuracy (CRITICAL)

### 3.1 Switch to full dataset (if not already)
- [ ] Change `DATA_FILE` to `"babycry_features_full_dataset.npz"` or `"babycry_features_smart_balanced.npz"`
- [ ] Verify dataset has 6,520+ samples
- [ ] Check class distribution is reasonable
- [ ] Verify no data corruption

### 3.2 Implement per-sample normalization
- [ ] Create `normalize_per_sample()` function
- [ ] Replace global normalization with per-sample
- [ ] Apply to both training and test data
- [ ] Save normalization method to file
- [ ] Test training for 5 epochs

### 3.3 Optimize training configuration for 90%+
- [ ] Set initial learning rate to 0.001 (with weight_decay=1e-4)
- [ ] Increase scheduler patience to 7
- [ ] Increase early stopping patience to 20
- [ ] Set max epochs to 150
- [ ] Add gradient clipping (max_norm=1.0)

### 3.4 Add target-based early stopping
- [ ] Modify early stopping to continue until 90%+ reached
- [ ] Only stop if accuracy >= 90% AND no improvement for patience epochs
- [ ] Log when target is reached
- [ ] Continue for stability after reaching target

### 3.5 Verify class weights are working
- [ ] Print class weights after computation
- [ ] Verify minority classes have higher weights
- [ ] Check loss function is using weights correctly
- [ ] Monitor per-class performance during training

## 4. Targeted Fixes (Based on Diagnosis)

### 4.1 If Overfitting Detected (train_acc >> val_acc)
- [ ] Increase dropout rates in all dropout layers
  - [ ] `dropout1`: 0.25 → 0.4
  - [ ] `dropout2`: 0.25 → 0.4
  - [ ] `dropout3`: 0.3 → 0.5
  - [ ] `dropout4`: 0.5 → 0.6
  - [ ] `dropout5`: 0.4 → 0.5
- [ ] Add L2 regularization (weight_decay=1e-4) to optimizer
- [ ] Reduce early stopping patience from 15 to 10
- [ ] Test training for 10 epochs

### 4.2 If Exploding Gradients Detected (grad_norm > 10)
- [ ] Reduce gradient clipping max_norm from 1.0 to 0.5
- [ ] Further reduce learning rate to 0.00001
- [ ] Check weight initialization
- [ ] Verify batch normalization is working
- [ ] Test training for 5 epochs

### 4.3 If Vanishing Gradients Detected (grad_norm < 0.0001)
- [ ] Replace ReLU with LeakyReLU(0.1)
- [ ] Add residual connections to CNN blocks
- [ ] Re-initialize weights using Kaiming initialization
- [ ] Reduce number of layers if necessary
- [ ] Test training for 5 epochs

### 4.4 If Data Normalization Issues
- [ ] Implement per-sample normalization function
- [ ] Apply to training data before creating DataLoader
- [ ] Apply to validation data
- [ ] Verify normalization consistency
- [ ] Test training for 5 epochs

## 5. Enhanced Monitoring

### 5.1 Add detailed logging
- [ ] Log learning rate at each epoch
- [ ] Log gradient norm at each epoch
- [ ] Log weight norm at each epoch
- [ ] Log train-val accuracy gap
- [ ] Save all metrics to CSV file

### 5.2 Create visualization
- [ ] Plot accuracy curves (train vs val)
- [ ] Plot loss curves (train vs val)
- [ ] Plot learning rate schedule
- [ ] Plot gradient norm over time
- [ ] Plot overfitting gap over time
- [ ] Save as `training_diagnostics.png`

### 5.3 Add early warning system
- [ ] Warn if val_acc drops > 5% in one epoch
- [ ] Warn if train-val gap > 15%
- [ ] Warn if gradient norm > 10 or < 0.0001
- [ ] Warn if loss becomes NaN or Inf
- [ ] Print warnings to console immediately

## 6. Model Architecture Improvements (If Needed)

### 6.1* Add residual connections
- [ ] Create `ResidualBlock` class
- [ ] Replace regular conv blocks with residual blocks
- [ ] Test forward pass with dummy data
- [ ] Train for 5 epochs and compare

### 6.2* Improve weight initialization
- [ ] Create `init_weights()` function
- [ ] Use Kaiming initialization for Conv2d and Linear layers
- [ ] Apply to model before training
- [ ] Test training for 5 epochs

### 6.3* Add learning rate warmup
- [ ] Implement warmup scheduler
- [ ] Start with very low LR (1e-6)
- [ ] Gradually increase to target LR over 5 epochs
- [ ] Test training with warmup

## 7. Validation and Testing

### 7.1 Test with optimized configuration
- [ ] Run full training with all improvements applied
- [ ] Monitor for 50+ epochs (or until 90%+ reached)
- [ ] Verify accuracy reaches 90%+
- [ ] Check training stability
- [ ] Save best model

### 7.2 Verify 90%+ accuracy achievement
- [ ] Confirm validation accuracy >= 90%
- [ ] Check test set accuracy >= 90%
- [ ] Verify train-val gap < 10%
- [ ] Check all classes have > 70% recall
- [ ] Document final metrics

### 7.3 Verify model quality
- [ ] Load best model
- [ ] Run inference on test set
- [ ] Check confusion matrix
- [ ] Verify no class collapse (all predictions same)
- [ ] Check per-class performance

## 8. Documentation

### 8.1 Document root cause and solution
- [ ] Write summary of identified issue
- [ ] Explain why accuracy was degrading
- [ ] Document diagnostic evidence
- [ ] Save to `accuracy_fix_report.md`

### 8.2 Document 90%+ achievement
- [ ] List all improvements applied to reach 90%+
- [ ] Document final accuracy and per-class performance
- [ ] Provide training configuration that worked
- [ ] Save as evidence of meeting requirements

### 8.3 Update training script comments
- [ ] Add comments explaining gradient clipping
- [ ] Add comments explaining learning rate choice
- [ ] Add comments explaining dropout rates
- [ ] Document monitoring system

## Priority Order

**CRITICAL (Do First - Verify Setup):**
0. Task 0.1 - Check dataset being used
0. Task 0.2 - Check normalization method  
0. Task 0.3 - Quick diagnostic run

**CRITICAL (Path to 90%+):**
1. Task 3.1 - Switch to full dataset
2. Task 3.2 - Implement per-sample normalization
3. Task 3.3 - Optimize training configuration
4. Task 3.4 - Add target-based early stopping

**HIGH (Monitoring & Diagnosis):**
5. Task 1.1 - Add monitoring
6. Task 1.2 - Run diagnostic
7. Task 2.1 - Add gradient clipping
8. Task 2.2 - Reduce learning rate if needed

**MEDIUM (Targeted Fixes if Needed):**
9. Task 4.x - Apply targeted fix based on diagnosis
10. Task 5.1 - Enhanced logging
11. Task 5.2 - Visualization

**LOW (After 90%+ Achieved):**
12. Task 7.x - Validation and testing
13. Task 8.x - Documentation

## Notes

- **PRIMARY GOAL:** Achieve 90%+ validation accuracy
- Start by verifying you're using the right dataset and normalization
- Per-sample normalization is CRITICAL for 90%+ accuracy
- Don't accept "stable" training at lower accuracy
- Tasks marked with `*` are optional enhancements
- Focus on reaching 90%+ first, optimize later
