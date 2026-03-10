# Tasks: Fix Training Dimension Mismatch

## 1. Diagnostic Phase

### 1.1 Add comprehensive shape logging
- [ ] Create `log_shape()` utility function
- [ ] Add logging after data loading
- [ ] Add logging after normalization
- [ ] Add logging after channel dimension addition
- [ ] Add logging in model forward pass at each major stage

### 1.2 Run diagnostic training
- [ ] Execute training with shape logging enabled
- [ ] Capture and document actual tensor shapes at each stage
- [ ] Identify the exact location of dimension mismatch
- [ ] Document expected vs actual shapes

## 2. Data Pipeline Fix

### 2.1 Add data shape validation
- [ ] Validate loaded data has correct dimensions (3D array)
- [ ] Assert positive feature dimensions
- [ ] Validate channel dimension addition produces 4D array
- [ ] Add informative error messages for shape issues

### 2.2 Verify preprocessing consistency
- [ ] Confirm normalization preserves shape
- [ ] Verify train/test split maintains shape
- [ ] Check DataLoader output shapes

## 3. Model Architecture Fix

### 3.1 Implement dimension calculation
- [ ] Add `_calculate_lstm_input_size()` method to BabyCryModel
- [ ] Perform dummy forward pass through CNN blocks
- [ ] Calculate correct LSTM input dimensions
- [ ] Return calculated feature dimension

### 3.2 Fix LSTM initialization
- [ ] Remove `self.lstm1 = None` from `__init__`
- [ ] Initialize `lstm1` with calculated input size in `__init__`
- [ ] Remove dynamic LSTM initialization from `forward()`
- [ ] Ensure LSTM layers are on correct device

### 3.3 Add forward pass logging
- [ ] Add conditional shape logging (only during training)
- [ ] Log shapes after each CNN block
- [ ] Log shape after reshape for LSTM
- [ ] Log shapes after each LSTM layer
- [ ] Log shape after attention layer

## 4. Testing and Validation

### 4.1 Test model initialization
- [ ] Create model instance with sample input shape
- [ ] Verify all layers are initialized correctly
- [ ] Check LSTM input size matches CNN output
- [ ] Confirm model moves to GPU if available

### 4.2 Test forward pass
- [ ] Create dummy input batch
- [ ] Run forward pass
- [ ] Verify output shape is (batch_size, num_classes)
- [ ] Check no exceptions are raised

### 4.3 Test backward pass
- [ ] Run forward pass with dummy data
- [ ] Compute loss
- [ ] Run backward pass
- [ ] Verify all parameters have gradients
- [ ] Check no NaN or Inf in gradients

### 4.4 Run training for one epoch
- [ ] Start training loop
- [ ] Complete one full epoch
- [ ] Verify no dimension errors
- [ ] Check training and validation metrics are computed
- [ ] Confirm model can be saved

## 5. Documentation and Cleanup

### 5.1 Update code comments
- [ ] Document the dimension flow through the model
- [ ] Add comments explaining reshape operations
- [ ] Document LSTM input size calculation

### 5.2 Create dimension reference
- [ ] Document input shape: (batch, 1, 157, 128)
- [ ] Document shape after each CNN block
- [ ] Document LSTM input/output shapes
- [ ] Document final output shape

### 5.3* Optional: Add shape assertion tests
- [ ] Create unit test for `_calculate_lstm_input_size()`
- [ ] Create test for forward pass with various batch sizes
- [ ] Add property-based test for shape consistency

## 6. Verification

### 6.1 Full training run
- [ ] Run complete training with fixed code
- [ ] Monitor for any dimension-related warnings
- [ ] Verify training completes successfully
- [ ] Check model achieves reasonable accuracy (>70%)

### 6.2 Model inference test
- [ ] Load trained model
- [ ] Run inference on test data
- [ ] Verify predictions have correct shape
- [ ] Check confidence scores are valid probabilities

## Notes

- Tasks marked with `*` are optional enhancements
- Focus on getting training to work first (tasks 1-4)
- Documentation (task 5) can be done after successful training
- All shape logging should be conditional (only during training) to avoid performance overhead during inference
