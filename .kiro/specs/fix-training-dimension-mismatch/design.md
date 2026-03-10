# Design: Fix Training Dimension Mismatch

## 1. Overview

This design addresses the dimension mismatch issue in the PyTorch training pipeline by implementing proper shape handling, diagnostic logging, and dynamic model initialization.

## 2. Root Cause Analysis

### Current Data Flow:
```
Load .npz → Shape: (N, 157, 128)
Add channel dim → Shape: (N, 1, 157, 128)
CNN processing → Shape varies
Reshape for LSTM → Potential mismatch
```

### Identified Issues:
1. **Hardcoded dimension assumptions** in model architecture
2. **Insufficient shape validation** between pipeline stages
3. **Dynamic LSTM initialization** may fail on first forward pass
4. **Reshape operations** in forward pass may produce unexpected dimensions

## 3. Technical Solution

### 3.1 Enhanced Dimension Diagnostics

Add comprehensive shape logging at critical points:

```python
def log_shape(name, tensor):
    """Log tensor shape for debugging"""
    print(f"  {name}: {tensor.shape}")
```

Insert logging after:
- Data loading
- Normalization
- Channel dimension addition
- Each CNN block
- Reshape for LSTM
- LSTM outputs
- Dense layer inputs

### 3.2 Data Preprocessing Validation

```python
# After loading data
print("="*70)
print("DATA SHAPE VALIDATION")
print("="*70)
print(f"Original X shape: {X.shape}")
print(f"Expected format: (samples, height, width)")

# Validate dimensions
assert len(X.shape) == 3, f"Expected 3D array, got {len(X.shape)}D"
assert X.shape[1] > 0 and X.shape[2] > 0, "Invalid feature dimensions"

# After adding channel dimension
print(f"After channel dim: {X.shape}")
print(f"Expected format: (samples, channels, height, width)")
assert len(X.shape) == 4, f"Expected 4D array after channel addition"
```

### 3.3 Model Architecture Fix

#### Problem: Dynamic LSTM Initialization
The current code initializes `lstm1` as `None` and creates it during the first forward pass. This can cause issues with device placement and gradient tracking.

#### Solution: Calculate dimensions upfront

```python
class BabyCryModel(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 157, 128)):
        super(BabyCryModel, self).__init__()
        
        # Calculate dimensions through CNN blocks
        # Input: (batch, 1, 157, 128)
        # After conv1-2 + pool1: (batch, 64, 78, 64)  # 157//2=78, 128//2=64
        # After conv3-4 + pool2: (batch, 128, 39, 32)  # 78//2=39, 64//2=32
        # After conv5 + pool3: (batch, 256, 19, 16)    # 39//2=19, 32//2=16
        
        # Calculate LSTM input size
        # After reshape: (batch, seq_len, features)
        # seq_len = one of the spatial dimensions (e.g., 16)
        # features = channels * other_spatial_dim (e.g., 256 * 19)
        
        self.lstm_input_size = self._calculate_lstm_input_size(input_shape)
        
        # Initialize LSTM layers with calculated size
        self.lstm1 = nn.LSTM(self.lstm_input_size, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        
    def _calculate_lstm_input_size(self, input_shape):
        """Calculate LSTM input size by doing a dummy forward pass through CNN"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            
            # Simulate CNN blocks
            x = self.pool1(self.conv2(self.conv1(x)))  # After block 1
            x = self.pool2(self.conv4(self.conv3(x)))  # After block 2
            x = self.pool3(self.conv5(x))              # After block 3
            
            # Reshape for LSTM
            batch_size = x.size(0)
            x = x.view(batch_size, x.size(1) * x.size(2), x.size(3))
            x = x.permute(0, 2, 1)
            
            return x.size(2)  # Return feature dimension
    
    def forward(self, x):
        # Add shape logging
        if self.training:
            print(f"Input shape: {x.shape}")
        
        # Conv blocks with logging
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        if self.training:
            print(f"After conv block 1: {x.shape}")
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        if self.training:
            print(f"After conv block 2: {x.shape}")
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        if self.training:
            print(f"After conv block 3: {x.shape}")
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.view(batch_size, x.size(1) * x.size(2), x.size(3))
        x = x.permute(0, 2, 1)
        if self.training:
            print(f"After reshape for LSTM: {x.shape}")
        
        # LSTM layers
        x, _ = self.lstm1(x)
        if self.training:
            print(f"After LSTM1: {x.shape}")
        x, _ = self.lstm2(x)
        if self.training:
            print(f"After LSTM2: {x.shape}")
        
        # Attention
        x = self.attention(x)
        if self.training:
            print(f"After attention: {x.shape}")
        
        # Dense layers
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x
```

### 3.4 Alternative: Simplified Reshape Strategy

If the above approach is too complex, use a simpler flatten-then-reshape strategy:

```python
def forward(self, x):
    # ... CNN blocks ...
    
    # Flatten spatial dimensions
    batch_size = x.size(0)
    x = x.view(batch_size, -1)  # Flatten to (batch, features)
    
    # Reshape for LSTM: treat as sequence of feature vectors
    # Option 1: Single timestep with all features
    x = x.unsqueeze(1)  # (batch, 1, features)
    
    # Option 2: Split features into sequence
    # seq_len = 16  # Choose appropriate sequence length
    # features_per_step = x.size(1) // seq_len
    # x = x.view(batch_size, seq_len, features_per_step)
    
    # ... LSTM layers ...
```

## 4. Implementation Plan

### Phase 1: Add Diagnostics
1. Add shape logging throughout the pipeline
2. Run training to identify exact dimension mismatch
3. Document actual vs expected shapes

### Phase 2: Fix Model Architecture
1. Implement `_calculate_lstm_input_size` method
2. Initialize LSTM layers in `__init__` with correct dimensions
3. Remove dynamic LSTM initialization from `forward`

### Phase 3: Validate and Test
1. Run training for one epoch
2. Verify no dimension errors
3. Check model can make predictions
4. Validate gradient flow

## 5. Correctness Properties

### Property 1: Shape Consistency Through Pipeline
**Validates: Requirements 3.1, 3.2**

For all batches in the training set:
- After loading: `X.shape == (N, H, W)` where N is batch size
- After channel addition: `X.shape == (N, 1, H, W)`
- After each CNN block: dimensions reduce by factor of 2 (due to pooling)
- Before LSTM: `X.shape == (N, seq_len, features)` where `seq_len * features` matches flattened CNN output
- After LSTM: `X.shape == (N, seq_len, lstm_hidden * 2)` for bidirectional
- After attention: `X.shape == (N, lstm_hidden * 2)`
- Final output: `X.shape == (N, num_classes)`

### Property 2: Model Forward Pass Completeness
**Validates: Requirements 3.3**

For any valid input batch:
- Forward pass completes without exceptions
- Output shape matches `(batch_size, num_classes)`
- All intermediate tensors have positive dimensions
- No NaN or Inf values in output

### Property 3: Gradient Flow
**Validates: Requirements 3.3**

After backward pass:
- All model parameters have gradients
- No gradient is NaN or Inf
- Gradient magnitudes are within reasonable bounds (< 100)

## 6. Testing Strategy

### Unit Tests
- Test `_calculate_lstm_input_size` with various input shapes
- Test reshape operations independently
- Validate each CNN block output shape

### Integration Tests
- Test full forward pass with dummy data
- Test backward pass and gradient computation
- Test training loop for one batch

### Property-Based Tests
- Generate random valid input shapes
- Verify shape consistency property holds
- Test with different batch sizes

## 7. Rollback Plan

If the fix doesn't work:
1. Revert to original `train_pytorch.py`
2. Try alternative reshape strategy (flatten approach)
3. Consider simplifying model architecture (remove LSTM, use only CNN)

## 8. Success Criteria

- [ ] Training completes one full epoch without dimension errors
- [ ] All shape logging shows expected dimensions
- [ ] Model achieves > 70% validation accuracy
- [ ] No warnings or errors in training output
