# Requirements Document: Fix Model Accuracy Crisis

## Introduction

The baby cry detection model (best_model_improved_90.pth) has a critical accuracy failure. While documentation claims 90%+ accuracy based on training metrics, real-world testing on actual audio files reveals only 44.4% accuracy (20/45 correct predictions). The model exhibits severe class imbalance bias, with 6 out of 9 classes achieving 0% accuracy and over-predicting "hungry" for 60%+ of samples. This represents a fundamental training-inference mismatch that blocks the entire project.

This specification defines requirements to rebuild the model with genuine 80%+ accuracy across all 9 cry classes, validated on real audio files rather than training metrics alone.

## Glossary

- **Model**: The CNN-based neural network that classifies baby cry audio into 9 categories
- **Training_Accuracy**: Accuracy measured on the training/validation dataset during model training
- **Real_World_Accuracy**: Accuracy measured by running inference on actual WAV files from the dataset
- **Class_Imbalance_Bias**: When a model over-predicts certain classes due to imbalanced training data
- **Feature_Extractor**: Component that converts raw audio signals into log-mel spectrograms
- **Inference_Pipeline**: The complete process from audio input to classification prediction
- **Validation_Protocol**: Testing methodology that measures model performance on real audio files
- **Class_Performance**: Per-class accuracy metrics showing how well the model predicts each cry type
- **Training_Dataset**: The 6,520 audio files in the Data/ folder used for model training
- **Confusion_Matrix**: Matrix showing predicted vs actual classifications for all classes

## Requirements

### Requirement 1: Accurate Real-World Performance Measurement

**User Story:** As a developer, I want to measure model accuracy on actual audio files, so that I can verify the model works in real-world conditions rather than just training metrics.

#### Acceptance Criteria

1. THE Validation_Protocol SHALL test the Model on at least 5 samples per class from the Training_Dataset
2. THE Validation_Protocol SHALL compute Real_World_Accuracy as (correct predictions / total predictions) * 100
3. THE Validation_Protocol SHALL generate a Confusion_Matrix showing predicted vs actual labels for all 9 classes
4. THE Validation_Protocol SHALL report per-class accuracy for each of the 9 cry types
5. WHEN validation completes, THE Validation_Protocol SHALL save results to a validation report file
6. THE Validation_Protocol SHALL use the same Feature_Extractor and preprocessing as the Inference_Pipeline

### Requirement 2: Minimum Class Performance Standards

**User Story:** As a developer, I want every cry class to achieve minimum accuracy thresholds, so that the model works reliably for all cry types, not just a subset.

#### Acceptance Criteria

1. THE Model SHALL achieve at least 70% Class_Performance for each of the 9 cry classes individually
2. THE Model SHALL achieve at least 80% Real_World_Accuracy across all classes combined
3. IF any class achieves less than 70% Class_Performance, THEN THE training process SHALL identify it as requiring additional training data or rebalancing
4. THE Model SHALL NOT over-predict any single class for more than 30% of all samples
5. WHEN tested on belly_pain samples, THE Model SHALL achieve at least 70% accuracy (currently 0%)
6. WHEN tested on burping samples, THE Model SHALL achieve at least 70% accuracy (currently 0%)
7. WHEN tested on discomfort samples, THE Model SHALL achieve at least 70% accuracy (currently 0%)
8. WHEN tested on tired samples, THE Model SHALL achieve at least 70% accuracy (currently 0%)

### Requirement 3: Balanced Training Data

**User Story:** As a developer, I want balanced training data across all classes, so that the model learns to recognize all cry types equally well without bias.

#### Acceptance Criteria

1. THE training process SHALL analyze class distribution in the Training_Dataset before training begins
2. WHEN class imbalance exceeds 2:1 ratio, THE training process SHALL apply balancing strategies
3. THE training process SHALL use class-weighted loss functions to compensate for remaining imbalances
4. THE training process SHALL apply data augmentation only to underrepresented classes
5. THE training process SHALL validate that no class has fewer than 300 training samples after balancing
6. THE training process SHALL log the final class distribution before training begins

### Requirement 4: Training-Inference Consistency

**User Story:** As a developer, I want training and inference to use identical preprocessing, so that training accuracy reflects real-world performance.

#### Acceptance Criteria

1. THE Feature_Extractor SHALL use identical preprocessing parameters during training and inference
2. THE Feature_Extractor SHALL save normalization parameters during training and load them during inference
3. THE Feature_Extractor SHALL use the same audio sample rate (16000 Hz) for training and inference
4. THE Feature_Extractor SHALL use the same audio duration (5 seconds) for training and inference
5. THE Feature_Extractor SHALL use the same mel-spectrogram parameters (n_mels, n_fft, hop_length) for training and inference
6. WHEN normalization parameters are saved, THE Feature_Extractor SHALL include method type and all statistical parameters
7. THE Inference_Pipeline SHALL validate that loaded normalization parameters match the Model version

### Requirement 5: Overfitting Prevention

**User Story:** As a developer, I want to prevent overfitting, so that the model generalizes well to unseen audio rather than memorizing training data.

#### Acceptance Criteria

1. THE training process SHALL split data into training (70%), validation (15%), and test (15%) sets
2. THE training process SHALL use early stopping with patience of at least 15 epochs
3. THE Model SHALL include dropout layers with dropout rate between 0.3 and 0.5
4. THE training process SHALL monitor validation loss and stop when it stops improving
5. WHEN Training_Accuracy exceeds Real_World_Accuracy by more than 15%, THEN THE Model SHALL be considered overfitted
6. THE training process SHALL apply regularization techniques (L2 weight decay, batch normalization)
7. THE training process SHALL save only the model checkpoint with best validation accuracy

### Requirement 6: Comprehensive Training Metrics

**User Story:** As a developer, I want detailed training metrics, so that I can diagnose issues and verify the model is learning correctly.

#### Acceptance Criteria

1. THE training process SHALL log Training_Accuracy and validation accuracy for each epoch
2. THE training process SHALL log per-class accuracy on the validation set every 5 epochs
3. THE training process SHALL generate a Confusion_Matrix on the validation set after training completes
4. THE training process SHALL save training history to a CSV file including epoch, train_loss, train_acc, val_loss, val_acc
5. THE training process SHALL plot training and validation accuracy curves and save as an image
6. THE training process SHALL log the learning rate for each epoch
7. WHEN training completes, THE training process SHALL run the Validation_Protocol on the test set and report Real_World_Accuracy

### Requirement 7: Model Architecture Validation

**User Story:** As a developer, I want to validate the model architecture is appropriate, so that the model has sufficient capacity to learn all 9 classes without overfitting.

#### Acceptance Criteria

1. THE Model SHALL have at least 3 convolutional blocks with increasing filter sizes
2. THE Model SHALL use batch normalization after each convolutional layer
3. THE Model SHALL use dropout after each pooling layer
4. THE Model SHALL use global average pooling before the final classification layer
5. THE Model SHALL have between 1M and 10M trainable parameters
6. WHEN the Model is instantiated, THE Model SHALL log its architecture summary including layer types and parameter counts
7. THE Model SHALL use LeakyReLU or ReLU activation functions (not sigmoid or tanh in hidden layers)

### Requirement 8: Hungry Class Bias Elimination

**User Story:** As a developer, I want to eliminate the "hungry" over-prediction bias, so that the model predicts all classes fairly rather than defaulting to "hungry".

#### Acceptance Criteria

1. WHEN tested on non-hungry samples, THE Model SHALL predict "hungry" for less than 20% of samples
2. THE training process SHALL analyze prediction distribution on validation set and flag if any class exceeds 30% of predictions
3. THE training process SHALL use class-weighted loss with higher penalties for over-predicted classes
4. THE Model SHALL achieve at least 70% precision for the "hungry" class (true positives / all hungry predictions)
5. THE Model SHALL achieve at least 70% recall for the "hungry" class (true positives / all actual hungry samples)

### Requirement 9: Reproducible Training Process

**User Story:** As a developer, I want a reproducible training process, so that I can retrain the model with consistent results and compare different approaches.

#### Acceptance Criteria

1. THE training process SHALL set random seeds for Python, NumPy, and PyTorch before training begins
2. THE training process SHALL log all hyperparameters (learning rate, batch size, epochs, optimizer settings) to a config file
3. THE training process SHALL save the Model, normalization parameters, and label classes with matching version identifiers
4. THE training process SHALL log the PyTorch version, CUDA version, and device type used for training
5. WHEN the same hyperparameters and random seed are used, THE training process SHALL produce models with accuracy within 2% of each other
6. THE training process SHALL save a training manifest file listing all training data files used

### Requirement 10: Real-World Validation Protocol

**User Story:** As a developer, I want a standardized validation protocol, so that I can consistently measure real-world accuracy and compare model versions.

#### Acceptance Criteria

1. THE Validation_Protocol SHALL test on at least 45 audio files (5 per class minimum)
2. THE Validation_Protocol SHALL load audio files using the same librosa settings as training
3. THE Validation_Protocol SHALL run inference using the complete Inference_Pipeline including all preprocessing
4. THE Validation_Protocol SHALL generate a detailed report showing: overall accuracy, per-class accuracy, confusion matrix, and misclassified samples
5. THE Validation_Protocol SHALL save the validation report with timestamp and model version identifier
6. THE Validation_Protocol SHALL flag any class with accuracy below 70% as "FAILING"
7. WHEN validation completes, THE Validation_Protocol SHALL output a pass/fail status based on the 80% overall accuracy threshold

