# Fix False Positive Detection Bugfix Design

## Overview

The baby cry detection system incorrectly classifies non-baby sounds (cat sounds, user voice, dog barking) as baby cry types instead of correctly identifying them as "non_cry" or showing "No baby cry detected". This design implements a hybrid detection architecture combining rule-based pre-filtering with model classification to reject non-baby sounds before they reach the neural network classifier.

The fix introduces a 5-stage detection pipeline:
1. Energy-based silence detection (existing)
2. Frequency analysis for baby cry characteristics (1000-4000 Hz dominant energy)
3. Spectral pattern matching (harmonic structure analysis)
4. Temporal pattern analysis (cry rhythm detection)
5. Model classification with adjusted thresholds (existing, enhanced)

This approach ensures that cat sounds, dog barking, and human speech are rejected early in the pipeline, preventing false positives while maintaining accurate classification of actual baby cries.

## Glossary

- **Bug_Condition (C)**: The condition that triggers the bug - when non-baby sounds (cat, dog, human voice) are input to the system
- **Property (P)**: The desired behavior when non-baby sounds are detected - system shows "No baby cry detected" or classifies as "non_cry"
- **Preservation**: Existing baby cry classification accuracy and real-time processing that must remain unchanged by the fix
- **Hybrid Detection**: Combined rule-based pre-filtering and model-based classification approach
- **Frequency Analysis**: Examination of dominant frequency bands to identify baby cry characteristics (1000-4000 Hz)
- **Spectral Pattern Matching**: Analysis of harmonic structure and spectral envelope to distinguish baby cries from other sounds
- **Temporal Pattern Analysis**: Detection of cry rhythm patterns (burst-pause-burst structure typical of baby cries)
- **Pre-filtering Stage**: Rule-based checks that reject non-baby sounds before model classification
- **Mel Spectrogram**: Time-frequency representation of audio with 128 mel bins used for feature extraction
- **best_model_improved_90.pth**: The CNN model trained with 90% accuracy that should be used instead of phone_augmented model
- **non_cry class**: The trained class containing 601 files (167 cat sounds, 113 dog barking) that the model should predict for non-baby sounds

## Bug Details

### Fault Condition

The bug manifests when non-baby sounds (cat sounds, dog barking, human voice) are input to the real-time detection system. The system incorrectly predicts baby cry classes (burping, cold_hot, discomfort, scared, etc.) instead of correctly identifying these sounds as "non_cry" or showing "No baby cry detected".

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type AudioSignal (5 seconds at 16000 Hz)
  OUTPUT: boolean
  
  RETURN (isCatSound(input) OR isDogSound(input) OR isHumanVoice(input))
         AND NOT (dominantFrequencyInRange(input, 1000, 4000))
         AND NOT (hasTypicalBabyCryHarmonics(input))
         AND NOT (hasBurstPausePattern(input))
         AND modelPrediction(input) != "non_cry"
         AND modelPrediction(input) != "silence"
END FUNCTION
```

### Examples

- **Cat Sound Example**: Playing cat meow from Data/non_cry/ directory → System predicts "scared" (confidence 0.75) instead of "non_cry"
  - Expected: "No baby cry detected" or "non_cry"
  - Actual: "scared" with high confidence

- **User Voice Example**: User speaks into microphone → System predicts "burping" (confidence 0.68) instead of "non_cry"
  - Expected: "No baby cry detected" or "non_cry"
  - Actual: "burping" or "cold_hot" with moderate confidence

- **Dog Barking Example**: Playing dog bark sound → System predicts "discomfort" (confidence 0.72) instead of "non_cry"
  - Expected: "No baby cry detected" or "non_cry"
  - Actual: "discomfort" with high confidence

- **Edge Case - Loud Background Noise**: Loud music or TV sounds → Should be rejected as non-baby sound
  - Expected: "No baby cry detected" or "non_cry"

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Actual baby cry classification (belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired) must continue to work with same accuracy
- Silence detection for low-energy audio must continue to work correctly
- Audio capture and resampling from 48000 Hz to 16000 Hz must remain unchanged
- Web application real-time classification and display on http://localhost:5000 must continue to function
- Mel spectrogram feature extraction with 128 mel bins must remain unchanged
- Model inference pipeline and confidence thresholding must remain functional

**Scope:**
All inputs that ARE actual baby cries should be completely unaffected by this fix. This includes:
- All 8 baby cry types (belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired)
- Silence or very low energy audio
- Real-time processing performance and latency

## Hypothesized Root Cause

Based on the bug description and code analysis, the most likely issues are:

1. **Missing Pre-filtering Stages**: The system currently only checks energy levels (silence detection) but does not perform frequency analysis, spectral pattern matching, or temporal pattern analysis to reject non-baby sounds before model classification
   - Current pipeline: Energy check → Model classification
   - Missing: Frequency analysis, spectral matching, temporal analysis

2. **Model Confusion on Non-cry Class**: The model was trained with non_cry class (601 files including cats and dogs) but may not be predicting this class correctly due to:
   - Insufficient confidence threshold adjustment for non_cry predictions
   - Model may be overfitting to baby cry features and not learning to reject non-baby sounds
   - Wrong model being used (phone_augmented instead of best_model_improved_90.pth)

3. **Lack of Frequency-based Rejection**: Baby cries have dominant energy in 1000-4000 Hz range, but the system does not check this characteristic
   - Cat meows: 700-1500 Hz (lower fundamental frequency)
   - Dog barks: 500-1000 Hz (much lower fundamental frequency)
   - Human voice: 85-255 Hz fundamental (much lower than baby cries)
   - Baby cries: 1000-4000 Hz dominant energy with harmonics

4. **No Harmonic Structure Analysis**: Baby cries have specific harmonic patterns that differ from animal sounds
   - Baby cries: Clear harmonic structure with fundamental + harmonics
   - Cat meows: Different harmonic ratios and formant structure
   - Dog barks: Noisy, less harmonic structure
   - Human speech: Formant structure different from baby cries

5. **Missing Temporal Pattern Detection**: Baby cries have characteristic burst-pause-burst patterns
   - Baby cries: 0.5-2 second bursts with pauses
   - Cat meows: Shorter, more varied duration
   - Dog barks: Very short bursts (0.1-0.3 seconds)
   - Human speech: Continuous with different rhythm

## Correctness Properties

Property 1: Fault Condition - Non-baby Sound Rejection

_For any_ audio input where the sound is not a baby cry (cat sound, dog bark, human voice, or other non-baby sound), the fixed detection system SHALL reject the audio through pre-filtering stages (frequency analysis, spectral pattern matching, temporal pattern analysis) OR classify it as "non_cry", resulting in "No baby cry detected" message displayed to the user.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Baby Cry Classification Accuracy

_For any_ audio input that IS an actual baby cry (belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired), the fixed detection system SHALL produce the same classification result as the original system, preserving the 90% accuracy for baby cry type identification and maintaining real-time processing performance.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `realtime_pytorch.py`

**Function**: `run_realtime_monitor` (add pre-filtering pipeline before model inference)

**Specific Changes**:

1. **Add Frequency Analysis Function**: Create `analyze_frequency_characteristics(signal)` function
   - Extract FFT of the audio signal
   - Calculate dominant frequency bands
   - Check if dominant energy is in 1000-4000 Hz range (baby cry characteristic)
   - Return boolean: True if baby cry frequency pattern detected, False otherwise
   - Threshold: At least 60% of energy should be in 1000-4000 Hz range for baby cries

2. **Add Spectral Pattern Matching Function**: Create `check_harmonic_structure(signal)` function
   - Extract harmonic peaks from FFT
   - Calculate harmonic-to-noise ratio (HNR)
   - Analyze formant structure
   - Compare against baby cry spectral templates
   - Return boolean: True if baby cry harmonic pattern detected, False otherwise
   - Threshold: HNR > 10 dB for baby cries (cats/dogs have lower HNR)

3. **Add Temporal Pattern Analysis Function**: Create `detect_cry_rhythm(signal)` function
   - Calculate RMS energy envelope over time
   - Detect burst-pause patterns
   - Measure burst duration and pause duration
   - Check for typical baby cry rhythm (0.5-2 second bursts)
   - Return boolean: True if baby cry temporal pattern detected, False otherwise
   - Threshold: Burst duration 0.5-2 seconds with pauses 0.2-1 second

4. **Integrate 5-Stage Pipeline in run_realtime_monitor**: Modify the detection loop
   - Stage 1: Energy check (existing) - reject silence
   - Stage 2: Frequency analysis - reject if not in baby cry frequency range
   - Stage 3: Spectral pattern matching - reject if harmonic structure doesn't match baby cry
   - Stage 4: Temporal pattern analysis - reject if rhythm doesn't match baby cry pattern
   - Stage 5: Model classification (existing) - classify baby cry type
   - Each rejection stage prints diagnostic message: "Non-baby sound detected (frequency/spectral/temporal) - skipping prediction"

5. **Adjust Model Selection**: Ensure `best_model_improved_90.pth` is used
   - Modify `_CANDIDATES` priority list to prefer best_model_improved_90.pth
   - This model has better non_cry class training (601 files)
   - Remove or deprioritize phone_augmented model

6. **Add Threshold Tuning Parameters**: Create configurable thresholds at module level
   - `FREQUENCY_ENERGY_THRESHOLD = 0.6` (60% energy in 1000-4000 Hz)
   - `HARMONIC_NOISE_RATIO_THRESHOLD = 10.0` (dB)
   - `BURST_DURATION_MIN = 0.5` (seconds)
   - `BURST_DURATION_MAX = 2.0` (seconds)
   - `PAUSE_DURATION_MIN = 0.2` (seconds)
   - `PAUSE_DURATION_MAX = 1.0` (seconds)
   - Allow easy tuning based on testing results

7. **Add Diagnostic Logging**: Enhance logging to track rejection reasons
   - Log which stage rejected the audio (frequency/spectral/temporal)
   - Log measured values vs thresholds for debugging
   - Add optional verbose mode for detailed analysis

**File**: `app_pytorch.py`

**Function**: Audio processing endpoint (apply same pre-filtering pipeline)

**Specific Changes**:
1. Import the new pre-filtering functions from realtime_pytorch.py
2. Apply the same 5-stage pipeline before model classification
3. Return "No baby cry detected" message when pre-filtering rejects audio
4. Maintain existing web interface and real-time display functionality

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly through the 5-stage pipeline and preserves existing baby cry classification accuracy.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that play cat sounds, dog sounds, and record user voice, then observe what the UNFIXED system predicts. Analyze the frequency characteristics, spectral patterns, and temporal patterns of these sounds to confirm they differ from baby cries. Run these tests on the UNFIXED code to observe failures and understand the root cause.

**Test Cases**:
1. **Cat Sound Test**: Play cat meow files from Data/non_cry/ directory (will fail on unfixed code)
   - Expected failure: System predicts "scared" or other baby cry class
   - Measure: Dominant frequency (should be 700-1500 Hz, not 1000-4000 Hz)
   - Measure: Harmonic structure (should differ from baby cry)

2. **Dog Bark Test**: Play dog barking files from Data/non_cry/ directory (will fail on unfixed code)
   - Expected failure: System predicts "discomfort" or other baby cry class
   - Measure: Dominant frequency (should be 500-1000 Hz, not 1000-4000 Hz)
   - Measure: Temporal pattern (should be very short bursts 0.1-0.3s)

3. **User Voice Test**: Record user speaking into microphone (will fail on unfixed code)
   - Expected failure: System predicts "burping" or "cold_hot"
   - Measure: Fundamental frequency (should be 85-255 Hz, not 1000-4000 Hz)
   - Measure: Formant structure (should differ from baby cry)

4. **Edge Case - Loud Music Test**: Play loud background music or TV sounds (may fail on unfixed code)
   - Expected failure: System may predict random baby cry class
   - Measure: Spectral pattern (should not match baby cry harmonic structure)

**Expected Counterexamples**:
- Non-baby sounds are classified as baby cry types with moderate to high confidence (0.6-0.8)
- Frequency analysis will show dominant energy outside 1000-4000 Hz range for cats, dogs, human voice
- Spectral analysis will show different harmonic structure compared to baby cries
- Temporal analysis will show different burst-pause patterns
- Possible causes: Missing pre-filtering stages, model confusion on non_cry class, wrong model being used

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds (non-baby sounds), the fixed system produces the expected behavior (rejection or non_cry classification).

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := hybrid_detection_pipeline(input)
  ASSERT (result.rejected_at_stage IN [2, 3, 4]) OR (result.prediction == "non_cry")
  ASSERT result.display_message == "No baby cry detected"
END FOR
```

**Test Plan**: After implementing the 5-stage pipeline, test with the same cat sounds, dog sounds, and user voice to verify they are rejected at appropriate stages.

**Test Cases**:
1. **Cat Sound Rejection**: Verify cat sounds are rejected at frequency analysis stage (Stage 2)
   - Assert: Dominant frequency not in 1000-4000 Hz range
   - Assert: System displays "No baby cry detected"

2. **Dog Bark Rejection**: Verify dog barks are rejected at frequency or temporal analysis stage
   - Assert: Either frequency check fails OR temporal pattern check fails
   - Assert: System displays "No baby cry detected"

3. **User Voice Rejection**: Verify user voice is rejected at frequency or spectral analysis stage
   - Assert: Fundamental frequency too low OR formant structure doesn't match
   - Assert: System displays "No baby cry detected"

4. **Threshold Tuning**: Test with borderline cases to tune thresholds
   - Test with baby-like sounds (high-pitched animal sounds)
   - Adjust thresholds to minimize false negatives while eliminating false positives

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold (actual baby cries), the fixed system produces the same result as the original system.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  result_original := original_detection_system(input)
  result_fixed := hybrid_detection_pipeline(input)
  ASSERT result_fixed.prediction == result_original.prediction
  ASSERT result_fixed.confidence ≈ result_original.confidence (within 5%)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all baby cry inputs

**Test Plan**: Observe behavior on UNFIXED code first for actual baby cry sounds, then write property-based tests capturing that behavior. Verify the 5-stage pipeline passes baby cries through to model classification without rejection.

**Test Cases**:
1. **Baby Cry Classification Preservation**: Test all 8 baby cry types pass through pre-filtering
   - Test files: belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired
   - Assert: All stages 2-4 pass (frequency, spectral, temporal checks return True)
   - Assert: Model prediction matches original system (same cry type)
   - Assert: Confidence within 5% of original system

2. **Silence Detection Preservation**: Verify low-energy audio still classified as silence
   - Test with very quiet audio (energy < 0.01)
   - Assert: Stage 1 rejects (existing energy check)
   - Assert: No change in silence detection behavior

3. **Real-time Performance Preservation**: Verify processing latency remains acceptable
   - Measure: Time from audio capture to prediction display
   - Assert: Latency increase < 100ms (pre-filtering should be fast)
   - Assert: No audio dropouts or buffer issues

4. **Web Application Preservation**: Verify app_pytorch.py continues to work correctly
   - Test: Real-time classification display on http://localhost:5000
   - Assert: UI updates correctly with predictions
   - Assert: "No baby cry detected" message displays for non-baby sounds
   - Assert: Baby cry predictions display correctly with guidance

### Unit Tests

- Test frequency analysis function with synthetic signals at different frequency ranges
- Test spectral pattern matching with known baby cry vs cat/dog spectrograms
- Test temporal pattern detection with synthetic burst-pause patterns
- Test edge cases (very short audio, very loud audio, distorted audio)
- Test threshold boundary conditions (audio just above/below thresholds)

### Property-Based Tests

- Generate random baby cry audio samples and verify they pass all pre-filtering stages
- Generate random non-baby audio samples and verify they are rejected at appropriate stages
- Test with varying noise levels added to baby cries (should still pass)
- Test with varying noise levels added to non-baby sounds (should still be rejected)
- Test across many frequency ranges to verify 1000-4000 Hz threshold is correct

### Integration Tests

- Test full pipeline with real cat sound files from Data/non_cry/ directory
- Test full pipeline with real dog bark files from Data/non_cry/ directory
- Test full pipeline with recorded user voice samples
- Test full pipeline with all 8 baby cry types from training data
- Test web application end-to-end with microphone input
- Test real-time monitoring script with various audio inputs
- Test model selection (verify best_model_improved_90.pth is loaded)
- Test threshold tuning with presentation panel test cases (cat sounds)

### Threshold Tuning Strategy

**Approach**: Start with conservative thresholds and tune based on test results

**Phase 1 - Initial Thresholds** (based on literature and analysis):
- Frequency energy threshold: 0.6 (60% in 1000-4000 Hz)
- Harmonic-to-noise ratio: 10 dB
- Burst duration: 0.5-2.0 seconds
- Pause duration: 0.2-1.0 seconds

**Phase 2 - Validation with Test Data**:
- Run exploratory tests with cat sounds, dog sounds, user voice
- Measure actual frequency distributions, HNR values, temporal patterns
- Adjust thresholds to achieve 100% rejection of non-baby sounds

**Phase 3 - Preservation Validation**:
- Test with all baby cry samples from training data
- Ensure 0% false negatives (no baby cries rejected)
- If baby cries are rejected, relax thresholds slightly

**Phase 4 - Edge Case Testing**:
- Test with borderline cases (high-pitched cat meows, baby-like sounds)
- Fine-tune thresholds to balance false positives vs false negatives
- Document final threshold values and rationale

**Success Criteria**:
- 100% rejection of cat sounds, dog sounds, user voice
- 0% rejection of actual baby cries (all 8 types)
- Processing latency increase < 100ms
- Ready for presentation panel demonstration tomorrow
