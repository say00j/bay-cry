# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - Non-baby Sound False Positive Detection
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: Test with concrete failing cases (cat sounds from Data/non_cry/, dog sounds, user voice recording)
  - Test that cat sounds from Data/non_cry/ directory are incorrectly classified as baby cry types (not "non_cry")
  - Test that dog barking sounds are incorrectly classified as baby cry types (not "non_cry")
  - Test that user voice input is incorrectly classified as baby cry types (not "non_cry")
  - The test assertions should verify: system displays "No baby cry detected" OR classifies as "non_cry"
  - Run test on UNFIXED code (realtime_pytorch.py without hybrid detection pipeline)
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the bug exists)
  - Document counterexamples found: which baby cry types are incorrectly predicted for each non-baby sound
  - Measure frequency characteristics (dominant frequency should be outside 1000-4000 Hz for non-baby sounds)
  - Measure spectral patterns (harmonic structure should differ from baby cries)
  - Measure temporal patterns (burst-pause patterns should differ from baby cries)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Baby Cry Classification Accuracy
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for actual baby cry inputs (all 8 types)
  - Test with belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired samples
  - Write property-based tests capturing observed classification results and confidence levels
  - Property-based testing generates many test cases for stronger guarantees
  - Verify silence detection still works for low-energy audio (energy < 0.01)
  - Verify real-time processing performance (measure baseline latency)
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3. Implement 5-stage hybrid detection pipeline

  - [x] 3.1 Add frequency analysis function to realtime_pytorch.py
    - Create `analyze_frequency_characteristics(signal)` function
    - Extract FFT of the audio signal
    - Calculate dominant frequency bands
    - Check if dominant energy is in 1000-4000 Hz range (baby cry characteristic)
    - Return boolean: True if baby cry frequency pattern detected, False otherwise
    - Threshold: At least 60% of energy should be in 1000-4000 Hz range for baby cries
    - _Bug_Condition: isBugCondition(input) where (isCatSound OR isDogSound OR isHumanVoice) AND NOT dominantFrequencyInRange(input, 1000, 4000)_
    - _Expected_Behavior: Non-baby sounds rejected at frequency analysis stage (Stage 2)_
    - _Preservation: Baby cry frequency characteristics (1000-4000 Hz) pass through unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 3.2 Add spectral pattern matching function to realtime_pytorch.py
    - Create `check_harmonic_structure(signal)` function
    - Extract harmonic peaks from FFT
    - Calculate harmonic-to-noise ratio (HNR)
    - Analyze formant structure
    - Compare against baby cry spectral templates
    - Return boolean: True if baby cry harmonic pattern detected, False otherwise
    - Threshold: HNR > 10 dB for baby cries (cats/dogs have lower HNR)
    - _Bug_Condition: isBugCondition(input) where NOT hasTypicalBabyCryHarmonics(input)_
    - _Expected_Behavior: Non-baby sounds rejected at spectral analysis stage (Stage 3)_
    - _Preservation: Baby cry harmonic structure passes through unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 3.3 Add temporal pattern analysis function to realtime_pytorch.py
    - Create `detect_cry_rhythm(signal)` function
    - Calculate RMS energy envelope over time
    - Detect burst-pause patterns
    - Measure burst duration and pause duration
    - Check for typical baby cry rhythm (0.5-2 second bursts)
    - Return boolean: True if baby cry temporal pattern detected, False otherwise
    - Threshold: Burst duration 0.5-2 seconds with pauses 0.2-1 second
    - _Bug_Condition: isBugCondition(input) where NOT hasBurstPausePattern(input)_
    - _Expected_Behavior: Non-baby sounds rejected at temporal analysis stage (Stage 4)_
    - _Preservation: Baby cry temporal patterns pass through unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 3.4 Add configurable threshold parameters at module level
    - `FREQUENCY_ENERGY_THRESHOLD = 0.6` (60% energy in 1000-4000 Hz)
    - `HARMONIC_NOISE_RATIO_THRESHOLD = 10.0` (dB)
    - `BURST_DURATION_MIN = 0.5` (seconds)
    - `BURST_DURATION_MAX = 2.0` (seconds)
    - `PAUSE_DURATION_MIN = 0.2` (seconds)
    - `PAUSE_DURATION_MAX = 1.0` (seconds)
    - Allow easy tuning based on testing results
    - _Requirements: 2.5_

  - [x] 3.5 Integrate 5-stage pipeline into run_realtime_monitor detection loop
    - Stage 1: Energy check (existing) - reject silence
    - Stage 2: Frequency analysis - reject if not in baby cry frequency range
    - Stage 3: Spectral pattern matching - reject if harmonic structure doesn't match baby cry
    - Stage 4: Temporal pattern analysis - reject if rhythm doesn't match baby cry pattern
    - Stage 5: Model classification (existing) - classify baby cry type
    - Each rejection stage prints diagnostic message: "Non-baby sound detected (frequency/spectral/temporal) - skipping prediction"
    - Add diagnostic logging to track rejection reasons and measured values
    - _Bug_Condition: isBugCondition(input) where non-baby sounds reach model classification_
    - _Expected_Behavior: Non-baby sounds rejected at stages 2-4 before model classification_
    - _Preservation: Baby cries pass through all stages to model classification unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3_

  - [x] 3.6 Adjust model selection to use best_model_improved_90.pth
    - Modify `_CANDIDATES` priority list to prefer best_model_improved_90.pth
    - This model has better non_cry class training (601 files including cats and dogs)
    - Remove or deprioritize phone_augmented model
    - _Bug_Condition: Wrong model being used may contribute to non_cry class confusion_
    - _Expected_Behavior: Correct model with better non_cry training is loaded_
    - _Preservation: Model inference pipeline remains unchanged_
    - _Requirements: 2.4, 3.1_

  - [x] 3.7 Update app_pytorch.py to use hybrid detection pipeline
    - Import the new pre-filtering functions from realtime_pytorch.py
    - Apply the same 5-stage pipeline before model classification
    - Return "No baby cry detected" message when pre-filtering rejects audio
    - Maintain existing web interface and real-time display functionality
    - _Bug_Condition: Web application also needs pre-filtering to reject non-baby sounds_
    - _Expected_Behavior: Web app displays "No baby cry detected" for non-baby sounds_
    - _Preservation: Web interface and real-time display functionality unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 3.4_

  - [x] 3.8 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Non-baby Sound Rejection
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - Verify cat sounds are rejected (display "No baby cry detected" or classify as "non_cry")
    - Verify dog sounds are rejected (display "No baby cry detected" or classify as "non_cry")
    - Verify user voice is rejected (display "No baby cry detected" or classify as "non_cry")
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 3.9 Verify preservation tests still pass
    - **Property 2: Preservation** - Baby Cry Classification Accuracy
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - Verify all 8 baby cry types still classified correctly (same predictions as unfixed code)
    - Verify confidence levels within 5% of original system
    - Verify silence detection still works for low-energy audio
    - Verify real-time processing latency increase < 100ms
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after fix (no false negatives for baby cries)

- [x] 4. Threshold tuning and validation

  - [x] 4.1 Test with cat sounds from Data/non_cry/ directory
    - Run hybrid detection pipeline with all cat sound files
    - Verify 100% rejection rate (all cat sounds rejected at stages 2-4)
    - Document which stage rejects each cat sound (frequency/spectral/temporal)
    - Measure actual frequency distributions and compare to threshold
    - If any cat sounds pass through, adjust thresholds more conservatively
    - _Requirements: 2.1, 2.5_

  - [x] 4.2 Test with dog sounds from Data/non_cry/ directory
    - Run hybrid detection pipeline with all dog bark files
    - Verify 100% rejection rate (all dog sounds rejected at stages 2-4)
    - Document which stage rejects each dog sound (frequency/spectral/temporal)
    - Measure actual temporal patterns (burst duration should be 0.1-0.3s)
    - If any dog sounds pass through, adjust thresholds more conservatively
    - _Requirements: 2.2, 2.5_

  - [x] 4.3 Test with user voice recordings
    - Record multiple user voice samples (speaking, singing, humming)
    - Run hybrid detection pipeline with all voice recordings
    - Verify 100% rejection rate (all voice samples rejected at stages 2-4)
    - Measure fundamental frequency (should be 85-255 Hz, much lower than baby cries)
    - If any voice samples pass through, adjust thresholds more conservatively
    - _Requirements: 2.3, 2.5_

  - [x] 4.4 Validate baby cry classification is preserved (no false negatives)
    - Test with representative samples from all 8 baby cry types
    - Verify 0% rejection rate (no baby cries rejected by pre-filtering stages)
    - Verify classification accuracy matches original system (within 5% confidence)
    - If any baby cries are rejected, relax thresholds slightly
    - Balance false positive elimination with false negative prevention
    - Document final threshold values and rationale
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [x] 4.5 Test edge cases and borderline sounds
    - Test with high-pitched cat meows (borderline baby-like sounds)
    - Test with baby-like animal sounds (if available)
    - Test with loud background noise (music, TV sounds)
    - Test with distorted or low-quality audio
    - Fine-tune thresholds to handle edge cases correctly
    - Document edge case handling and threshold adjustments
    - _Requirements: 2.4, 2.5_

- [x] 5. Final validation for presentation tomorrow

  - [x] 5.1 Run complete test suite
    - Execute all bug condition exploration tests (should pass)
    - Execute all preservation property tests (should pass)
    - Execute all threshold tuning validation tests (should pass)
    - Verify 100% rejection of non-baby sounds (cat, dog, voice)
    - Verify 0% false negatives for baby cries (all 8 types)
    - _Requirements: All requirements 2.1-3.5_

  - [x] 5.2 Test real-time monitoring with microphone
    - Start realtime_pytorch.py and test with live microphone input
    - Test with cat sounds played through speakers
    - Test with user speaking into microphone
    - Test with actual baby cry recordings
    - Verify real-time display shows correct results
    - Verify diagnostic messages appear for rejected sounds
    - _Requirements: 2.1, 2.2, 2.3, 3.3, 3.4_

  - [x] 5.3 Test web application end-to-end
    - Start app_pytorch.py and access http://localhost:5000
    - Test with microphone input (cat sounds, voice, baby cries)
    - Verify "No baby cry detected" message displays for non-baby sounds
    - Verify baby cry predictions display correctly with guidance
    - Verify UI updates in real-time without lag
    - _Requirements: 2.1, 2.2, 2.3, 3.4_

  - [x] 5.4 Measure and document performance metrics
    - Measure processing latency (time from audio capture to prediction)
    - Verify latency increase < 100ms compared to original system
    - Measure CPU usage during real-time monitoring
    - Document performance characteristics for presentation
    - _Requirements: 3.3_

  - [x] 5.5 Prepare presentation demonstration
    - Prepare cat sound files for live demonstration
    - Prepare dog sound files for live demonstration
    - Prepare user voice demonstration script
    - Prepare baby cry samples for positive validation
    - Test complete demonstration flow end-to-end
    - Document expected results for each demonstration case
    - _Requirements: All requirements 2.1-3.5_

- [x] 6. Checkpoint - Ensure all tests pass and system is ready for presentation
  - Verify all bug condition exploration tests pass (non-baby sounds rejected)
  - Verify all preservation tests pass (baby cries classified correctly)
  - Verify all threshold tuning tests pass (100% rejection, 0% false negatives)
  - Verify real-time monitoring works correctly with microphone
  - Verify web application works correctly end-to-end
  - Verify performance metrics meet requirements (latency < 100ms increase)
  - System is ready for presentation panel demonstration tomorrow
  - Ask the user if questions arise
