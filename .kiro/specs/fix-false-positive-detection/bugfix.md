# Bugfix Requirements Document

## Introduction

The baby cry detection system is incorrectly classifying non-baby sounds (cat sounds, user voice, dog barking) as baby cry types instead of correctly identifying them as "non_cry" or showing "No baby cry detected". This is a critical issue as the presentation panel will test the system with cat sounds tomorrow, and the system must correctly reject non-baby sounds to demonstrate proper functionality.

The system has a trained "non_cry" class with 601 files including cat sounds (167 files) and dog barking (113 files), but the model is not predicting this class correctly during real-time detection. Instead, it assigns random cry classes like "burping", "cold_hot", "discomfort", or "scared" to non-baby audio inputs.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN user speaks into the microphone THEN the system predicts baby cry classes such as "burping", "cold_hot", or other cry types instead of "non_cry"

1.2 WHEN cat sounds are played (including files from Data/non_cry/ directory) THEN the system predicts random baby cry classes such as "scared", "discomfort", or other cry types instead of "non_cry"

1.3 WHEN dog barking sounds are played THEN the system predicts baby cry classes instead of "non_cry"

1.4 WHEN any non-baby sound is input to the system THEN the model fails to correctly identify it as "non_cry" or show "No baby cry detected"

### Expected Behavior (Correct)

2.1 WHEN user speaks into the microphone THEN the system SHALL show "No baby cry detected" or classify as "non_cry"

2.2 WHEN cat sounds are played (including files from Data/non_cry/ directory) THEN the system SHALL show "No baby cry detected" or classify as "non_cry"

2.3 WHEN dog barking sounds are played THEN the system SHALL show "No baby cry detected" or classify as "non_cry"

2.4 WHEN any non-baby sound is input to the system THEN the system SHALL correctly identify it as "non_cry" or show "No baby cry detected" message

2.5 WHEN non-baby sounds are detected through the hybrid detection pipeline (energy check, frequency analysis, spectral pattern matching, temporal pattern analysis) THEN the system SHALL reject the audio before model classification and show "No baby cry detected"

### Unchanged Behavior (Regression Prevention)

3.1 WHEN actual baby cry sounds are played (belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, tired) THEN the system SHALL CONTINUE TO correctly classify the specific cry type

3.2 WHEN silence or very low energy audio is input THEN the system SHALL CONTINUE TO classify as "silence" or show appropriate low-energy detection message

3.3 WHEN audio is captured from the laptop microphone at 48000 Hz sample rate THEN the system SHALL CONTINUE TO process and resample the audio correctly to 16000 Hz for model input

3.4 WHEN the web application (app_pytorch.py) receives audio data THEN the system SHALL CONTINUE TO perform real-time classification and display results on http://localhost:5000

3.5 WHEN the real-time detection script (realtime_pytorch.py) processes microphone input THEN the system SHALL CONTINUE TO extract Mel spectrogram features with 128 mel bins and perform classification
