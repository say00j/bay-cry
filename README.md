# Baby Cry Detection System - Properly Trained Model

## Overview

This is a baby cry detection system that classifies baby cries into 9 categories using a CNN-based deep learning model. The system is designed to work with real-world audio input from a BM-800 microphone.

## Features

- **9 Cry Types:** belly_pain, burping, cold_hot, discomfort, hungry, lonely, scared, silence, tired
- **Real-time Detection:** Monitors microphone input and classifies cries every 5 seconds
- **Web Interface:** Flask-based web UI for monitoring and control
- **High Accuracy:** Trained on 6,520 audio samples with proper validation
- **Phone-to-Mic Support:** Works with audio played from phone through microphone

## System Requirements

### Hardware
- **Microphone:** BM-800 or similar condenser microphone
- **GPU:** NVIDIA GPU with CUDA support (recommended) or CPU
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** 5 GB free space

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

## Installation

1. **Install Python dependencies:**
```bash
pip install torch torchaudio numpy librosa sounddevice scikit-learn matplotlib seaborn flask scipy
```

2. **Verify GPU support (optional but recommended):**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Training the Model

### Step 1: Prepare Dataset

Ensure your audio files are organized in the `Data/` folder:
```
Data/
├── belly_pain/*.wav
├── burping/*.wav
├── cold_hot/*.wav
├── discomfort/*.wav
├── hungry/*.wav
├── lonely/*.wav
├── scared/*.wav
├── silence/*.wav
└── tired/*.wav
```

### Step 2: Train the Model

Run the training script:
```bash
python train_model_properly.py
```

**Training time:**
- With GPU: 30-60 minutes
- With CPU: 3-5 hours

**Expected output:**
- `best_model_retrained.pth` - Trained model (target: 80%+ accuracy)
- `label_classes_retrained.npy` - Class labels
- `normalization_params_retrained.npy` - Normalization parameters
- `training_history_retrained.csv` - Training metrics
- `training_curves_retrained.png` - Training visualization
- `confusion_matrix_retrained.png` - Performance matrix

### Step 3: Verify Model Performance

The training script automatically validates the model on real audio files. Look for:

```
Real-World Accuracy: X/90 = 0.XXX (XX.X%)

Per-class requirements (minimum 70%):
  ✓ belly_pain  : 0.XXX >= 0.70 (PASS)
  ✓ burping     : 0.XXX >= 0.70 (PASS)
  ...
```

**Success criteria:**
- Overall accuracy >= 80%
- Each class accuracy >= 70%

## Usage

### Real-time Detection (Command Line)

1. **Connect your BM-800 microphone** to your laptop
2. **Run the detection script:**
```bash
python realtime_pytorch.py
```

3. **Play baby cry audio from your phone** near the microphone
4. **The system will detect and classify** the cry type every 5 seconds

**Output example:**
```
Listening...
Pre-filtering: PASS (frequency: 0.45, HNR: -8.2 dB, valid_bursts: 2)

Detected: hungry (confidence: 0.87, energy: 0.0234)
Suggested parental guidance (not medical advice):
- What you can try now: Offer a feed if it is close to or past feeding time...
- When to consult a doctor: Consult a doctor if baby refuses feeds repeatedly...
```

### Web Interface

1. **Start the web server:**
```bash
python app_pytorch.py
```

2. **Open your browser:**
```
http://localhost:5000
```

3. **Features:**
   - Start/stop monitoring
   - View real-time detections
   - See detection history
   - Send email alerts (requires configuration)

### Email Notifications (Optional)

Set environment variables for email alerts:

**Windows (Command Prompt):**
```cmd
set CRY_EMAIL=your-email@gmail.com
set CRY_APP_PASSWORD=your-app-password
set CRY_TO_EMAIL=recipient@email.com
```

**Windows (PowerShell):**
```powershell
$env:CRY_EMAIL="your-email@gmail.com"
$env:CRY_APP_PASSWORD="your-app-password"
$env:CRY_TO_EMAIL="recipient@email.com"
```

**Note:** For Gmail, create an [App Password](https://support.google.com/accounts/answer/185833).

## Testing

### Test Model on Audio Files

```bash
python test_model_on_files.py
```

This tests the model on actual WAV files from the dataset to verify real-world performance.

### Investigate Model Behavior

```bash
python investigate_model_bias.py
```

This analyzes prediction distribution and identifies any class bias issues.

### Analyze Model Weights

```bash
python analyze_model_weights.py
```

This examines the model's internal weights to diagnose training issues.

## Phone-to-Mic Setup

### Hardware Setup

1. **Connect BM-800 microphone:**
   - Plug USB cable into laptop
   - Ensure microphone is powered on
   - Check Windows recognizes the device

2. **Position microphone:**
   - Place mic 20-30 cm from phone speaker
   - Avoid background noise
   - Adjust mic gain if needed

3. **Phone audio:**
   - Play baby cry audio at moderate volume
   - Use clear, high-quality recordings
   - Avoid speaker distortion

### Audio Quality Tips

- **Volume:** Set phone volume to 60-80% (not max)
- **Distance:** Keep phone 20-30 cm from mic
- **Environment:** Quiet room with minimal echo
- **Mic gain:** Adjust BM-800 gain knob to avoid clipping
- **Test:** Run `python realtime_pytorch.py` and check RMS energy values

## Troubleshooting

### Model Not Detecting Cries

**Problem:** System shows "Stage 1 - Low energy detected"
**Solution:**
- Increase phone volume
- Move phone closer to microphone
- Check microphone is connected and working
- Adjust BM-800 gain knob

**Problem:** System shows "Stage 2/3/4 - Non-baby sound detected"
**Solution:**
- These are pre-filtering stages rejecting non-baby sounds
- If rejecting real baby cries, thresholds may be too strict
- Check `INVESTIGATION_REPORT.md` for threshold tuning

### Low Accuracy

**Problem:** Model predicts wrong cry types
**Solution:**
1. Check model file: `best_model_retrained.pth` exists
2. Verify training completed successfully
3. Check real-world validation results from training
4. Retrain with more epochs if accuracy < 80%

### Microphone Issues

**Problem:** No audio input detected
**Solution:**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
- Verify BM-800 appears in device list
- Check Windows microphone permissions
- Test with Windows Sound Recorder

### GPU Not Detected

**Problem:** Training uses CPU instead of GPU
**Solution:**
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```
- Install CUDA toolkit
- Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Model Architecture

**Type:** Deep CNN with Global Average Pooling

**Structure:**
- 4 CNN blocks (64 → 128 → 256 → 512 filters)
- Batch normalization after each conv layer
- Dropout (0.2-0.5) for regularization
- Global Average Pooling
- 3 dense layers (512 → 256 → 128 → 9 classes)
- LeakyReLU activation

**Parameters:** ~5.1M trainable parameters

**Input:** Mel spectrogram (128 mel bins, ~157 time frames)
- Sample rate: 16000 Hz
- Duration: 5 seconds
- FFT size: 2048
- Hop length: 512
- Frequency range: 20-8000 Hz

**Normalization:** Per-sample robust normalization (median + MAD)

## Project Structure

```
.
├── Data/                          # Training audio files (6,520 samples)
├── .kiro/                         # Specs and configuration
├── train_model_properly.py        # Training script
├── realtime_pytorch.py            # Real-time detection
├── app_pytorch.py                 # Web interface
├── test_model_on_files.py         # Validation script
├── investigate_model_bias.py      # Bias analysis
├── analyze_model_weights.py       # Weight analysis
├── INVESTIGATION_REPORT.md        # Detailed investigation findings
├── MODEL_ACCURACY_REPORT.md       # Accuracy analysis
└── README.md                      # This file
```

## Medical Disclaimer

⚠️ **This system provides suggestions only and is NOT a substitute for professional medical advice.** Always consult a healthcare provider for medical concerns about your baby.

## License

This project is for educational and research purposes.

## Support

For issues or questions:
1. Check `INVESTIGATION_REPORT.md` for detailed analysis
2. Review training output and validation results
3. Test with `test_model_on_files.py`
4. Verify microphone setup with `realtime_pytorch.py`

---

**Last Updated:** 2026-02-25
**Model Version:** Retrained (properly validated)
**Target Accuracy:** 80%+ real-world, 70%+ per-class
