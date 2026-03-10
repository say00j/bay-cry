/* ══════════════════════════════════════════════════════
   Baby Cry — app.js
   Sections:
     1. Shared utilities (toast, feed)
     2. Server-monitor polling (/status, /start, /stop, etc.)
     3. Live Mic Recorder (MediaRecorder → /predict-audio)
   ══════════════════════════════════════════════════════ */

/* ── 1. Shared utilities ─────────────────────────────── */
const statusText = document.getElementById('statusText');
const feedList = document.getElementById('feedList');
const toast = document.getElementById('toast');
let lastDetectionStamp = null;

function showToast(msg, duration = 3000) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), duration);
}

function makeFeedItem(entry) {
  const li = document.createElement('li');
  const meta = document.createElement('div'); meta.className = 'feed-meta';
  const label = document.createElement('div'); label.className = 'label'; label.textContent = entry.label;
  const time = document.createElement('div'); time.className = 'time'; time.textContent = `${entry.date} ${entry.time}`;
  meta.appendChild(label); meta.appendChild(time);

  const right = document.createElement('div'); right.className = 'feed-right';
  const conf = document.createElement('div'); conf.className = 'conf'; conf.textContent = (entry.confidence || '').toString();
  const guidance = document.createElement('div'); guidance.className = 'guidance';
  guidance.textContent = entry.guidance ? entry.guidance.advice : '';
  right.appendChild(conf); right.appendChild(guidance);

  li.appendChild(meta); li.appendChild(right);
  return li;
}

function prependDetectionToFeed(entry, highlight = false) {
  const placeholder = document.querySelector('#feedList .empty');
  if (placeholder) placeholder.remove();

  const li = makeFeedItem(entry);
  feedList.insertBefore(li, feedList.firstChild);
  if (highlight) {
    li.classList.add('new');
    setTimeout(() => li.classList.remove('new'), 1200);
  }
  while (feedList.children.length > 50) feedList.removeChild(feedList.lastChild);
}

/* ── 2. Server-monitor polling ───────────────────────── */
async function fetchStatus() {
  try {
    const res = await fetch('/status');
    if (!res.ok) throw new Error('Status request failed');
    return await res.json();
  } catch (e) { console.error(e); return null; }
}

async function getStatusAndUpdate() {
  const data = await fetchStatus();
  if (!data) return;
  statusText.textContent = data.running ? 'Running' : 'Stopped';
  statusText.className = data.running ? 'status-badge' : 'status-badge muted';
  if (data.last_detection) {
    const stamp = `${data.last_detection.date} ${data.last_detection.time}`;
    if (stamp !== lastDetectionStamp) {
      prependDetectionToFeed(data.last_detection, true);
      lastDetectionStamp = stamp;
      showToast('New detection: ' + data.last_detection.label);
    }
  }
}

async function refreshFeed() {
  try {
    const res = await fetch('/logs?n=10');
    const arr = await res.json();
    feedList.innerHTML = '';
    if (!arr || arr.length === 0) {
      const li = document.createElement('li'); li.className = 'empty';
      li.textContent = 'No detections yet.'; feedList.appendChild(li); return;
    }
    arr.reverse().forEach(e => prependDetectionToFeed(e, false));
  } catch (e) { console.error(e); }
}

// Server-monitor buttons
document.getElementById('startBtn').addEventListener('click', async () => {
  const email_on = document.getElementById('email_on_detect').checked;
  const throttle = parseInt(document.getElementById('throttle').value || '60', 10);
  const res = await fetch('/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email_on_detect: email_on, throttle_seconds: throttle })
  });
  const json = await res.json();
  showToast(json.started ? 'Monitor started' : 'Monitor already running');
  getStatusAndUpdate();
});

document.getElementById('stopBtn').addEventListener('click', async () => {
  const res = await fetch('/stop', { method: 'POST' });
  const json = await res.json();
  showToast(json.stopped ? 'Monitor stopped' : 'Monitor was not running');
  getStatusAndUpdate();
});

document.getElementById('sendReportBtn').addEventListener('click', async () => {
  try {
    const res = await fetch('/send-report', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({})
    });
    let json = null;
    try { json = await res.json(); } catch { json = { sent: res.ok }; }
    if (json && json.sent) showToast('Report sent');
    else showToast('Failed: ' + (json && (json.reason || json.error) || 'unknown'));
  } catch (e) { console.error(e); showToast('Failed to send report (network)'); }
});

document.getElementById('refreshBtn').addEventListener('click', refreshFeed);

// Polling
getStatusAndUpdate();
setInterval(getStatusAndUpdate, 3000);
setInterval(refreshFeed, 10000);


/* ── 3. Live Mic Recorder ────────────────────────────── */
// Uses Web Audio API ScriptProcessor to capture RAW PCM samples,
// then encodes a proper WAV file client-side before uploading.
// This bypasses the WebM/Opus-only limitation of MediaRecorder on Chrome/Edge.

const micToggleBtn = document.getElementById('micToggleBtn');
const micStatus = document.getElementById('micStatus');
const livePredCard = document.getElementById('livePredCard');
const livePredLabel = document.getElementById('livePredLabel');
const livePredConf = document.getElementById('livePredConf');
const confBarFill = document.getElementById('confBarFill');
const livePredGuid = document.getElementById('livePredGuidance');
const allProbsWrap = document.getElementById('allProbsWrap');
const waveCanvas = document.getElementById('waveformCanvas');
const pulseDot = document.getElementById('pulseDot');
const waveCtx = waveCanvas.getContext('2d');

const SAMPLE_RATE = 16000;  // target sample rate sent to server
const CHUNK_SECS = 5;      // seconds per inference chunk

let mediaStream = null;
let audioCtx = null;
let analyserNode = null;
let scriptNode = null;
let waveAnimFrame = null;
let isRecording = false;
let pcmBuffer = [];        // accumulates Float32 samples
let sampleCount = 0;

// ── WAV encoder ─────────────────────────────────────
function float32ArrayToWav(samples, sampleRate) {
  // Convert Float32 PCM → 16-bit PCM WAV ArrayBuffer
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);  // PCM chunk size
  view.setUint16(20, 1, true);  // PCM format
  view.setUint16(22, 1, true);  // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);  // block align
  view.setUint16(34, 16, true);  // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }
  return buffer;
}

// ── Downsample helper ────────────────────────────────
function downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio = fromRate / toRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let offsetResult = 0, offsetBuffer = 0;
  while (offsetResult < newLen) {
    const nextOffset = Math.round((offsetResult + 1) * ratio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffset && i < buffer.length; i++) {
      accum += buffer[i]; count++;
    }
    result[offsetResult] = accum / (count || 1);
    offsetResult++;
    offsetBuffer = nextOffset;
  }
  return result;
}

// ── Waveform drawing ─────────────────────────────────
function drawWaveform() {
  if (!analyserNode) return;
  const W = waveCanvas.width, H = waveCanvas.height;
  const bufLen = analyserNode.fftSize;
  const dataArr = new Float32Array(bufLen);
  analyserNode.getFloatTimeDomainData(dataArr);

  waveCtx.clearRect(0, 0, W, H);
  const grad = waveCtx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#6C5CE7');
  grad.addColorStop(0.5, '#00cec9');
  grad.addColorStop(1, '#6C5CE7');
  waveCtx.beginPath();
  waveCtx.strokeStyle = grad;
  waveCtx.lineWidth = 2;
  const sliceW = W / bufLen;
  let x = 0;
  for (let i = 0; i < bufLen; i++) {
    const y = ((dataArr[i] + 1) / 2) * H;
    if (i === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
    x += sliceW;
  }
  waveCtx.stroke();
  waveAnimFrame = requestAnimationFrame(drawWaveform);
}

function drawFlatLine() {
  const W = waveCanvas.width, H = waveCanvas.height;
  waveCtx.clearRect(0, 0, W, H);
  waveCtx.beginPath();
  waveCtx.strokeStyle = 'rgba(108,92,231,0.25)';
  waveCtx.lineWidth = 1.5;
  waveCtx.moveTo(0, H / 2); waveCtx.lineTo(W, H / 2);
  waveCtx.stroke();
}
drawFlatLine();

// ── Send WAV blob to server ──────────────────────────
async function sendWavBlob(samples, sampleRate) {
  if (samples.length < sampleRate * 0.5) return; // skip if < 0.5 s
  setMicStatus('🔄 Processing…', false);

  const wavBuf = float32ArrayToWav(samples, sampleRate);
  const blob = new Blob([wavBuf], { type: 'audio/wav' });
  const fd = new FormData();
  fd.append('audio', blob, 'chunk.wav');

  try {
    const res = await fetch('/predict-audio', { method: 'POST', body: fd });
    const json = await res.json();
    if (json.error) {
      console.error('Server error:', json.error);
      setMicStatus('⚠ ' + json.error, true);
      return;
    }
    showPrediction(json);
    setMicStatus('🎤 Recording — listening…', true);
  } catch (e) {
    console.error('Upload failed:', e);
    setMicStatus('⚠ Upload failed — still recording', true);
  }
}

// ── Display prediction result ────────────────────────
function showPrediction(json) {
  const label = json.label || 'unknown';
  const conf = json.confidence || 0;

  livePredLabel.textContent = label.replace(/_/g, ' ');
  livePredConf.textContent = `${(conf * 100).toFixed(1)}%`;
  confBarFill.style.width = `${(conf * 100).toFixed(1)}%`;

  const guidance = json.guidance;
  if (guidance && guidance.advice) {
    livePredGuid.innerHTML =
      `<strong style="color:#a29bfe">💡 What to try:</strong> ${guidance.advice}<br>` +
      (guidance.doctor ? `<strong style="color:#fd79a8">🩺 Doctor:</strong> ${guidance.doctor}` : '');
  } else {
    livePredGuid.textContent = '';
  }

  if (json.all_probs) {
    const entries = Object.entries(json.all_probs).sort((a, b) => b[1] - a[1]);
    allProbsWrap.innerHTML = '';
    entries.forEach(([cls, prob]) => {
      const row = document.createElement('div'); row.className = 'prob-row';
      const name = document.createElement('span'); name.textContent = cls.replace(/_/g, ' ');
      const bg = document.createElement('div'); bg.className = 'prob-bar-bg';
      const fill = document.createElement('div');
      fill.className = 'prob-bar-fill' + (cls === label ? ' top' : '');
      fill.style.width = `${(prob * 100).toFixed(1)}%`;
      bg.appendChild(fill);
      const pct = document.createElement('span'); pct.textContent = `${(prob * 100).toFixed(0)}%`;
      row.appendChild(name); row.appendChild(bg); row.appendChild(pct);
      allProbsWrap.appendChild(row);
    });
    allProbsWrap.classList.add('visible');
  }

  livePredCard.classList.add('visible');
  const now = new Date();
  prependDetectionToFeed({
    label,
    confidence: conf.toFixed(4),
    date: now.toISOString().slice(0, 10),
    time: now.toTimeString().slice(0, 8),
    guidance
  }, true);
  showToast(`🎙️ Live: ${label.replace(/_/g, ' ')} (${(conf * 100).toFixed(0)}%)`);
}

// ── Status helper ────────────────────────────────────
function setMicStatus(msg, active) {
  micStatus.textContent = msg;
  micStatus.className = active ? 'active' : '';
  pulseDot.className = active ? 'pulse-dot active' : 'pulse-dot';
}

// ── Start recording ──────────────────────────────────
async function startMic() {
  if (isRecording) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (e) {
    showToast('❌ Microphone access denied');
    setMicStatus('❌ Mic access denied', false);
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const nativeRate = audioCtx.sampleRate; // e.g. 44100 or 48000

  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 2048;

  const source = audioCtx.createMediaStreamSource(mediaStream);
  source.connect(analyserNode);

  // ScriptProcessorNode to capture raw PCM (bufferSize 4096)
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  scriptNode.onaudioprocess = (e) => {
    const chunk = e.inputBuffer.getChannelData(0); // Float32Array
    // Downsample on the fly to SAMPLE_RATE
    const downsampled = downsample(chunk, nativeRate, SAMPLE_RATE);
    for (let i = 0; i < downsampled.length; i++) pcmBuffer.push(downsampled[i]);
    sampleCount += downsampled.length;

    // When we have CHUNK_SECS worth, send it
    if (sampleCount >= SAMPLE_RATE * CHUNK_SECS) {
      const toSend = new Float32Array(pcmBuffer.splice(0, SAMPLE_RATE * CHUNK_SECS));
      sampleCount -= SAMPLE_RATE * CHUNK_SECS;
      sendWavBlob(toSend, SAMPLE_RATE);
    }
  };

  source.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);

  isRecording = true;
  pcmBuffer = [];
  sampleCount = 0;
  micToggleBtn.textContent = '⏹ Stop Live Mic';
  micToggleBtn.classList.add('recording');
  setMicStatus('🎤 Recording — listening…', true);
  drawWaveform();
}

// ── Stop recording ───────────────────────────────────
function stopMic() {
  if (!isRecording) return;
  if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (waveAnimFrame) { cancelAnimationFrame(waveAnimFrame); waveAnimFrame = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  analyserNode = null;
  pcmBuffer = [];
  sampleCount = 0;
  isRecording = false;

  micToggleBtn.textContent = '▶ Start Live Mic';
  micToggleBtn.classList.remove('recording');
  setMicStatus('Stopped — click to start', false);
  drawFlatLine();
}

micToggleBtn.addEventListener('click', () => {
  if (isRecording) stopMic(); else startMic();
});


/* ── 4. File Upload Handler ─────────────────────────────── */
const audioFileInput = document.getElementById('audioFileInput');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');

uploadBtn.addEventListener('click', async () => {
  const file = audioFileInput.files[0];
  if (!file) {
    showToast('Please select an audio file first', 2000);
    return;
  }

  uploadStatus.textContent = 'Uploading...';
  uploadBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('audio', file);

    const response = await fetch('/predict-audio', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok) {
      uploadStatus.textContent = `✓ Predicted: ${data.label}`;
      uploadStatus.style.color = '#00b894';
      
      // Display result in the live prediction card
      displayLivePrediction(data);
      
      // Add to feed
      const entry = {
        label: data.label,
        confidence: data.confidence ? (data.confidence * 100).toFixed(1) + '%' : 'N/A',
        date: new Date().toLocaleDateString(),
        time: new Date().toLocaleTimeString(),
        guidance: data.guidance || null
      };
      prependDetectionToFeed(entry, true);
      
      showToast(`File uploaded: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`, 3000);
    } else {
      uploadStatus.textContent = `✗ Error: ${data.error || 'Unknown error'}`;
      uploadStatus.style.color = '#d63031';
      showToast(`Upload failed: ${data.error || 'Unknown error'}`, 3000);
    }
  } catch (err) {
    uploadStatus.textContent = `✗ Network error`;
    uploadStatus.style.color = '#d63031';
    showToast('Network error during upload', 3000);
    console.error('Upload error:', err);
  } finally {
    uploadBtn.disabled = false;
    setTimeout(() => {
      uploadStatus.textContent = '';
    }, 5000);
  }
});

// Helper function to display prediction (reuse existing display logic)
function displayLivePrediction(data) {
  const card = document.getElementById('livePredCard');
  const labelEl = document.getElementById('livePredLabel');
  const confEl = document.getElementById('livePredConf');
  const barFill = document.getElementById('confBarFill');
  const guidanceEl = document.getElementById('livePredGuidance');
  const allProbsWrap = document.getElementById('allProbsWrap');

  card.classList.add('visible');
  
  if (data.label === 'no_baby_detected') {
    labelEl.textContent = 'No Baby Cry Detected';
    confEl.textContent = data.reason || 'filtered';
    barFill.style.width = '0%';
    guidanceEl.textContent = '';
    allProbsWrap.classList.remove('visible');
  } else {
    labelEl.textContent = data.label.replace(/_/g, ' ');
    const confPct = (data.confidence * 100).toFixed(1);
    confEl.textContent = confPct + '%';
    barFill.style.width = confPct + '%';
    
    if (data.guidance && data.guidance.advice) {
      guidanceEl.innerHTML = `<strong>Advice:</strong> ${data.guidance.advice}`;
    } else {
      guidanceEl.textContent = '';
    }

    // Show all probabilities
    if (data.all_probs && Object.keys(data.all_probs).length > 0) {
      allProbsWrap.classList.add('visible');
      const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);
      allProbsWrap.innerHTML = sorted.map(([cls, prob], idx) => {
        const pct = (prob * 100).toFixed(1);
        const topClass = idx === 0 ? 'top' : '';
        return `
          <div class="prob-row">
            <span>${cls.replace(/_/g, ' ')}</span>
            <div class="prob-bar-bg"><div class="prob-bar-fill ${topClass}" style="width:${pct}%"></div></div>
            <span>${pct}%</span>
          </div>
        `;
      }).join('');
    } else {
      allProbsWrap.classList.remove('visible');
    }
  }
}
