let audioContext, mediaStream, sourceNode, processorNode;
let recording = false;
let pcmData = [];
let sampleRate = 22050;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const timerEl = document.getElementById("timer");
const playback = document.getElementById("playback");
const audioFile = document.getElementById("audioFile");
const audioFileUpload = document.getElementById("audioFileUpload");
const submitBtn = document.getElementById("submitBtn");
const uploadedFileDisplay = document.getElementById("uploadedFileDisplay");
const uploadedFileName = document.getElementById("uploadedFileName");

let t0 = null;
let timerInt = null;

function pad2(n){ return String(n).padStart(2,"0"); }

function setTimer(sec){
  const m = Math.floor(sec/60), s = Math.floor(sec%60);
  timerEl.textContent = `${pad2(m)}:${pad2(s)}`;
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function writeWav(int16Data, sr) {
  const buffer = new ArrayBuffer(44 + int16Data.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  // RIFF header
  writeString(0, "RIFF");
  view.setUint32(4, 36 + int16Data.length * 2, true);
  writeString(8, "WAVE");

  // fmt chunk
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);          // chunk size
  view.setUint16(20, 1, true);           // PCM
  view.setUint16(22, 1, true);           // mono
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);      // byte rate (sr * blockAlign)
  view.setUint16(32, 2, true);           // block align
  view.setUint16(34, 16, true);          // bits per sample

  // data chunk
  writeString(36, "data");
  view.setUint32(40, int16Data.length * 2, true);

  let offset = 44;
  for (let i = 0; i < int16Data.length; i++, offset += 2) {
    view.setInt16(offset, int16Data[i], true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function enableSubmit() {
    if (audioFileUpload.files.length > 0 || audioFile.files.length > 0) {
        submitBtn.disabled = false;
    }
}

function handleAudioFile(file) {
    if (file && file.type.startsWith("audio/")) {
        const dt = new DataTransfer();
        dt.items.add(file);
        audioFile.files = dt.files;

        audioFileUpload.style.display = 'none';
        uploadedFileName.textContent = file.name;
        uploadedFileDisplay.style.display = 'block';

        playback.src = URL.createObjectURL(file);
        enableSubmit();
    }
}

if (audioFileUpload) {
    audioFileUpload.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            handleAudioFile(file);
        }
    });
}

async function startRecording() {
  pcmData = [];
  recording = true;
  submitBtn.disabled = true;

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  sampleRate = audioContext.sampleRate;

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  processorNode = audioContext.createScriptProcessor(4096, 1, 1);

  processorNode.onaudioprocess = (e) => {
    if (!recording) return;
    const ch = e.inputBuffer.getChannelData(0);
    pcmData.push(new Float32Array(ch));
  };

  sourceNode.connect(processorNode);
  processorNode.connect(audioContext.destination);

  // timer
  t0 = Date.now();
  timerInt = setInterval(() => {
    const sec = (Date.now() - t0) / 1000;
    setTimer(sec);
  }, 250);
}

async function stopRecording() {
  recording = false;
  if (timerInt) clearInterval(timerInt);

  // stop nodes
  if (processorNode) processorNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioContext) await audioContext.close();

  // concat PCM
  const total = pcmData.reduce((a, b) => a + b.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const chunk of pcmData) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  const int16 = floatTo16BitPCM(merged);
  const wavBlob = writeWav(int16, sampleRate);

  // create a File and place into hidden <input type=file>
  const file = new File([wavBlob], "recording.wav", { type: "audio/wav" });
  const dt = new DataTransfer();
  dt.items.add(file);
  audioFile.files = dt.files;

  if (uploadedFileDisplay) {
    uploadedFileDisplay.style.display = 'none';
  }
  if (audioFileUpload) {
    audioFileUpload.style.display = 'block';
    audioFileUpload.value = '';
  }

  // playback preview
  playback.src = URL.createObjectURL(wavBlob);
  enableSubmit();
}

if (startBtn && stopBtn) {
  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setTimer(0);
    await startRecording();
  });

  stopBtn.addEventListener("click", async () => {
    stopBtn.disabled = true;
    startBtn.disabled = false;
    await stopRecording();
  });
}
