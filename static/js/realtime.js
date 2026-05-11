/**
 * realtime.js — Real-time voice conversion via SocketIO + WebAudio API
 *
 * Flow:
 *   getUserMedia → AudioWorkletNode (capture PCM) → buffer → SocketIO → server
 *   server → FreeVC → SocketIO → AudioContext (playback)
 */
(function () {
  "use strict";

  const SAMPLE_RATE = 16000;       // Must match config.REALTIME_SAMPLE_RATE
  const BUFFER_DURATION_S = 1.5;  // Accumulate this many seconds before sending
  const BUFFER_SIZE = Math.floor(SAMPLE_RATE * BUFFER_DURATION_S);

  // ── DOM references ───────────────────────────────────────────────────────
  const btnStart      = document.getElementById("btn-start");
  const btnStop       = document.getElementById("btn-stop");
  const selProfile    = document.getElementById("rt-sel-profile");
  const statusDot     = document.getElementById("status-dot");
  const statusText    = document.getElementById("status-text");
  const latencyLabel  = document.getElementById("latency-label");
  const latencyBar    = document.getElementById("latency-bar");
  const inputLevel    = document.getElementById("input-level");
  const outputLevel   = document.getElementById("output-level");
  const logPanel      = document.getElementById("log-panel");
  const canvas        = document.getElementById("waveform-canvas");
  const ctx2d         = canvas.getContext("2d");

  // ── State ─────────────────────────────────────────────────────────────────
  let socket         = null;
  let audioCtx       = null;
  let mediaStream    = null;
  let sourceNode     = null;
  let workletNode    = null;
  let analyser       = null;
  let isRunning      = false;
  let pcmBuffer      = new Int16Array(0);   // accumulation buffer
  let sendTimestamp  = 0;
  let animFrame      = null;

  // ── Logging ───────────────────────────────────────────────────────────────
  function log(msg, cls = "text-muted") {
    const el = document.createElement("div");
    el.className = `small ${cls}`;
    el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logPanel.appendChild(el);
    logPanel.scrollTop = logPanel.scrollHeight;
  }
  window.clearLog = () => { logPanel.innerHTML = ""; };

  // ── Status helpers ────────────────────────────────────────────────────────
  function setStatus(text, dotClass = "bg-secondary") {
    statusText.textContent = text;
    statusDot.className = `status-dot ${dotClass}`;
  }

  // ── SocketIO connection ───────────────────────────────────────────────────
  function connectSocket() {
    socket = io({ transports: ["websocket"] });

    socket.on("connect", () => {
      log("Đã kết nối server.", "text-success");
      setStatus("Kết nối OK", "bg-success");
      // Request model loading
      socket.emit("load_models");
    });

    socket.on("disconnect", () => {
      log("Mất kết nối server.", "text-danger");
      setStatus("Mất kết nối", "bg-danger");
      if (isRunning) stopCapture();
    });

    socket.on("vc_status", (data) => {
      log(`Status: ${data.status}${data.message ? " — " + data.message : ""}`);
      if (data.status === "ready") {
        setStatus("Sẵn sàng", "bg-success");
        btnStart.disabled = false;
        log("Model đã sẵn sàng!", "text-success");
      } else if (data.status === "loading") {
        setStatus("Đang tải model…", "bg-warning text-dark");
        btnStart.disabled = true;
      } else if (data.status === "error") {
        setStatus("Lỗi model", "bg-danger");
        log(`Lỗi: ${data.message}`, "text-danger");
      }
    });

    socket.on("converted", (data) => {
      const now = performance.now();
      const latency = Math.round(now - sendTimestamp);
      latencyLabel.textContent = `${latency} ms`;
      const pct = Math.min(100, (latency / 3000) * 100);
      latencyBar.style.width = `${pct}%`;
      latencyBar.className = `progress-bar ${latency < 800 ? "bg-success" : latency < 1500 ? "bg-warning" : "bg-danger"}`;

      const rawBytes = data.audio;
      playAudioBytes(rawBytes);
    });
  }

  // ── Start / Stop ──────────────────────────────────────────────────────────
  btnStart.addEventListener("click", async () => {
    if (!selProfile.value) { alert("Vui lòng chọn Voice Profile."); return; }
    await startCapture();
  });

  btnStop.addEventListener("click", () => stopCapture());

  async function startCapture() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
    } catch (err) {
      log(`Không thể mở micro: ${err.message}`, "text-danger");
      return;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    await audioCtx.audioWorklet.addModule("/static/js/recorder-worklet.js");

    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-recorder");
    analyser    = audioCtx.createAnalyser();
    analyser.fftSize = 512;

    sourceNode.connect(workletNode);
    sourceNode.connect(analyser);

    workletNode.port.onmessage = onWorkletMessage;

    if (!socket || !socket.connected) connectSocket();

    isRunning = true;
    btnStart.classList.add("d-none");
    btnStop.classList.remove("d-none");
    setStatus("Đang thu âm…", "active bg-success");
    statusDot.classList.add("active");
    log("Bắt đầu thu âm.", "text-info");

    animFrame = requestAnimationFrame(drawWaveform);
  }

  function stopCapture() {
    isRunning = false;
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (sourceNode)  { sourceNode.disconnect();  sourceNode  = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (audioCtx)    { audioCtx.close(); audioCtx = null; }
    if (animFrame)   { cancelAnimationFrame(animFrame); animFrame = null; }

    pcmBuffer = new Int16Array(0);
    btnStop.classList.add("d-none");
    btnStart.classList.remove("d-none");
    statusDot.classList.remove("active");
    setStatus("Đã dừng", "bg-secondary");
    inputLevel.style.width = "0%";
    outputLevel.style.width = "0%";
    ctx2d.clearRect(0, 0, canvas.width, canvas.height);
    log("Đã dừng.", "text-warning");
  }

  // ── AudioWorklet message handler ──────────────────────────────────────────
  function onWorkletMessage(event) {
    if (!isRunning || !socket || !socket.connected) return;

    // event.data is Float32Array from worklet
    const float32 = event.data;
    const int16 = float32ToInt16(float32);

    // Update input level
    const rms = Math.sqrt(float32.reduce((s, x) => s + x * x, 0) / float32.length);
    inputLevel.style.width = `${Math.min(100, rms * 300)}%`;

    // Accumulate in buffer
    const newBuf = new Int16Array(pcmBuffer.length + int16.length);
    newBuf.set(pcmBuffer);
    newBuf.set(int16, pcmBuffer.length);
    pcmBuffer = newBuf;

    if (pcmBuffer.length >= BUFFER_SIZE) {
      const toSend = pcmBuffer.buffer.slice(0, BUFFER_SIZE * 2);
      pcmBuffer = pcmBuffer.slice(BUFFER_SIZE);
      sendTimestamp = performance.now();
      socket.emit("audio_chunk", {
        audio: toSend,
        profile_id: selProfile.value,
      });
    }
  }

  // ── Playback converted audio ──────────────────────────────────────────────
  let playbackCtx = null;
  let nextPlayTime = 0;

  async function playAudioBytes(rawBytes) {
    if (!playbackCtx) {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      nextPlayTime = playbackCtx.currentTime;
    }

    let int16Array;
    if (rawBytes instanceof ArrayBuffer) {
      int16Array = new Int16Array(rawBytes);
    } else if (rawBytes instanceof Uint8Array || ArrayBuffer.isView(rawBytes)) {
      int16Array = new Int16Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength / 2);
    } else {
      // Might be a plain object / binary string from socket.io
      const ab = new Uint8Array(Object.values(rawBytes)).buffer;
      int16Array = new Int16Array(ab);
    }

    const float32 = int16ToFloat32(int16Array);

    // Level meter
    const rms = Math.sqrt(float32.reduce((s, x) => s + x * x, 0) / float32.length);
    outputLevel.style.width = `${Math.min(100, rms * 300)}%`;

    const buffer = playbackCtx.createBuffer(1, float32.length, SAMPLE_RATE);
    buffer.copyToChannel(float32, 0);
    const src = playbackCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(playbackCtx.destination);

    const startAt = Math.max(playbackCtx.currentTime, nextPlayTime);
    src.start(startAt);
    nextPlayTime = startAt + buffer.duration;
  }

  // ── Waveform visualizer ───────────────────────────────────────────────────
  function drawWaveform() {
    if (!isRunning || !analyser) return;
    animFrame = requestAnimationFrame(drawWaveform);

    const bufLen = analyser.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyser.getByteTimeDomainData(data);

    const W = canvas.offsetWidth;
    const H = canvas.height;
    canvas.width = W;

    ctx2d.fillStyle = "#1a1a2e";
    ctx2d.fillRect(0, 0, W, H);

    ctx2d.lineWidth = 1.5;
    ctx2d.strokeStyle = "#4fc3f7";
    ctx2d.beginPath();

    const step = W / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = data[i] / 128.0;
      const y = (v * H) / 2;
      i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
      x += step;
    }
    ctx2d.stroke();
  }

  // ── PCM conversion helpers ────────────────────────────────────────────────
  function float32ToInt16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 32768 : s * 32767;
    }
    return out;
  }

  function int16ToFloat32(i16) {
    const out = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) out[i] = i16[i] / 32768.0;
    return out;
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  // Connect socket immediately so model loading starts on page load
  connectSocket();
})();
