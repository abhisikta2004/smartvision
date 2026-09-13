const CLASS_COLORS = {
  helmet: "#2ecc71",
  gloves: "#27ae60",
  vest: "#f1c40f",
  boots: "#3498db",
  goggles: "#1abc9c",
  Person: "#5dade2",
  none: "#95a5a6",
  no_helmet: "#e74c3c",
  no_goggle: "#c0392b",
  no_gloves: "#e67e22",
  no_boots: "#d35400",
};

const state = {
  view: "live",
  stream: null,
  looping: false,
  busy: false,
  lastAlertKey: "",
  lastAlertAt: 0,
  imageFile: null,
  videoFile: null,
  liveDetections: [],
};

const $ = (id) => document.getElementById(id);

function confValue() {
  return Number($("conf").value);
}

function heuristicsOn() {
  return $("heuristics").checked;
}

function appendForm(file, filename) {
  const fd = new FormData();
  fd.append("file", file, filename);
  fd.append("conf", String(confValue()));
  fd.append("heuristics", heuristicsOn() ? "true" : "false");
  return fd;
}

function setKpis(summary = {}) {
  $("kpi-people").textContent = summary.people ?? 0;
  $("kpi-worn").textContent = summary.worn_ppe ?? 0;
  $("kpi-missing").textContent = summary.missing_ppe ?? 0;
  const status = summary.status ?? "idle";
  const el = $("kpi-status");
  el.textContent = status;
  el.className = `status-${status}`;
}

function beep() {
  if (!$("sound").checked) return;
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "square";
  osc.frequency.value = 880;
  gain.gain.value = 0.04;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.12);
}

function logAlerts(alerts, source) {
  if (!alerts?.length) return;
  const key = alerts.map((a) => a.type).join(",");
  const now = Date.now();
  if (key === state.lastAlertKey && now - state.lastAlertAt < 2500) return;
  state.lastAlertKey = key;
  state.lastAlertAt = now;
  beep();
  const stamp = new Date().toLocaleTimeString();
  alerts.forEach((alert) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="sev">${alert.severity}</span><span class="time">${stamp}</span><div><strong>${alert.title}</strong> · ${alert.message} (${source})</div>`;
    $("alert-log").prepend(li);
  });
}

function renderLiveAlerts(alerts) {
  const box = $("alert-live");
  if (!alerts?.length) {
    box.className = "alert-live empty";
    box.textContent = "No violations in the current frame.";
    return;
  }
  const top = alerts[0];
  box.className = `alert-live ${top.severity}`;
  box.innerHTML = alerts
    .map((a) => `<div><strong>${a.title}</strong> — ${a.message}${a.count > 1 ? ` ×${a.count}` : ""}</div>`)
    .join("");
}

function paintLive(video, canvas, detections) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.fillStyle = "#070905";
  ctx.fillRect(0, 0, width, height);
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!srcW || !srcH) return;
  const scale = Math.min(width / srcW, height / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;
  ctx.drawImage(video, ox, oy, dw, dh);
  (detections || []).forEach((det) => {
    const [x1, y1, x2, y2] = det.bbox;
    const color = CLASS_COLORS[det.class_name] || "#d4ff4a";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    ctx.fillStyle = color;
    ctx.font = "12px sans-serif";
    ctx.fillText(`${det.class_name} ${det.confidence.toFixed(2)}`, ox + x1 * scale + 4, oy + y1 * scale - 6);
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const el = $("model-status");
    if (data.model_ready) {
      el.textContent = "Model ready";
    } else {
      el.textContent = "Train model first";
      el.style.color = "#ffb020";
    }
  } catch {
    $("model-status").textContent = "API offline";
  }
}

async function startCamera() {
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  const video = $("camera");
  video.srcObject = state.stream;
  await video.play();
  $("start-cam").disabled = true;
  $("stop-cam").disabled = false;
  $("live-banner").textContent = "Monitoring";
  state.looping = true;
  state.liveDetections = [];
  previewLive();
  loopLive();
}

function stopCamera() {
  state.looping = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  state.liveDetections = [];
  $("start-cam").disabled = false;
  $("stop-cam").disabled = true;
  $("live-banner").textContent = "Camera idle";
}

function previewLive() {
  const video = $("camera");
  const overlay = $("overlay");
  const tick = () => {
    if (!state.looping) return;
    paintLive(video, overlay, state.liveDetections);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function loopLive() {
  const video = $("camera");
  while (state.looping) {
    if (state.busy || video.readyState < 2) {
      await new Promise((r) => setTimeout(r, 80));
      continue;
    }
    state.busy = true;
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth || 640;
    frame.height = video.videoHeight || 480;
    frame.getContext("2d").drawImage(video, 0, 0);
    const blob = await new Promise((r) => frame.toBlob(r, "image/jpeg", 0.72));
    try {
      const res = await fetch("/api/detect/frame", { method: "POST", body: appendForm(blob, "frame.jpg") });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      state.liveDetections = data.detections || [];
      setKpis(data.summary);
      renderLiveAlerts(data.alerts);
      logAlerts(data.alerts, "live");
      $("live-banner").textContent = `${data.summary.status} · ${data.latency_ms} ms`;
    } catch (err) {
      $("live-banner").textContent = "Detection error";
      console.error(err);
      await new Promise((r) => setTimeout(r, 800));
    } finally {
      state.busy = false;
    }
  }
}

async function inspectImage() {
  if (!state.imageFile) return;
  $("inspect-image").disabled = true;
  $("inspect-image").textContent = "Inspecting…";
  try {
    const res = await fetch("/api/detect/image", { method: "POST", body: appendForm(state.imageFile, state.imageFile.name) });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $("image-preview").src = data.annotated_url;
    $("image-drop").classList.add("hidden");
    $("image-download").href = data.annotated_url;
    $("image-download").classList.remove("hidden");
    setKpis(data.summary);
    renderLiveAlerts(data.alerts);
    logAlerts(data.alerts, "image");
  } catch (err) {
    alert("Image inspection failed. Train the model if weights are missing.");
    console.error(err);
  } finally {
    $("inspect-image").disabled = false;
    $("inspect-image").textContent = "Run inspection";
  }
}

async function inspectVideo() {
  if (!state.videoFile) return;
  $("inspect-video").disabled = true;
  $("inspect-video").textContent = "Scanning…";
  $("video-hint").textContent = "Scanning clip. This can take a minute.";
  try {
    const res = await fetch("/api/detect/video", { method: "POST", body: appendForm(state.videoFile, state.videoFile.name) });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $("video-preview").src = data.annotated_url;
    $("video-drop").classList.add("hidden");
    $("video-download").href = data.annotated_url;
    $("video-download").classList.remove("hidden");
    setKpis(data.summary);
    renderLiveAlerts(data.alerts);
    logAlerts(data.alerts, "video");
    $("video-hint").textContent = `${data.summary.processed_frames} frames scored · ${data.alerts.length} alert types`;
  } catch (err) {
    alert("Video scan failed. Train the model if weights are missing.");
    console.error(err);
  } finally {
    $("inspect-video").disabled = false;
    $("inspect-video").textContent = "Scan video";
  }
}

function showView(name) {
  state.view = name;
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  ["live", "image", "video"].forEach((id) => {
    $(`view-${id}`).classList.toggle("hidden", id !== name);
  });
}

function isImageFile(file) {
  if (!file) return false;
  if (file.type && file.type.startsWith("image/")) return true;
  return /\.(jpe?g|png|gif|webp|bmp|tif|tiff|heic|heif)$/i.test(file.name || "");
}

function fileFromDataTransfer(dataTransfer) {
  if (!dataTransfer) return null;
  const files = [];
  if (dataTransfer.files?.length) files.push(...dataTransfer.files);
  if (!files.length && dataTransfer.items) {
    for (const item of dataTransfer.items) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
  }
  return files.find(isImageFile) || null;
}

function draggingFiles(event) {
  const types = event.dataTransfer?.types;
  return !!types && [...types].includes("Files");
}

function acceptImageFile(file) {
  if (!isImageFile(file)) return;
  state.imageFile = file;
  $("image-preview").src = URL.createObjectURL(file);
  $("image-drop").classList.add("hidden");
  $("inspect-image").disabled = false;
}

function wireImageDrop() {
  const stage = $("image-stage");
  const highlight = (on) => stage.classList.toggle("dragover", on);

  const onDragOver = (event) => {
    if (!draggingFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    highlight(true);
  };

  window.addEventListener("dragenter", (event) => {
    if (!draggingFiles(event)) return;
    event.preventDefault();
  });
  window.addEventListener("dragover", (event) => {
    if (!draggingFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    if (state.view === "image") highlight(true);
  });
  window.addEventListener("dragleave", (event) => {
    if (event.relatedTarget) return;
    highlight(false);
  });
  window.addEventListener("drop", (event) => {
    if (!draggingFiles(event)) return;
    event.preventDefault();
    highlight(false);
    const file = fileFromDataTransfer(event.dataTransfer);
    if (!file) return;
    showView("image");
    acceptImageFile(file);
  });

  ["dragenter", "dragover"].forEach((name) => stage.addEventListener(name, onDragOver));
  stage.addEventListener("dragleave", (event) => {
    if (!stage.contains(event.relatedTarget)) highlight(false);
  });
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    highlight(false);
    const file = fileFromDataTransfer(event.dataTransfer);
    if (file) acceptImageFile(file);
  });
}

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.dataset.view));
});

$("conf").addEventListener("input", () => {
  $("conf-value").textContent = Number($("conf").value).toFixed(2);
});
$("start-cam").addEventListener("click", () => startCamera().catch((err) => {
  alert("Could not open the camera.");
  console.error(err);
}));
$("stop-cam").addEventListener("click", stopCamera);
$("inspect-image").addEventListener("click", inspectImage);
$("inspect-video").addEventListener("click", inspectVideo);
$("sample-image").addEventListener("click", async () => {
  const res = await fetch("/samples/missing-ppe.jpg");
  const blob = await res.blob();
  acceptImageFile(new File([blob], "missing-ppe.jpg", { type: "image/jpeg" }));
  await inspectImage();
});
$("clear-alerts").addEventListener("click", () => {
  $("alert-log").innerHTML = "";
  renderLiveAlerts([]);
});

wireImageDrop();
$("image-input").addEventListener("change", () => {
  if ($("image-input").files[0]) acceptImageFile($("image-input").files[0]);
});
$("video-input").addEventListener("change", () => {
  if ($("video-input").files[0]) {
    const file = $("video-input").files[0];
    state.videoFile = file;
    $("video-preview").src = URL.createObjectURL(file);
    $("video-drop").classList.add("hidden");
    $("inspect-video").disabled = false;
  }
});

checkHealth();
