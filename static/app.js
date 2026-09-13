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
  zoneConfig: { zones: [], default_required_ppe: ["helmet", "vest", "boots"], violation_threshold_seconds: 2 },
  occupants: [],
  drawing: false,
  draftPoints: [],
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

function setKpis(summary = {}, dangerZone = null) {
  $("kpi-people").textContent = summary.people ?? 0;
  $("kpi-worn").textContent = summary.worn_ppe ?? 0;
  $("kpi-missing").textContent = summary.missing_ppe ?? 0;
  const status = summary.status ?? "idle";
  const el = $("kpi-status");
  el.textContent = status;
  el.className = `status-${status}`;

  const zone = dangerZone || {
    workers_in_zones: summary.workers_in_zones ?? 0,
    active_critical_violations: summary.active_critical_violations ?? 0,
    by_zone: summary.zone_violations || {},
  };
  $("kpi-in-zones").textContent = zone.workers_in_zones ?? 0;
  $("kpi-zone-critical").textContent = zone.active_critical_violations ?? 0;
  const parts = Object.entries(zone.by_zone || {}).map(([name, count]) => `${name}: ${count}`);
  if (!state.zoneConfig.zones?.length) {
    $("kpi-zone-breakdown").textContent = "No zones configured";
  } else {
    $("kpi-zone-breakdown").textContent = parts.length ? parts.join(" · ") : "No active zone violations";
  }
  const box = $("zone-status");
  if (!state.zoneConfig.zones?.length) {
    box.className = "zone-status empty";
    box.textContent = "No danger zones yet. Draw a polygon on the live camera.";
  } else {
    const names = state.zoneConfig.zones.map((z) => z.name).join(", ");
    box.className = zone.active_critical_violations ? "zone-status active" : "zone-status";
    box.textContent = `${state.zoneConfig.zones.length} zone(s): ${names}. Workers inside: ${zone.workers_in_zones ?? 0}. Critical: ${zone.active_critical_violations ?? 0}.`;
  }
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
  const key = alerts.map((a) => `${a.type}:${a.worker_id || ""}:${a.zone_name || ""}`).join(",");
  const now = Date.now();
  if (key === state.lastAlertKey && now - state.lastAlertAt < 2500) return;
  state.lastAlertKey = key;
  state.lastAlertAt = now;
  beep();
  const stamp = new Date().toLocaleTimeString();
  alerts.forEach((alert) => {
    const li = document.createElement("li");
    const extra = alert.zone_name ? ` · ${alert.zone_name}` : "";
    const worker = alert.worker_id ? ` Worker #${alert.worker_id}` : "";
    li.innerHTML = `<span class="sev">${alert.severity}</span><span class="time">${alert.timestamp || stamp}</span><div><strong>${alert.title}</strong> · ${alert.message}${worker}${extra} (${source})</div>`;
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

function layoutVideo(video, canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  if (!srcW || !srcH) return null;
  const scale = Math.min(width / srcW, height / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;
  return { width, height, srcW, srcH, scale, dw, dh, ox, oy };
}

function eventToNorm(event, video, canvas) {
  const layout = layoutVideo(video, canvas);
  if (!layout) return null;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  const nx = (x - layout.ox) / layout.dw;
  const ny = (y - layout.oy) / layout.dh;
  if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return null;
  return [nx, ny];
}

function paintLive(video, canvas, detections) {
  const ctx = canvas.getContext("2d");
  const layout = layoutVideo(video, canvas);
  ctx.fillStyle = "#070905";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!layout) return;
  const { scale, dw, dh, ox, oy, srcW, srcH } = layout;
  ctx.drawImage(video, ox, oy, dw, dh);

  (state.zoneConfig.zones || []).forEach((zone) => {
    const pts = (zone.points || []).map(([x, y]) => {
      const px = x > 1.5 ? (x / (zone.frame_width || srcW)) : x;
      const py = y > 1.5 ? (y / (zone.frame_height || srcH)) : y;
      return [ox + px * srcW * scale, oy + py * srcH * scale];
    });
    if (pts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle = "rgba(255, 90, 74, 0.18)";
    ctx.fill();
    ctx.strokeStyle = "#ff5a4a";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    ctx.fillStyle = "#ffd4cf";
    ctx.font = "12px sans-serif";
    ctx.fillText(`DANGER ZONE · ${zone.name}`, cx - 70, cy);
  });

  if (state.draftPoints.length) {
    const pts = state.draftPoints.map(([x, y]) => [ox + x * srcW * scale, oy + y * srcH * scale]);
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.strokeStyle = "#d4ff4a";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    pts.forEach(([x, y]) => {
      ctx.fillStyle = "#d4ff4a";
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

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

  (state.occupants || []).forEach((occ) => {
    if (!occ.zone_name) return;
    const [x1, y1, x2, y2] = occ.bbox;
    ctx.strokeStyle = occ.confirmed ? "#ff5a4a" : "#ffb020";
    ctx.lineWidth = 3;
    ctx.strokeRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    const fx = ox + occ.foot[0] * scale;
    const fy = oy + occ.foot[1] * scale;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(fx, fy, 4, 0, Math.PI * 2);
    ctx.fill();
    const missing = (occ.missing_ppe || []).join(", ") || "none";
    const label = occ.confirmed
      ? `CRITICAL  Worker #${occ.track_id}  Missing: ${missing}  Danger Zone: ${occ.zone_name}`
      : `Worker #${occ.track_id} in ${occ.zone_name}`;
    ctx.font = "12px sans-serif";
    ctx.fillText(label, ox + x1 * scale + 4, oy + y2 * scale + 14);
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
  state.occupants = [];
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
      state.occupants = data.danger_zone?.occupants || [];
      setKpis(data.summary, data.danger_zone);
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
    setKpis(data.summary, data.danger_zone);
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
    setKpis(data.summary, data.danger_zone);
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

function selectedRequiredPpe() {
  return [...document.querySelectorAll(".req-ppe:checked")].map((el) => el.value);
}

function setDrawMode(on) {
  state.drawing = on;
  $("overlay").classList.toggle("drawing", on);
  $("draw-zone").textContent = on ? "Drawing…" : "Draw danger zone";
  $("undo-zone-point").disabled = !on || state.draftPoints.length === 0;
  $("finish-zone").disabled = !on || state.draftPoints.length < 3;
  $("live-hint").textContent = on
    ? "Click the video to add polygon points. Finish with at least 3 points."
    : "Uses your webcam. Frames stay on this machine.";
}

async function persistZones() {
  const res = await fetch("/api/zones", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.zoneConfig),
  });
  if (!res.ok) throw new Error(await res.text());
  state.zoneConfig = await res.json();
}

async function loadZones() {
  try {
    const res = await fetch("/api/zones");
    if (!res.ok) return;
    state.zoneConfig = await res.json();
    setKpis({}, { workers_in_zones: 0, active_critical_violations: 0, by_zone: {} });
  } catch {
    /* keep empty config */
  }
}

function finishDraftZone() {
  if (state.draftPoints.length < 3) return;
  const video = $("camera");
  const zones = state.zoneConfig.zones || [];
  const name = $("zone-name").value.trim() || `Danger Zone ${zones.length + 1}`;
  zones.push({
    id: `zone_${Date.now()}`,
    name,
    points: state.draftPoints.map(([x, y]) => [x, y]),
    required_ppe: selectedRequiredPpe(),
    severity: "critical",
    frame_width: video.videoWidth || null,
    frame_height: video.videoHeight || null,
  });
  state.zoneConfig.zones = zones;
  state.draftPoints = [];
  setDrawMode(false);
  persistZones().catch((err) => {
    console.error(err);
    alert("Could not save danger zones.");
  });
}

function wireZoneDrawing() {
  const overlay = $("overlay");
  overlay.addEventListener("click", (event) => {
    if (!state.drawing) return;
    const point = eventToNorm(event, $("camera"), overlay);
    if (!point) return;
    state.draftPoints.push(point);
    setDrawMode(true);
  });
  overlay.addEventListener("dblclick", (event) => {
    event.preventDefault();
    if (state.drawing) finishDraftZone();
  });
  $("draw-zone").addEventListener("click", () => {
    if (!state.looping) {
      alert("Start the live camera first, then draw the zone on the video frame.");
      return;
    }
    state.draftPoints = [];
    setDrawMode(!state.drawing);
  });
  $("undo-zone-point").addEventListener("click", () => {
    state.draftPoints.pop();
    setDrawMode(true);
  });
  $("finish-zone").addEventListener("click", finishDraftZone);
  $("clear-zones").addEventListener("click", () => {
    state.zoneConfig.zones = [];
    state.draftPoints = [];
    setDrawMode(false);
    persistZones().catch((err) => console.error(err));
  });
}

wireZoneDrawing();
loadZones();
checkHealth();

