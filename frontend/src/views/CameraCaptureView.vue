<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, shallowRef, watch } from "vue";
import { useRoute } from "vue-router";
import "../session-ui.css";
import { useActionWs } from "../composables/useActionWs";
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;

const route = useRoute();
const ws = useActionWs();

const sessionId = computed(() => String(route.query.session || "default"));
const captureView = computed<"front" | "side">(() =>
  route.query.view === "side" ? "side" : "front",
);
const viewLabel = computed(() => captureView.value === "front" ? "正面" : "侧面");

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const availableDevices = ref<MediaDeviceInfo[]>([]);
const selectedDeviceId = ref<string | null>(null);
const cameraStatus = ref<"idle" | "loading" | "ready" | "error">("idle");
const modelStatus = ref<"idle" | "loading" | "ready" | "error">("idle");
const poseStatus = ref<"idle" | "detected" | "missing">("idle");
const inferenceFps = ref(0);
const inferMs = ref<number | null>(null);
const sentFrames = ref(0);

let stream: MediaStream | null = null;
let animFrameId: number | null = null;
let lastInferenceAt = 0;
let fpsFrameCount = 0;
let lastFpsSampleAt = performance.now();
const pose = shallowRef<PoseLandmarker | null>(null);

function storageKey() {
  return `cameraCaptureDevice:${captureView.value}`;
}

async function enumerateDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableDevices.value = devices.filter(d => d.kind === "videoinput");
    const saved = localStorage.getItem(storageKey());
    if (saved) selectedDeviceId.value = saved;
  } catch (err) {
    console.error("Failed to enumerate devices:", err);
  }
}

async function loadPose() {
  if (pose.value || modelStatus.value === "loading") return;
  modelStatus.value = "loading";
  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    pose.value = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    modelStatus.value = "ready";
  } catch (err) {
    console.error("Failed to load PoseLandmarker:", err);
    modelStatus.value = "error";
  }
}

function extractNamedLandmarks(landmarks: NormalizedLandmark[]) {
  const indices: Record<string, number> = {
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,    RIGHT_WRIST: 16,
    LEFT_HIP: 23,      RIGHT_HIP: 24,
    LEFT_KNEE: 25,     RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,    RIGHT_ANKLE: 28,
  };
  const named: Record<string, { x: number; y: number; z: number; visibility: number }> = {};
  for (const [name, idx] of Object.entries(indices)) {
    const lm = landmarks[idx];
    if (!lm) continue;
    named[name] = {
      x: lm.x,
      y: lm.y,
      z: lm.z ?? 0,
      visibility: lm.visibility ?? 0,
    };
  }
  return named;
}

function drawPose(ctx: CanvasRenderingContext2D, landmarks: NormalizedLandmark[], w: number, h: number) {
  const connections = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [25, 27], [24, 26], [26, 28],
  ];
  ctx.strokeStyle = "#22c55e";
  ctx.fillStyle = "#22c55e";
  ctx.lineWidth = 2;
  for (const [a, b] of connections) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb || (pa.visibility ?? 0) < 0.5 || (pb.visibility ?? 0) < 0.5) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  }
  for (const lm of landmarks) {
    if ((lm.visibility ?? 0) < 0.5) continue;
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateFps(now: number) {
  fpsFrameCount += 1;
  if (now - lastFpsSampleAt >= 1000) {
    inferenceFps.value = Math.round(fpsFrameCount * 1000 / (now - lastFpsSampleAt));
    fpsFrameCount = 0;
    lastFpsSampleAt = now;
  }
}

function captureLoop() {
  const video = videoRef.value;
  const canvas = canvasRef.value;
  const landmarker = pose.value;
  if (!video || !canvas || !landmarker) {
    animFrameId = requestAnimationFrame(captureLoop);
    return;
  }

  const now = performance.now();
  if (now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS) {
    animFrameId = requestAnimationFrame(captureLoop);
    return;
  }
  lastInferenceAt = now;

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    animFrameId = requestAnimationFrame(captureLoop);
    return;
  }

  const ctx = canvas.getContext("2d");
  if (ctx && video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }

  const t = performance.now();
  const result = landmarker.detectForVideo(video, now);
  inferMs.value = Math.round((performance.now() - t) * 10) / 10;
  const landmarks = result.landmarks?.[0] || null;

  if (landmarks) {
    poseStatus.value = "detected";
    if (ctx) drawPose(ctx, landmarks, canvas.width, canvas.height);
    ws.sendViewFrame(
      sessionId.value,
      captureView.value,
      extractNamedLandmarks(landmarks),
      Date.now(),
      now,
    );
    sentFrames.value += 1;
    updateFps(now);
  } else {
    poseStatus.value = "missing";
  }

  animFrameId = requestAnimationFrame(captureLoop);
}

async function startCamera() {
  if (!videoRef.value) return;
  cameraStatus.value = "loading";
  try {
    stopCamera();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: selectedDeviceId.value ? { exact: selectedDeviceId.value } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: captureView.value === "front" ? "user" : undefined,
      },
      audio: false,
    });
    videoRef.value.srcObject = stream;
    await videoRef.value.play();
    cameraStatus.value = "ready";
    await loadPose();
    animFrameId = requestAnimationFrame(captureLoop);
  } catch (err) {
    console.error("Failed to start camera:", err);
    cameraStatus.value = "error";
  }
}

function stopCamera() {
  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
}

watch(selectedDeviceId, (val) => {
  if (val) localStorage.setItem(storageKey(), val);
});

onMounted(async () => {
  ws.connect();
  await enumerateDevices();
  await startCamera();
});

onUnmounted(() => {
  stopCamera();
  pose.value?.close();
  pose.value = null;
  ws.disconnect();
});
</script>

<template>
  <main class="capture-page">
    <section class="capture-main">
      <video ref="videoRef" class="capture-video" playsinline muted aria-hidden="true" />
      <canvas ref="canvasRef" class="capture-canvas" aria-hidden="true" />
      <div class="capture-badge">
        <strong>{{ viewLabel }}采集</strong>
        <span class="cc-mono">{{ sessionId }}</span>
      </div>
    </section>

    <aside class="capture-panel">
      <p class="data-section__label">采集设置</p>
      <label class="capture-field">
        <span>摄像头</span>
        <select v-model="selectedDeviceId" class="cc-select">
          <option :value="null">默认摄像头</option>
          <option
            v-for="device in availableDevices"
            :key="device.deviceId"
            :value="device.deviceId"
          >
            {{ device.label || `摄像头 ${device.deviceId.slice(0, 8)}` }}
          </option>
        </select>
      </label>
      <button class="cc-btn cc-btn--secondary" type="button" @click="startCamera">
        重启摄像头
      </button>

      <div class="capture-divider" />

      <div class="capture-stat">
        <span>前端 FPS</span>
        <strong class="cc-mono">{{ inferenceFps }}</strong>
      </div>
      <div class="capture-stat">
        <span>推理耗时</span>
        <strong class="cc-mono">{{ inferMs ?? "--" }}ms</strong>
      </div>
      <div class="capture-stat">
        <span>发送帧数</span>
        <strong class="cc-mono">{{ sentFrames }}</strong>
      </div>
      <div class="capture-stat">
        <span>摄像头</span>
        <strong>{{ cameraStatus }}</strong>
      </div>
      <div class="capture-stat">
        <span>姿态模型</span>
        <strong>{{ modelStatus }}</strong>
      </div>
      <div class="capture-stat">
        <span>姿态</span>
        <strong>{{ poseStatus }}</strong>
      </div>
      <div class="capture-stat">
        <span>后端</span>
        <strong>{{ ws.status.value }}</strong>
      </div>
    </aside>
  </main>
</template>

<style scoped>
.capture-page {
  min-height: 100vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  background: #0f1117;
  color: #e5e7eb;
  font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.capture-main {
  position: relative;
  overflow: hidden;
  background: #05070c;
}

.capture-video,
.capture-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

.capture-badge {
  position: absolute;
  top: 18px;
  left: 18px;
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(15, 17, 23, 0.78);
  border: 1px solid rgba(255, 255, 255, 0.12);
}

.capture-panel {
  background: #13151f;
  border-left: 1px solid #1e2030;
  padding: 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
}

.capture-field {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 12px;
  color: #9ca3af;
}

.cc-select {
  width: 100%;
  min-width: 0;
  background: #1a1d27;
  border: 1px solid #2a2d3a;
  border-radius: 6px;
  color: #e5e7eb;
  font-size: 12px;
  padding: 8px 10px;
}

.cc-btn {
  background: #22c55e;
  border: none;
  border-radius: 6px;
  color: #0f1117;
  font-weight: 600;
  cursor: pointer;
  padding: 9px 12px;
}

.cc-btn--secondary {
  background: #2a2d3a;
  color: #e5e7eb;
}

.capture-divider {
  height: 1px;
  background: #1e2030;
}

.capture-stat {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
  color: #9ca3af;
}

.capture-stat strong {
  color: #e5e7eb;
}
</style>
