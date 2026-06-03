<script setup lang="ts">
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type RuntimeStatus = "idle" | "loading" | "ready" | "error";
type LandmarkPoint = { x: number; y: number; z?: number };

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
] as const;

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const cameraStatus = ref<RuntimeStatus>("idle");
const modelStatus = ref<RuntimeStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelMessage = ref("等待手势模型加载");
const handCount = ref(0);
const inferenceFps = ref(0);
const lastHandedness = ref("无");

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let handLandmarker: HandLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let latestHands: LandmarkPoint[][] = [];

const debugDetails = computed(() => [
  { label: "手部数量", value: String(handCount.value) },
  { label: "实际推理 FPS", value: inferenceFps.value.toFixed(1) },
  { label: "左右手", value: lastHandedness.value },
  { label: "模型", value: HAND_LANDMARKER_MODEL_URL },
]);

function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function stopCamera() {
  stopRenderLoop();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;

  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }

  handCount.value = 0;
  latestHands = [];
}

function disposeModel() {
  handLandmarker?.close();
  handLandmarker = null;
}

async function loadModel() {
  if (handLandmarker || modelStatus.value === "loading") {
    return;
  }

  modelStatus.value = "loading";
  modelMessage.value = "正在加载 MediaPipe Hand Landmarker";

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_LANDMARKER_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    modelStatus.value = "ready";
    modelMessage.value = "手势模型已就绪。";
  } catch (error) {
    disposeModel();
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error ? error.message : "手势模型加载失败。";
  }
}

function updateInferenceFps(now: number) {
  inferenceFrameCount += 1;

  if (!lastFpsSampleAt) {
    lastFpsSampleAt = now;
    return;
  }

  const elapsed = now - lastFpsSampleAt;

  if (elapsed >= 1000) {
    inferenceFps.value = (inferenceFrameCount * 1000) / elapsed;
    inferenceFrameCount = 0;
    lastFpsSampleAt = now;
  }
}

function drawHands(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "rgba(250, 204, 21, 0.95)";
  context.lineWidth = Math.max(3, canvas.width / 280);

  latestHands.forEach((hand) => {
    HAND_CONNECTIONS.forEach(([fromIndex, toIndex]) => {
      const from = hand[fromIndex];
      const to = hand[toIndex];

      if (!from || !to) {
        return;
      }

      context.beginPath();
      context.moveTo(from.x * canvas.width, from.y * canvas.height);
      context.lineTo(to.x * canvas.width, to.y * canvas.height);
      context.stroke();
    });

    hand.forEach((point) => {
      context.beginPath();
      context.fillStyle = "#ffffff";
      context.strokeStyle = "rgba(15, 23, 42, 0.72)";
      context.lineWidth = 2;
      context.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    });
  });

  context.restore();
}

function runInference(now: number) {
  const video = videoRef.value;

  if (
    !handLandmarker ||
    !video ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    isInferencing ||
    now - lastInferenceStartedAt < MIN_INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  isInferencing = true;
  lastInferenceStartedAt = now;

  try {
    const result = handLandmarker.detectForVideo(video, now);
    latestHands = result.landmarks as LandmarkPoint[][];
    handCount.value = latestHands.length;
    lastHandedness.value =
      result.handednesses
        ?.flat()
        .map((item) => item.categoryName)
        .join(" / ") || "无";
    updateInferenceFps(now);
  } finally {
    isInferencing = false;
  }
}

function renderFrame() {
  const video = videoRef.value;
  const canvas = canvasRef.value;
  const context = canvas?.getContext("2d");

  if (!video || !canvas || !context) {
    return;
  }

  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  runInference(performance.now());
  drawHands(context, canvas);

  animationFrameId = requestAnimationFrame(renderFrame);
}

async function startCamera() {
  stopCamera();
  cameraStatus.value = "loading";
  cameraMessage.value = "正在请求摄像头权限";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });

    if (!videoRef.value) {
      return;
    }

    videoRef.value.srcObject = stream;
    await videoRef.value.play();
    cameraStatus.value = "ready";
    cameraMessage.value = "摄像头已就绪，请举起手。";
    animationFrameId = requestAnimationFrame(renderFrame);
  } catch (error) {
    cameraStatus.value = "error";
    cameraMessage.value =
      error instanceof Error ? error.message : "摄像头启动失败。";
  }
}

onMounted(() => {
  void startCamera();
  void loadModel();
});

onBeforeUnmount(() => {
  stopCamera();
  disposeModel();
});
</script>

<template>
  <main class="single-tool-page">
    <section class="stage-panel" aria-label="手势识别区域">
      <div class="stage-toolbar">
        <div>
          <p class="eyebrow">Gesture</p>
          <h1>手势识别</h1>
        </div>
        <span class="status-pill" :class="`status-${cameraStatus}`">
          {{ cameraStatus }}
        </span>
      </div>

      <div class="camera-stage">
        <div class="camera-frame">
          <video ref="videoRef" aria-label="摄像头原始画面" playsinline muted />
          <canvas ref="canvasRef" aria-label="手部关键点渲染层" />
          <div v-if="cameraStatus !== 'ready'" class="camera-overlay">
            <span>Hand</span>
            <p>{{ cameraMessage }}</p>
          </div>
          <div v-else class="camera-message" aria-live="polite">
            {{ cameraMessage }}
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button type="button" @click="startCamera">开始</button>
        <button type="button" class="secondary" @click="stopCamera">重置</button>
      </div>
    </section>

    <aside class="side-panel" aria-label="手势识别指标">
      <div class="metrics-grid">
        <section class="metric-card">
          <span>手部数量</span>
          <strong>{{ handCount }}</strong>
          <small>最多 2 只手</small>
        </section>
        <section class="metric-card">
          <span>推理 FPS</span>
          <strong>{{ inferenceFps.toFixed(1) }}</strong>
          <small>目标 20 FPS</small>
        </section>
      </div>

      <details class="debug-panel">
        <summary>
          <span>手势模型</span>
          <strong :class="`debug-status status-${modelStatus}`">
            {{ modelStatus }}
          </strong>
        </summary>
        <div class="debug-body">
          <p>{{ modelMessage }}</p>
          <dl>
            <template v-for="item in debugDetails" :key="item.label">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </template>
          </dl>
        </div>
      </details>
    </aside>
  </main>
</template>
