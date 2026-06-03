<script setup lang="ts">
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type RuntimeStatus = "idle" | "loading" | "ready" | "error";
type FaceBox = { originX: number; originY: number; width: number; height: number };

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const FACE_DETECTOR_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const cameraStatus = ref<RuntimeStatus>("idle");
const modelStatus = ref<RuntimeStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelMessage = ref("等待人脸检测模型加载");
const faceCount = ref(0);
const inferenceFps = ref(0);

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let faceDetector: FaceDetector | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let latestFaceBoxes: FaceBox[] = [];

const debugDetails = computed(() => [
  { label: "人脸数量", value: String(faceCount.value) },
  { label: "实际推理 FPS", value: inferenceFps.value.toFixed(1) },
  { label: "模型", value: FACE_DETECTOR_MODEL_URL },
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

  faceCount.value = 0;
  latestFaceBoxes = [];
}

function disposeModel() {
  faceDetector?.close();
  faceDetector = null;
}

async function loadModel() {
  if (faceDetector || modelStatus.value === "loading") {
    return;
  }

  modelStatus.value = "loading";
  modelMessage.value = "正在加载 MediaPipe Face Detector";

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_DETECTOR_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
    });
    modelStatus.value = "ready";
    modelMessage.value = "人脸检测模型已就绪。";
  } catch (error) {
    disposeModel();
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error ? error.message : "人脸检测模型加载失败。";
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

function drawFaces(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  context.save();
  context.strokeStyle = "rgba(56, 189, 248, 0.95)";
  context.lineWidth = Math.max(3, canvas.width / 260);

  latestFaceBoxes.forEach((box) => {
    context.strokeRect(box.originX, box.originY, box.width, box.height);
  });

  context.restore();
}

function runInference(now: number) {
  const video = videoRef.value;

  if (
    !faceDetector ||
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
    const result = faceDetector.detectForVideo(video, now);
    latestFaceBoxes = result.detections
      .map((detection) => detection.boundingBox)
      .filter((box): box is FaceBox => Boolean(box));
    faceCount.value = latestFaceBoxes.length;
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
  drawFaces(context, canvas);

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
    cameraMessage.value = "摄像头已就绪，请正对摄像头。";
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
    <section class="stage-panel" aria-label="人脸检测区域">
      <div class="stage-toolbar">
        <div>
          <p class="eyebrow">Face</p>
          <h1>人脸检测</h1>
        </div>
        <span class="status-pill" :class="`status-${cameraStatus}`">
          {{ cameraStatus }}
        </span>
      </div>

      <div class="camera-stage">
        <div class="camera-frame">
          <video ref="videoRef" aria-label="摄像头原始画面" playsinline muted />
          <canvas ref="canvasRef" aria-label="人脸检测渲染层" />
          <div v-if="cameraStatus !== 'ready'" class="camera-overlay">
            <span>Face</span>
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

    <aside class="side-panel" aria-label="人脸检测指标">
      <div class="metrics-grid">
        <section class="metric-card">
          <span>人脸数量</span>
          <strong>{{ faceCount }}</strong>
          <small>检测框</small>
        </section>
        <section class="metric-card">
          <span>推理 FPS</span>
          <strong>{{ inferenceFps.toFixed(1) }}</strong>
          <small>目标 20 FPS</small>
        </section>
      </div>

      <details class="debug-panel">
        <summary>
          <span>人脸模型</span>
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
