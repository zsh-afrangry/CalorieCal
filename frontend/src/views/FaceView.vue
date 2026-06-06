<script setup lang="ts">
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type RuntimeStatus = "idle" | "loading" | "ready" | "error";
type FaceBox = { originX: number; originY: number; width: number; height: number };
type FaceLandmarkPoint = { x: number; y: number; z?: number };
type HeadMotionLabel = "等待检测" | "静止" | "点头" | "摇头" | "向左转" | "向右转";
type FaceMotionSample = {
  time: number;
  noseX: number;
  noseY: number;
  noseOffsetX: number;
  noseOffsetY: number;
  box: FaceBox;
};

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const MOTION_HISTORY_MS = 1400;
const MIN_HEAD_MOTION_DELTA = 0.008;
const FACE_LANDMARK_INDEX = {
  noseTip: 1,
  forehead: 10,
  chin: 152,
  leftCheek: 234,
  rightCheek: 454,
} as const;

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const cameraStatus = ref<RuntimeStatus>("idle");
const modelStatus = ref<RuntimeStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelMessage = ref("等待面部关键点模型加载");
const faceCount = ref(0);
const inferenceFps = ref(0);
const currentHeadMotion = ref<HeadMotionLabel>("等待检测");
const headDirection = ref("等待检测");
const faceMotionReason = ref("等待人脸入镜");
const noseOffsetState = ref("等待检测");

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let faceLandmarker: FaceLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let latestFaceBoxes: FaceBox[] = [];
let latestFaceLandmarks: FaceLandmarkPoint[][] = [];
const faceMotionHistory: FaceMotionSample[] = [];

const debugDetails = computed(() => [
  { label: "人脸数量", value: String(faceCount.value) },
  { label: "当前头部动作", value: currentHeadMotion.value },
  { label: "运动方向", value: headDirection.value },
  { label: "鼻尖偏移", value: noseOffsetState.value },
  { label: "判断原因", value: faceMotionReason.value },
  { label: "实际推理 FPS", value: inferenceFps.value.toFixed(1) },
  { label: "模型", value: FACE_LANDMARKER_MODEL_URL },
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
  latestFaceLandmarks = [];
  faceMotionHistory.length = 0;
  currentHeadMotion.value = "等待检测";
  headDirection.value = "等待检测";
  faceMotionReason.value = "等待人脸入镜";
  noseOffsetState.value = "等待检测";
}

function disposeModel() {
  faceLandmarker?.close();
  faceLandmarker = null;
}

async function loadModel() {
  if (faceLandmarker || modelStatus.value === "loading") {
    return;
  }

  modelStatus.value = "loading";
  modelMessage.value = "正在加载 MediaPipe Face Landmarker";

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_LANDMARKER_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    modelStatus.value = "ready";
    modelMessage.value = "人脸关键点模型已就绪。";
  } catch (error) {
    disposeModel();
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error ? error.message : "人脸关键点模型加载失败。";
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

function getRange(values: number[]) {
  return Math.max(...values) - Math.min(...values);
}

function countDirectionChanges(values: number[]) {
  let lastSign = 0;
  let changes = 0;

  for (let index = 1; index < values.length; index += 1) {
    const delta = values[index] - values[index - 1];
    const sign =
      Math.abs(delta) < MIN_HEAD_MOTION_DELTA ? 0 : delta > 0 ? 1 : -1;

    if (!sign) {
      continue;
    }

    if (lastSign && sign !== lastSign) {
      changes += 1;
    }

    lastSign = sign;
  }

  return changes;
}

function getFaceBox(landmarks: FaceLandmarkPoint[]): FaceBox {
  const xs = landmarks.map((point) => point.x);
  const ys = landmarks.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return {
    originX: minX,
    originY: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

function getFaceMotionSample(
  landmarks: FaceLandmarkPoint[],
  now: number,
): FaceMotionSample | null {
  const nose = landmarks[FACE_LANDMARK_INDEX.noseTip];

  if (!nose) {
    return null;
  }

  const box = getFaceBox(landmarks);
  const centerX = box.originX + box.width / 2;
  const centerY = box.originY + box.height / 2;

  return {
    time: now,
    noseX: nose.x,
    noseY: nose.y,
    noseOffsetX: box.width ? (nose.x - centerX) / box.width : 0,
    noseOffsetY: box.height ? (nose.y - centerY) / box.height : 0,
    box,
  };
}

function updateFaceMotionAnalysis(now: number, faces: FaceLandmarkPoint[][]) {
  if (!faces.length) {
    faceMotionHistory.length = 0;
    currentHeadMotion.value = "等待检测";
    headDirection.value = "等待检测";
    noseOffsetState.value = "等待检测";
    faceMotionReason.value = "未检测到人脸";
    return;
  }

  const primaryFace = faces[0];
  const sample = getFaceMotionSample(primaryFace, now);

  if (!sample) {
    faceMotionHistory.length = 0;
    currentHeadMotion.value = "等待检测";
    headDirection.value = "等待检测";
    noseOffsetState.value = "等待检测";
    faceMotionReason.value = "鼻尖关键点不足";
    return;
  }

  faceMotionHistory.push(sample);

  while (
    faceMotionHistory.length &&
    now - faceMotionHistory[0].time > MOTION_HISTORY_MS
  ) {
    faceMotionHistory.shift();
  }

  if (faceMotionHistory.length < 6) {
    currentHeadMotion.value = "静止";
    headDirection.value = "采样中";
    noseOffsetState.value = "采样中";
    faceMotionReason.value = "等待更多面部运动帧";
    return;
  }

  const first = faceMotionHistory[0];
  const noseXValues = faceMotionHistory.map((item) => item.noseX);
  const noseYValues = faceMotionHistory.map((item) => item.noseY);
  const offsetXValues = faceMotionHistory.map((item) => item.noseOffsetX);
  const xRange = getRange(noseXValues);
  const yRange = getRange(noseYValues);
  const xDirectionChanges = countDirectionChanges(noseXValues);
  const yDirectionChanges = countDirectionChanges(noseYValues);
  const xDelta = sample.noseX - first.noseX;
  const yDelta = sample.noseY - first.noseY;
  const offsetXRange = getRange(offsetXValues);

  headDirection.value =
    Math.abs(xDelta) > Math.abs(yDelta)
      ? xDelta > 0
        ? "向画面右侧"
        : "向画面左侧"
      : yDelta > 0
        ? "向下"
        : "向上";
  noseOffsetState.value = sample.noseOffsetX.toFixed(2);

  if (yRange > 0.055 && yDirectionChanges >= 1 && yRange > xRange * 1.15) {
    currentHeadMotion.value = "点头";
    faceMotionReason.value = `鼻尖纵向变化 ${yRange.toFixed(2)}，换向 ${yDirectionChanges} 次`;
    return;
  }

  if (xRange > 0.055 && xDirectionChanges >= 1 && xRange > yRange * 1.15) {
    currentHeadMotion.value = "摇头";
    faceMotionReason.value = `鼻尖横向变化 ${xRange.toFixed(2)}，换向 ${xDirectionChanges} 次`;
    return;
  }

  if (sample.noseOffsetX < -0.055 && offsetXRange < 0.08) {
    currentHeadMotion.value = "向左转";
    faceMotionReason.value = `鼻尖相对面部中心偏左 ${sample.noseOffsetX.toFixed(2)}`;
    return;
  }

  if (sample.noseOffsetX > 0.055 && offsetXRange < 0.08) {
    currentHeadMotion.value = "向右转";
    faceMotionReason.value = `鼻尖相对面部中心偏右 ${sample.noseOffsetX.toFixed(2)}`;
    return;
  }

  currentHeadMotion.value = "静止";
  faceMotionReason.value =
    `横向 ${xRange.toFixed(2)}，纵向 ${yRange.toFixed(2)}，鼻尖偏移 ${sample.noseOffsetX.toFixed(2)}`;
}

function drawFaces(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  context.save();
  context.strokeStyle = "rgba(56, 189, 248, 0.95)";
  context.lineWidth = Math.max(3, canvas.width / 260);

  latestFaceBoxes.forEach((box) => {
    context.strokeRect(
      box.originX * canvas.width,
      box.originY * canvas.height,
      box.width * canvas.width,
      box.height * canvas.height,
    );
  });

  context.fillStyle = "rgba(255, 255, 255, 0.9)";
  latestFaceLandmarks.forEach((face) => {
    [
      FACE_LANDMARK_INDEX.noseTip,
      FACE_LANDMARK_INDEX.forehead,
      FACE_LANDMARK_INDEX.chin,
      FACE_LANDMARK_INDEX.leftCheek,
      FACE_LANDMARK_INDEX.rightCheek,
    ].forEach((index) => {
      const point = face[index];

      if (!point) {
        return;
      }

      context.beginPath();
      context.arc(point.x * canvas.width, point.y * canvas.height, 4, 0, Math.PI * 2);
      context.fill();
    });
  });

  context.restore();
}

function runInference(now: number) {
  const video = videoRef.value;

  if (
    !faceLandmarker ||
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
    const result = faceLandmarker.detectForVideo(video, now);
    latestFaceLandmarks = result.faceLandmarks as FaceLandmarkPoint[][];
    latestFaceBoxes = latestFaceLandmarks.map(getFaceBox);
    faceCount.value = latestFaceLandmarks.length;
    updateFaceMotionAnalysis(now, latestFaceLandmarks);
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
    <section class="stage-panel" aria-label="头部动作识别区域">
      <div class="stage-toolbar">
        <div>
          <p class="eyebrow">Face</p>
          <h1>头部动作识别</h1>
        </div>
        <span class="status-pill" :class="`status-${cameraStatus}`">
          {{ cameraStatus }}
        </span>
      </div>

      <div class="camera-stage">
        <div class="camera-frame">
          <video ref="videoRef" aria-label="摄像头原始画面" playsinline muted />
          <canvas ref="canvasRef" aria-label="面部关键点渲染层" />
          <div v-if="cameraStatus !== 'ready'" class="camera-overlay">
            <span>Face</span>
            <p>{{ cameraMessage }}</p>
          </div>
          <div v-else class="camera-message" aria-live="polite">
            {{ cameraMessage }}
          </div>
          <div v-if="cameraStatus === 'ready'" class="motion-state-badge">
            <span>头部动作</span>
            <strong>{{ currentHeadMotion }}</strong>
            <small>{{ headDirection }} / 鼻尖 {{ noseOffsetState }}</small>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button type="button" @click="startCamera">开始</button>
        <button type="button" class="secondary" @click="stopCamera">重置</button>
      </div>
    </section>

    <aside class="side-panel" aria-label="头部动作识别指标">
      <div class="metrics-grid">
        <section class="metric-card">
          <span>当前动作</span>
          <strong>{{ currentHeadMotion }}</strong>
          <small>{{ faceMotionReason }}</small>
        </section>
        <section class="metric-card">
          <span>人脸数量</span>
          <strong>{{ faceCount }}</strong>
          <small>面部关键点</small>
        </section>
        <section class="metric-card">
          <span>推理 FPS</span>
          <strong>{{ inferenceFps.toFixed(1) }}</strong>
          <small>目标 20 FPS</small>
        </section>
      </div>

      <details class="debug-panel">
        <summary>
          <span>面部模型</span>
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
