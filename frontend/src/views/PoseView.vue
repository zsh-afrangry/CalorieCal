<script setup lang="ts">
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type CameraStatus = "idle" | "loading" | "ready" | "error";
type ModelStatus = "idle" | "loading" | "ready" | "error";
type PoseStatus = "idle" | "detecting" | "detected" | "missing";
type JumpingJackState = "unknown" | "closed" | "open";

type MetricCard = {
  label: string;
  value: string;
  detail?: string;
};

type LandmarkPoint = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
};

const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: {
    facingMode: "user",
    width: { ideal: 1280 },
    height: { ideal: 720 },
  },
  audio: false,
};

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const SMOOTHING_WINDOW_SIZE = 5;
const REQUIRED_LANDMARK_INDICES = [11, 12, 15, 16, 23, 24, 27, 28];
const MIN_LANDMARK_VISIBILITY = 0.5;
const LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;
const MIN_SHOULDER_WIDTH = 0.04;
const FOOT_OPEN_SHOULDER_RATIO = 1.15;
const FOOT_CLOSED_SHOULDER_RATIO = 0.85;
const WRIST_OPEN_SHOULDER_RATIO = 1.15;
const WRIST_CLOSED_SHOULDER_RATIO = 1.0;
const ARM_OPEN_MAX_TORSO_RATIO = 0.25;
const ARM_CLOSED_MIN_TORSO_RATIO = 0.38;
const ACTION_DEBOUNCE_FRAMES = 3;
const POSE_CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 29],
  [29, 31],
  [28, 30],
  [30, 32],
] as const;

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const cameraStatus = ref<CameraStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelStatus = ref<ModelStatus>("idle");
const modelMessage = ref("等待姿态模型加载");
const poseStatus = ref<PoseStatus>("idle");
const inferenceFps = ref(0);
const landmarkCompleteness = ref(0);
const rawLandmarkState = ref("等待推理");
const smoothedLandmarkState = ref("等待平滑数据");
const feetState = ref<JumpingJackState>("unknown");
const armsState = ref<JumpingJackState>("unknown");
const instantJumpingJackState = ref<JumpingJackState>("unknown");
const jumpingJackState = ref<JumpingJackState>("unknown");
const actionStateReason = ref("等待完整关键点");
const actionDebounceProgress = ref("0/3");

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let smoothedLandmarks: LandmarkPoint[] | null = null;
let pendingJumpingJackState: JumpingJackState = "unknown";
let pendingJumpingJackFrames = 0;
const landmarkHistory: LandmarkPoint[][] = [];

const metrics = computed<MetricCard[]>(() => [
  { label: "次数", value: "0", detail: "开合跳" },
  {
    label: "动作",
    value: jumpingJackState.value,
    detail: `脚 ${feetState.value} / 手 ${armsState.value}`,
  },
  { label: "时长", value: "00:00", detail: "运动计时" },
  { label: "频率", value: "0.0", detail: "次/分钟" },
  { label: "MET", value: "6.0", detail: "当前档位" },
  { label: "热量", value: "0.0", detail: "kcal" },
  { label: "置信度", value: "低", detail: "等待人体入镜" },
]);

const modelDetails = computed(() => [
  { label: "tasks-vision", value: MEDIAPIPE_TASKS_VERSION },
  { label: "wasm", value: MEDIAPIPE_WASM_URL },
  { label: "pose model", value: POSE_LANDMARKER_MODEL_URL },
]);

const poseDebugDetails = computed(() => [
  { label: "人体检测", value: poseStatus.value },
  { label: "关键点完整率", value: `${landmarkCompleteness.value.toFixed(0)}%` },
  { label: "实际推理 FPS", value: inferenceFps.value.toFixed(1) },
  { label: "目标推理 FPS", value: `${TARGET_INFERENCE_FPS}` },
  { label: "显示镜像", value: "video/canvas scaleX(-1)" },
  { label: "算法坐标", value: "MediaPipe 原始坐标" },
  { label: "原始关键点", value: rawLandmarkState.value },
  { label: "平滑关键点", value: smoothedLandmarkState.value },
  { label: "脚部状态", value: feetState.value },
  { label: "手臂状态", value: armsState.value },
  { label: "即时动作状态", value: instantJumpingJackState.value },
  { label: "稳定动作状态", value: jumpingJackState.value },
  { label: "防抖进度", value: actionDebounceProgress.value },
  { label: "状态判断", value: actionStateReason.value },
]);

function getCameraErrorMessage(error: unknown) {
  if (!(error instanceof DOMException)) {
    return "摄像头启动失败，请检查浏览器和设备状态。";
  }

  if (error.name === "NotAllowedError") {
    return "摄像头权限被拒绝，请在浏览器中允许摄像头访问。";
  }

  if (error.name === "NotFoundError") {
    return "未检测到可用摄像头设备。";
  }

  if (error.name === "NotReadableError") {
    return "摄像头正在被其他应用占用。";
  }

  return "摄像头启动失败，请确认当前页面运行在 localhost 或 HTTPS 环境。";
}

function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function stopCamera() {
  stopRenderLoop();

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }

  poseStatus.value = "idle";
  inferenceFps.value = 0;
  landmarkCompleteness.value = 0;
  rawLandmarkState.value = "等待推理";
  smoothedLandmarkState.value = "等待平滑数据";
  landmarkHistory.length = 0;
  smoothedLandmarks = null;
  resetJumpingJackAnalysis("等待完整关键点");
}

function disposePoseLandmarker() {
  poseLandmarker?.close();
  poseLandmarker = null;
}

async function loadPoseLandmarker() {
  if (poseLandmarker || modelStatus.value === "loading") {
    return;
  }

  modelStatus.value = "loading";
  modelMessage.value = "正在加载 MediaPipe Pose Landmarker";

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_LANDMARKER_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });

    modelStatus.value = "ready";
    modelMessage.value = "姿态模型已就绪，下一步将接入逐帧推理。";
  } catch (error) {
    disposePoseLandmarker();
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error
        ? `姿态模型加载失败：${error.message}`
        : "姿态模型加载失败，请检查网络和模型资源路径。";
  }
}

function toLandmarkPoints(landmarks: LandmarkPoint[]) {
  return landmarks.map((landmark) => ({
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
    visibility: landmark.visibility,
  }));
}

function updateSmoothedLandmarks(landmarks: LandmarkPoint[]) {
  landmarkHistory.push(toLandmarkPoints(landmarks));

  if (landmarkHistory.length > SMOOTHING_WINDOW_SIZE) {
    landmarkHistory.shift();
  }

  smoothedLandmarks = landmarks.map((_, index) => {
    const points = landmarkHistory
      .map((frame) => frame[index])
      .filter((point): point is LandmarkPoint => Boolean(point));
    const count = points.length || 1;

    return {
      x: points.reduce((sum, point) => sum + point.x, 0) / count,
      y: points.reduce((sum, point) => sum + point.y, 0) / count,
      z: points.reduce((sum, point) => sum + (point.z ?? 0), 0) / count,
      visibility:
        points.reduce((sum, point) => sum + (point.visibility ?? 1), 0) /
        count,
    };
  });
}

function updateLandmarkCompleteness(landmarks: LandmarkPoint[]) {
  const visibleCount = REQUIRED_LANDMARK_INDICES.filter((index) => {
    const landmark = landmarks[index];
    return (
      landmark &&
      (landmark.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY &&
      landmark.x >= 0 &&
      landmark.x <= 1 &&
      landmark.y >= 0 &&
      landmark.y <= 1
    );
  }).length;

  landmarkCompleteness.value =
    (visibleCount / REQUIRED_LANDMARK_INDICES.length) * 100;
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

function resetJumpingJackAnalysis(reason: string) {
  feetState.value = "unknown";
  armsState.value = "unknown";
  instantJumpingJackState.value = "unknown";
  jumpingJackState.value = "unknown";
  actionStateReason.value = reason;
  pendingJumpingJackState = "unknown";
  pendingJumpingJackFrames = 0;
  actionDebounceProgress.value = `0/${ACTION_DEBOUNCE_FRAMES}`;
}

function isVisibleLandmark(landmark: LandmarkPoint | undefined) {
  return Boolean(
    landmark &&
      (landmark.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY &&
      landmark.x >= 0 &&
      landmark.x <= 1 &&
      landmark.y >= 0 &&
      landmark.y <= 1,
  );
}

function getDistanceX(left: LandmarkPoint, right: LandmarkPoint) {
  return Math.abs(left.x - right.x);
}

function getAverageY(left: LandmarkPoint, right: LandmarkPoint) {
  return (left.y + right.y) / 2;
}

function getRequiredPoseLandmarks(landmarks: LandmarkPoint[]) {
  const required = {
    leftShoulder: landmarks[LANDMARK_INDEX.leftShoulder],
    rightShoulder: landmarks[LANDMARK_INDEX.rightShoulder],
    leftWrist: landmarks[LANDMARK_INDEX.leftWrist],
    rightWrist: landmarks[LANDMARK_INDEX.rightWrist],
    leftHip: landmarks[LANDMARK_INDEX.leftHip],
    rightHip: landmarks[LANDMARK_INDEX.rightHip],
    leftAnkle: landmarks[LANDMARK_INDEX.leftAnkle],
    rightAnkle: landmarks[LANDMARK_INDEX.rightAnkle],
  };

  const missingKey = Object.entries(required).find(
    ([, landmark]) => !isVisibleLandmark(landmark),
  )?.[0];

  return missingKey ? { missingKey, landmarks: null } : { missingKey: "", landmarks: required };
}

function classifyFeet(ankleDistance: number, shoulderWidth: number) {
  if (ankleDistance >= shoulderWidth * FOOT_OPEN_SHOULDER_RATIO) {
    return "open";
  }

  if (ankleDistance <= shoulderWidth * FOOT_CLOSED_SHOULDER_RATIO) {
    return "closed";
  }

  return "unknown";
}

function classifyArms(
  wristDistance: number,
  shoulderWidth: number,
  averageWristY: number,
  averageShoulderY: number,
  torsoHeight: number,
) {
  const wristHeightFromShoulder = averageWristY - averageShoulderY;

  if (
    wristDistance >= shoulderWidth * WRIST_OPEN_SHOULDER_RATIO &&
    wristHeightFromShoulder <= torsoHeight * ARM_OPEN_MAX_TORSO_RATIO
  ) {
    return "open";
  }

  if (
    wristDistance <= shoulderWidth * WRIST_CLOSED_SHOULDER_RATIO ||
    wristHeightFromShoulder >= torsoHeight * ARM_CLOSED_MIN_TORSO_RATIO
  ) {
    return "closed";
  }

  return "unknown";
}

function updateDebouncedJumpingJackState(nextState: JumpingJackState) {
  instantJumpingJackState.value = nextState;

  if (nextState === "unknown") {
    jumpingJackState.value = "unknown";
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
    actionDebounceProgress.value = `0/${ACTION_DEBOUNCE_FRAMES}`;
    return;
  }

  if (nextState === jumpingJackState.value) {
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
    actionDebounceProgress.value = `${ACTION_DEBOUNCE_FRAMES}/${ACTION_DEBOUNCE_FRAMES}`;
    return;
  }

  if (nextState !== pendingJumpingJackState) {
    pendingJumpingJackState = nextState;
    pendingJumpingJackFrames = 1;
  } else {
    pendingJumpingJackFrames += 1;
  }

  actionDebounceProgress.value = `${Math.min(
    pendingJumpingJackFrames,
    ACTION_DEBOUNCE_FRAMES,
  )}/${ACTION_DEBOUNCE_FRAMES}`;

  if (pendingJumpingJackFrames >= ACTION_DEBOUNCE_FRAMES) {
    jumpingJackState.value = nextState;
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
  }
}

function updateJumpingJackAnalysis(landmarks: LandmarkPoint[] | null) {
  if (!landmarks?.length) {
    resetJumpingJackAnalysis("未检测到完整人体");
    return;
  }

  const { missingKey, landmarks: pose } = getRequiredPoseLandmarks(landmarks);

  if (!pose) {
    resetJumpingJackAnalysis(`关键点不足: ${missingKey}`);
    return;
  }

  const shoulderWidth = getDistanceX(pose.leftShoulder, pose.rightShoulder);

  if (shoulderWidth < MIN_SHOULDER_WIDTH) {
    resetJumpingJackAnalysis("肩宽过小，用户可能离摄像头太远或未正对镜头");
    return;
  }

  const ankleDistance = getDistanceX(pose.leftAnkle, pose.rightAnkle);
  const wristDistance = getDistanceX(pose.leftWrist, pose.rightWrist);
  const averageShoulderY = getAverageY(pose.leftShoulder, pose.rightShoulder);
  const averageHipY = getAverageY(pose.leftHip, pose.rightHip);
  const averageWristY = getAverageY(pose.leftWrist, pose.rightWrist);
  const torsoHeight = Math.max(Math.abs(averageHipY - averageShoulderY), 0.05);
  const nextFeetState = classifyFeet(ankleDistance, shoulderWidth);
  const nextArmsState = classifyArms(
    wristDistance,
    shoulderWidth,
    averageWristY,
    averageShoulderY,
    torsoHeight,
  );
  const nextActionState =
    nextFeetState === "open" && nextArmsState === "open"
      ? "open"
      : nextFeetState === "closed" && nextArmsState === "closed"
        ? "closed"
        : "unknown";

  feetState.value = nextFeetState;
  armsState.value = nextArmsState;
  updateDebouncedJumpingJackState(nextActionState);

  actionStateReason.value =
    `脚踝/肩宽 ${ankleDistance / shoulderWidth >= 10 ? "9.9+" : (ankleDistance / shoulderWidth).toFixed(2)}, ` +
    `手腕/肩宽 ${wristDistance / shoulderWidth >= 10 ? "9.9+" : (wristDistance / shoulderWidth).toFixed(2)}, ` +
    `手腕高度 ${(averageWristY - averageShoulderY) / torsoHeight >= 10 ? "9.9+" : ((averageWristY - averageShoulderY) / torsoHeight).toFixed(2)}`;
}

function toCanvasPoint(
  landmark: LandmarkPoint,
  canvasWidth: number,
  canvasHeight: number,
) {
  return {
    x: landmark.x * canvasWidth,
    y: landmark.y * canvasHeight,
  };
}

function drawPoseOverlay(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[] | null,
) {
  if (!landmarks?.length) {
    return;
  }

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "rgba(45, 212, 191, 0.9)";
  context.lineWidth = Math.max(3, canvas.width / 260);

  POSE_CONNECTIONS.forEach(([fromIndex, toIndex]) => {
    const from = landmarks[fromIndex];
    const to = landmarks[toIndex];

    if (!isVisibleLandmark(from) || !isVisibleLandmark(to)) {
      return;
    }

    const start = toCanvasPoint(from, canvas.width, canvas.height);
    const end = toCanvasPoint(to, canvas.width, canvas.height);

    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  });

  landmarks.forEach((landmark, index) => {
    if (!isVisibleLandmark(landmark)) {
      return;
    }

    const point = toCanvasPoint(landmark, canvas.width, canvas.height);
    const isRequired = REQUIRED_LANDMARK_INDICES.includes(index);

    context.beginPath();
    context.fillStyle = isRequired ? "#facc15" : "#ffffff";
    context.strokeStyle = "rgba(15, 23, 42, 0.72)";
    context.lineWidth = 2;
    context.arc(point.x, point.y, isRequired ? 6 : 4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });

  context.restore();
}

async function runPoseInference(now: number) {
  const video = videoRef.value;

  if (
    !poseLandmarker ||
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
    const result = poseLandmarker.detectForVideo(video, now);
    const landmarks = result.landmarks[0] as LandmarkPoint[] | undefined;

    updateInferenceFps(now);

    if (!landmarks?.length) {
      poseStatus.value = "missing";
      landmarkCompleteness.value = 0;
      rawLandmarkState.value = "未检测到人体";
      smoothedLandmarkState.value = "无平滑数据";
      landmarkHistory.length = 0;
      smoothedLandmarks = null;
      resetJumpingJackAnalysis("未检测到完整人体");
      return;
    }

    poseStatus.value = "detected";
    rawLandmarkState.value = `${landmarks.length} 个关键点`;
    updateLandmarkCompleteness(landmarks);
    updateSmoothedLandmarks(landmarks);
    smoothedLandmarkState.value = smoothedLandmarks
      ? `${smoothedLandmarks.length} 个平滑关键点`
      : "无平滑数据";
    updateJumpingJackAnalysis(smoothedLandmarks);
  } catch (error) {
    poseStatus.value = "missing";
    rawLandmarkState.value =
      error instanceof Error ? error.message : "姿态推理失败";
    resetJumpingJackAnalysis("姿态推理失败");
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

  const videoWidth = video.videoWidth || 1280;
  const videoHeight = video.videoHeight || 720;

  if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
    canvas.width = videoWidth;
    canvas.height = videoHeight;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  void runPoseInference(performance.now());
  drawPoseOverlay(context, canvas, smoothedLandmarks);

  animationFrameId = requestAnimationFrame(renderFrame);
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraStatus.value = "error";
    cameraMessage.value = "当前浏览器不支持摄像头访问。";
    return;
  }

  stopCamera();
  cameraStatus.value = "loading";
  cameraMessage.value = "正在请求摄像头权限";

  try {
    stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);

    if (!videoRef.value) {
      return;
    }

    videoRef.value.srcObject = stream;
    await videoRef.value.play();

    cameraStatus.value = "ready";
    cameraMessage.value = "摄像头已就绪，Phase 3 将接入姿态识别。";
    stopRenderLoop();
    animationFrameId = requestAnimationFrame(renderFrame);
  } catch (error) {
    stopCamera();
    cameraStatus.value = "error";
    cameraMessage.value = getCameraErrorMessage(error);
  }
}

onMounted(() => {
  void startCamera();
  void loadPoseLandmarker();
});

onBeforeUnmount(() => {
  stopCamera();
  disposePoseLandmarker();
});
</script>

<template>
  <main class="app-shell">
    <section class="stage-panel" aria-label="摄像头识别区域">
      <div class="stage-toolbar">
        <div>
          <p class="eyebrow">CalorieCal MVP</p>
          <h1>开合跳识别</h1>
        </div>
        <span class="status-pill" :class="`status-${cameraStatus}`">
          {{ cameraStatus }}
        </span>
      </div>

      <div class="camera-stage">
        <div class="camera-frame">
          <video ref="videoRef" aria-label="摄像头原始画面" playsinline muted />
          <canvas ref="canvasRef" aria-label="摄像头画布渲染层" />
          <div v-if="cameraStatus !== 'ready'" class="camera-overlay">
            <span>Camera</span>
            <p>{{ cameraMessage }}</p>
          </div>
          <div v-else class="camera-message" aria-live="polite">
            {{ cameraMessage }}
          </div>
          <div
            v-if="cameraStatus === 'ready' && poseStatus === 'missing'"
            class="pose-hint"
          >
            未检测到完整人体，请全身入镜并正对摄像头。
          </div>
          <div
            v-if="cameraStatus === 'ready' && poseStatus === 'detected'"
            class="jump-state-badge"
          >
            <span>动作</span>
            <strong>{{ jumpingJackState }}</strong>
            <small>脚 {{ feetState }} / 手 {{ armsState }}</small>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button type="button" @click="startCamera">开始</button>
        <button type="button" class="secondary">暂停</button>
        <button type="button" class="secondary" @click="stopCamera">重置</button>
      </div>
    </section>

    <aside class="side-panel" aria-label="实时指标">
      <label class="weight-field">
        <span>体重 kg</span>
        <input type="number" min="20" max="200" value="60" />
      </label>

      <div class="metrics-grid">
        <section
          v-for="metric in metrics"
          :key="metric.label"
          class="metric-card"
        >
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small v-if="metric.detail">{{ metric.detail }}</small>
        </section>
      </div>

      <details class="debug-panel">
        <summary>
          <span>姿态模型</span>
          <strong :class="`debug-status status-${modelStatus}`">
            {{ modelStatus }}
          </strong>
        </summary>
        <div class="debug-body">
          <p>{{ modelMessage }}</p>
          <dl>
            <template v-for="item in modelDetails" :key="item.label">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </template>
          </dl>
        </div>
      </details>

      <details class="debug-panel">
        <summary>
          <span>姿态调试</span>
          <strong :class="`debug-status status-${poseStatus}`">
            {{ poseStatus }}
          </strong>
        </summary>
        <div class="debug-body">
          <p>显示层镜像，算法层保持 MediaPipe 原始坐标。</p>
          <dl>
            <template v-for="item in poseDebugDetails" :key="item.label">
              <dt>{{ item.label }}</dt>
              <dd>{{ item.value }}</dd>
            </template>
          </dl>
        </div>
      </details>
    </aside>
  </main>
</template>
