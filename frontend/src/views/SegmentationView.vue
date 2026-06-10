<script setup lang="ts">
import {
  FilesetResolver,
  PoseLandmarker,
  type MPMask,
} from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type CameraStatus = "idle" | "loading" | "ready" | "error";
type ModelStatus = "idle" | "loading" | "ready" | "error";
type PoseStatus = "idle" | "detected" | "missing";

type LandmarkPoint = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
};

type TrackedPoint = {
  label: string;
  index: number;
};

type DepthLookupPoint = {
  id: string;
  label: string;
  normX: number;
  normY: number;
  radius?: number;
  searchRadius?: number;
};

type DepthLookupResult = {
  id?: string;
  x: number;
  y: number;
  depthMm: number;
  medianDepthMm: number;
  foregroundDepthMm?: number;
  validRatio?: number;
  foregroundCandidate?: boolean;
  valid: boolean;
};

type DepthLookupResponse = {
  ok: boolean;
  width?: number;
  height?: number;
  error?: string;
  points?: DepthLookupResult[];
};

type DepthSampleMetric = {
  id: string;
  label: string;
  depthText: string;
  maskText: string;
  detail: string;
  valid: boolean;
  maskHit: boolean;
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
const TARGET_INFERENCE_FPS = 15;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const MASK_THRESHOLD = 0.5;
const MIN_LANDMARK_VISIBILITY = 0.5;
const DEPTH_SERVER_URL = "http://127.0.0.1:8765/depth";
const DEPTH_REQUEST_INTERVAL_MS = 250;
const TRACKED_POINTS: TrackedPoint[] = [
  { label: "鼻尖", index: 0 },
  { label: "左肩", index: 11 },
  { label: "右肩", index: 12 },
  { label: "左肘", index: 13 },
  { label: "右肘", index: 14 },
  { label: "左腕", index: 15 },
  { label: "右腕", index: 16 },
  { label: "左髋", index: 23 },
  { label: "右髋", index: 24 },
  { label: "左膝", index: 25 },
  { label: "右膝", index: 26 },
  { label: "左踝", index: 27 },
  { label: "右踝", index: 28 },
];

const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const cameraStatus = ref<CameraStatus>("idle");
const cameraMessage = ref("等待摄像头授权");
const modelStatus = ref<ModelStatus>("idle");
const modelMessage = ref("等待分割模型加载");
const poseStatus = ref<PoseStatus>("idle");
const inferenceFps = ref(0);
const maskSize = ref("未知");
const maskCoverage = ref(0);
const maskHitRate = ref(0);
const visiblePointCount = ref(0);
const hitPointCount = ref(0);
const sampleSummary = ref("等待人体入镜");
const depthStatus = ref<"idle" | "ready" | "error">("idle");
const depthMessage = ref("等待 RGB-D depth server");
const depthSampleMetrics = ref<DepthSampleMetric[]>([]);
const depthMaskHitRate = ref(0);

let stream: MediaStream | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let animationFrameId: number | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let latestLandmarks: LandmarkPoint[] = [];
let latestMaskData: Float32Array | Uint8Array | null = null;
let latestMaskWidth = 0;
let latestMaskHeight = 0;
let lastDepthRequestAt = 0;
let isDepthRequesting = false;

const cameraButtonLabel = computed(() =>
  cameraStatus.value === "ready" ? "停止摄像头" : "启动摄像头",
);

const metricCards = computed(() => [
  {
    label: "人体 Mask",
    value: maskSize.value,
    detail: `覆盖率 ${(maskCoverage.value * 100).toFixed(1)}%`,
  },
  {
    label: "关键点命中率",
    value: `${Math.round(maskHitRate.value * 100)}%`,
    detail: `${hitPointCount.value}/${visiblePointCount.value} 个可见点落在人像区域`,
  },
  {
    label: "推理 FPS",
    value: inferenceFps.value.toFixed(1),
    detail: "含 segmentation mask 输出",
  },
  {
    label: "姿态状态",
    value: poseStatus.value === "detected" ? "检测到人体" : "等待人体",
    detail: sampleSummary.value,
  },
  {
    label: "Depth 命中率",
    value: `${Math.round(depthMaskHitRate.value * 100)}%`,
    detail: depthMessage.value,
  },
]);

function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function disposePoseLandmarker() {
  poseLandmarker?.close();
  poseLandmarker = null;
}

function resetSegmentationState(message = "等待人体入镜") {
  poseStatus.value = "idle";
  maskSize.value = "未知";
  maskCoverage.value = 0;
  maskHitRate.value = 0;
  visiblePointCount.value = 0;
  hitPointCount.value = 0;
  sampleSummary.value = message;
  depthStatus.value = "idle";
  depthMessage.value = "等待 RGB-D depth server";
  depthSampleMetrics.value = [];
  depthMaskHitRate.value = 0;
  latestLandmarks = [];
  latestMaskData = null;
  latestMaskWidth = 0;
  latestMaskHeight = 0;
  lastDepthRequestAt = 0;
  isDepthRequesting = false;
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

  cameraStatus.value = "idle";
  cameraMessage.value = "摄像头已停止";
  inferenceFps.value = 0;
  resetSegmentationState("摄像头已停止");
}

function getCameraErrorMessage(error: unknown) {
  if (!(error instanceof DOMException)) {
    return "摄像头暂时不可用";
  }

  if (error.name === "NotAllowedError") {
    return "摄像头未授权";
  }

  if (error.name === "NotFoundError") {
    return "未找到摄像头";
  }

  if (error.name === "NotReadableError") {
    return "摄像头可能被其他应用占用";
  }

  return "摄像头暂时不可用";
}

async function loadPoseLandmarker() {
  if (poseLandmarker || modelStatus.value === "loading") {
    return;
  }

  modelStatus.value = "loading";
  modelMessage.value = "正在加载 Pose Landmarker + segmentation mask";

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: POSE_LANDMARKER_MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      outputSegmentationMasks: true,
    });

    modelStatus.value = "ready";
    modelMessage.value = "模型已就绪，正在输出人体 mask。";
  } catch (error) {
    disposePoseLandmarker();
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error
        ? `模型加载失败：${error.message}`
        : "模型加载失败，请检查网络和模型资源。";
  }
}

async function startCamera() {
  if (cameraStatus.value === "ready") {
    stopCamera();
    return;
  }

  cameraStatus.value = "loading";
  cameraMessage.value = "正在请求摄像头";

  try {
    stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);

    if (!videoRef.value) {
      throw new Error("video element is not ready");
    }

    videoRef.value.srcObject = stream;
    await videoRef.value.play();

    cameraStatus.value = "ready";
    cameraMessage.value = "摄像头已启动";
    startRenderLoop();
  } catch (error) {
    stopCamera();
    cameraStatus.value = "error";
    cameraMessage.value = getCameraErrorMessage(error);
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

function getVisibleTrackedPoints(landmarks: LandmarkPoint[]) {
  return TRACKED_POINTS.filter((point) => {
    const landmark = landmarks[point.index];
    return landmark && (landmark.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY;
  });
}

function readMaskValue(maskData: Float32Array | Uint8Array, x: number, y: number) {
  const clampedX = Math.min(Math.max(Math.round(x), 0), latestMaskWidth - 1);
  const clampedY = Math.min(Math.max(Math.round(y), 0), latestMaskHeight - 1);
  const rawValue = maskData[clampedY * latestMaskWidth + clampedX] ?? 0;

  return maskData instanceof Uint8Array ? rawValue / 255 : rawValue;
}

function getMaskValueAtNorm(normX: number, normY: number) {
  if (!latestMaskData || !latestMaskWidth || !latestMaskHeight) {
    return 0;
  }

  return readMaskValue(latestMaskData, normX * latestMaskWidth, normY * latestMaskHeight);
}

function getLandmark(index: number) {
  const landmark = latestLandmarks[index];

  if (!landmark || (landmark.visibility ?? 1) < MIN_LANDMARK_VISIBILITY) {
    return null;
  }

  return landmark;
}

function getMidPoint(
  first: LandmarkPoint | null,
  second: LandmarkPoint | null,
): LandmarkPoint | null {
  if (!first || !second) {
    return null;
  }

  return {
    x: (first.x + second.x) / 2,
    y: (first.y + second.y) / 2,
    z:
      first.z !== undefined && second.z !== undefined
        ? (first.z + second.z) / 2
        : undefined,
    visibility: Math.min(first.visibility ?? 1, second.visibility ?? 1),
  };
}

function buildDepthLookupPoints(): DepthLookupPoint[] {
  const leftWrist = getLandmark(15);
  const rightWrist = getLandmark(16);
  const leftShoulder = getLandmark(11);
  const rightShoulder = getLandmark(12);
  const leftHip = getLandmark(23);
  const rightHip = getLandmark(24);
  const leftAnkle = getLandmark(27);
  const rightAnkle = getLandmark(28);
  const shoulderCenter = getMidPoint(leftShoulder, rightShoulder);
  const hipCenter = getMidPoint(leftHip, rightHip);

  const candidates: Array<[string, string, LandmarkPoint | null, number, number]> = [
    ["leftWrist", "左腕", leftWrist, 5, 16],
    ["rightWrist", "右腕", rightWrist, 5, 16],
    ["shoulderCenter", "肩中心", shoulderCenter, 7, 20],
    ["hipCenter", "髋中心", hipCenter, 7, 20],
    ["leftAnkle", "左踝", leftAnkle, 6, 18],
    ["rightAnkle", "右踝", rightAnkle, 6, 18],
  ];

  return candidates
    .filter((candidate): candidate is [string, string, LandmarkPoint, number, number] =>
      Boolean(candidate[2]),
    )
    .map(([id, label, landmark, radius, searchRadius]) => ({
      id,
      label,
      normX: landmark.x,
      normY: landmark.y,
      radius,
      searchRadius,
    }));
}

function formatDepth(result: DepthLookupResult) {
  const foreground = result.foregroundDepthMm ?? 0;
  const median = result.medianDepthMm ?? 0;
  const preferred = foreground > 0 ? foreground : median;

  return preferred > 0 ? `${preferred}mm` : "无效";
}

async function updateDepthSamples(now: number) {
  if (
    isDepthRequesting ||
    !latestLandmarks.length ||
    now - lastDepthRequestAt < DEPTH_REQUEST_INTERVAL_MS
  ) {
    return;
  }

  const points = buildDepthLookupPoints();
  if (!points.length) {
    depthStatus.value = "idle";
    depthMessage.value = "关键点不足，暂不采样 depth";
    depthSampleMetrics.value = [];
    depthMaskHitRate.value = 0;
    return;
  }

  isDepthRequesting = true;
  lastDepthRequestAt = now;

  try {
    const response = await fetch(DEPTH_SERVER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points }),
    });
    const payload = (await response.json()) as DepthLookupResponse;

    if (!response.ok || !payload.ok) {
      throw new Error(payload.error ?? `HTTP ${response.status}`);
    }

    const resultMap = new Map((payload.points ?? []).map((point) => [point.id, point]));
    const metrics = points.map((point) => {
      const result = resultMap.get(point.id);
      const maskValue = getMaskValueAtNorm(point.normX, point.normY);
      const maskHit = maskValue >= MASK_THRESHOLD;

      return {
        id: point.id,
        label: point.label,
        depthText: result ? formatDepth(result) : "无返回",
        maskText: `${Math.round(maskValue * 100)}%`,
        detail: result
          ? `median ${result.medianDepthMm || 0} / fg ${
              result.foregroundDepthMm || 0
            } / valid ${result.validRatio ?? 0}`
          : "depth server 未返回该点",
        valid: Boolean(result?.valid),
        maskHit,
      };
    });
    const validMetrics = metrics.filter((metric) => metric.valid);
    const hitMetrics = validMetrics.filter((metric) => metric.maskHit);

    depthSampleMetrics.value = metrics;
    depthMaskHitRate.value = validMetrics.length ? hitMetrics.length / validMetrics.length : 0;
    depthStatus.value = "ready";
    depthMessage.value = `${hitMetrics.length}/${validMetrics.length} 个有效 depth 点落在人像 mask 内`;
  } catch (error) {
    depthStatus.value = "error";
    depthMessage.value =
      error instanceof Error
        ? `RGB-D 未连接或请求失败：${error.message}`
        : "RGB-D 未连接或请求失败";
    depthSampleMetrics.value = [];
    depthMaskHitRate.value = 0;
  } finally {
    isDepthRequesting = false;
  }
}

function updateMaskAnalysis(landmarks: LandmarkPoint[], mask: MPMask | undefined) {
  if (!mask) {
    latestMaskData = null;
    latestMaskWidth = 0;
    latestMaskHeight = 0;
    maskSize.value = "无 mask";
    maskCoverage.value = 0;
    maskHitRate.value = 0;
    visiblePointCount.value = 0;
    hitPointCount.value = 0;
    sampleSummary.value = "本帧未返回 segmentation mask";
    return;
  }

  latestMaskWidth = mask.width;
  latestMaskHeight = mask.height;
  maskSize.value = `${mask.width}x${mask.height}`;
  latestMaskData = mask.hasFloat32Array() ? mask.getAsFloat32Array() : mask.getAsUint8Array();

  let foregroundPixels = 0;
  for (let index = 0; index < latestMaskData.length; index += 1) {
    const value =
      latestMaskData instanceof Uint8Array
        ? latestMaskData[index] / 255
        : latestMaskData[index];
    if (value >= MASK_THRESHOLD) {
      foregroundPixels += 1;
    }
  }
  maskCoverage.value = latestMaskData.length ? foregroundPixels / latestMaskData.length : 0;

  const visiblePoints = getVisibleTrackedPoints(landmarks);
  const hitPoints = visiblePoints.filter((point) => {
    const landmark = landmarks[point.index];
    const value = readMaskValue(
      latestMaskData as Float32Array | Uint8Array,
      landmark.x * latestMaskWidth,
      landmark.y * latestMaskHeight,
    );

    return value >= MASK_THRESHOLD;
  });

  visiblePointCount.value = visiblePoints.length;
  hitPointCount.value = hitPoints.length;
  maskHitRate.value = visiblePoints.length ? hitPoints.length / visiblePoints.length : 0;
  sampleSummary.value = visiblePoints.length
    ? `${hitPoints.length}/${visiblePoints.length} 个关键点命中人体 mask`
    : "关键点可见率不足";
}

function drawVideoFrame(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  const video = videoRef.value;
  if (!video || !video.videoWidth || !video.videoHeight) {
    return;
  }

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  context.save();
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.scale(-1, 1);
  context.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  context.restore();
}

function drawMaskOverlay(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  if (!latestMaskData || !latestMaskWidth || !latestMaskHeight) {
    return;
  }

  const imageData = new ImageData(latestMaskWidth, latestMaskHeight);

  for (let index = 0; index < latestMaskData.length; index += 1) {
    const value =
      latestMaskData instanceof Uint8Array
        ? latestMaskData[index] / 255
        : latestMaskData[index];
    const alpha = value >= MASK_THRESHOLD ? Math.round(110 * value) : 0;
    const offset = index * 4;

    imageData.data[offset] = 56;
    imageData.data[offset + 1] = 189;
    imageData.data[offset + 2] = 248;
    imageData.data[offset + 3] = alpha;
  }

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = latestMaskWidth;
  maskCanvas.height = latestMaskHeight;
  const maskContext = maskCanvas.getContext("2d");
  if (!maskContext) {
    return;
  }

  maskContext.putImageData(imageData, 0, 0);

  context.save();
  context.scale(-1, 1);
  context.drawImage(maskCanvas, -canvas.width, 0, canvas.width, canvas.height);
  context.restore();
}

function drawPosePoints(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  if (!latestLandmarks.length) {
    return;
  }

  context.save();
  context.fillStyle = "#f8fafc";
  context.strokeStyle = "rgba(15, 23, 42, 0.75)";
  context.lineWidth = 2;

  for (const point of TRACKED_POINTS) {
    const landmark = latestLandmarks[point.index];
    if (!landmark || (landmark.visibility ?? 1) < MIN_LANDMARK_VISIBILITY) {
      continue;
    }

    const x = (1 - landmark.x) * canvas.width;
    const y = landmark.y * canvas.height;
    context.beginPath();
    context.arc(x, y, 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }

  context.restore();
}

function drawDepthSamplePoints(context: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
  if (!depthSampleMetrics.value.length || !latestLandmarks.length) {
    return;
  }

  const samplePoints = buildDepthLookupPoints();
  const metricMap = new Map(depthSampleMetrics.value.map((metric) => [metric.id, metric]));

  context.save();
  context.font = "13px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
  context.lineWidth = 2;

  for (const point of samplePoints) {
    const metric = metricMap.get(point.id);
    const x = (1 - point.normX) * canvas.width;
    const y = point.normY * canvas.height;

    context.fillStyle = metric?.maskHit ? "#22c55e" : "#ef4444";
    context.strokeStyle = "rgba(15, 23, 42, 0.8)";
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fill();
    context.stroke();

    context.fillStyle = "rgba(15, 23, 42, 0.82)";
    context.fillRect(x + 10, y - 11, 58, 22);
    context.fillStyle = "#ffffff";
    context.fillText(point.label, x + 15, y + 4);
  }

  context.restore();
}

async function runPoseInference(now: number) {
  const video = videoRef.value;

  if (
    !video ||
    !poseLandmarker ||
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
    latestLandmarks = (result.landmarks[0] as LandmarkPoint[] | undefined) ?? [];
    updateInferenceFps(now);

    if (!latestLandmarks.length) {
      poseStatus.value = "missing";
      latestMaskData = null;
      sampleSummary.value = "未检测到人体";
      return;
    }

    poseStatus.value = "detected";
    updateMaskAnalysis(latestLandmarks, result.segmentationMasks?.[0]);
    void updateDepthSamples(now);
  } catch (error) {
    poseStatus.value = "missing";
    sampleSummary.value =
      error instanceof Error ? `推理失败：${error.message}` : "推理失败";
  } finally {
    isInferencing = false;
  }
}

function renderFrame() {
  const canvas = canvasRef.value;
  const context = canvas?.getContext("2d");

  if (!canvas || !context) {
    return;
  }

  const now = performance.now();
  void runPoseInference(now);
  drawVideoFrame(context, canvas);
  drawMaskOverlay(context, canvas);
  drawPosePoints(context, canvas);
  drawDepthSamplePoints(context, canvas);

  animationFrameId = requestAnimationFrame(renderFrame);
}

function startRenderLoop() {
  stopRenderLoop();
  animationFrameId = requestAnimationFrame(renderFrame);
}

onMounted(() => {
  void loadPoseLandmarker();
});

onBeforeUnmount(() => {
  stopCamera();
  disposePoseLandmarker();
});
</script>

<template>
  <main class="segmentation-page">
    <section class="segmentation-stage">
      <div class="stage-header">
        <div>
          <p class="eyebrow">Task8 Segmentation</p>
          <h2>人体 Mask 测试</h2>
        </div>
        <button type="button" class="primary-button" @click="startCamera">
          {{ cameraButtonLabel }}
        </button>
      </div>

      <div class="stage-canvas-wrap">
        <video ref="videoRef" playsinline muted aria-hidden="true" />
        <canvas ref="canvasRef" aria-label="人体分割测试画面"></canvas>
        <div
          v-if="cameraStatus !== 'ready'"
          class="stage-placeholder"
          aria-live="polite"
        >
          <strong>{{ cameraMessage }}</strong>
          <span>{{ modelMessage }}</span>
        </div>
      </div>
    </section>

    <aside class="segmentation-panel" aria-label="人体分割测试指标">
      <section class="status-card">
        <span>模型状态</span>
        <strong>{{ modelStatus === "ready" ? "已就绪" : modelStatus }}</strong>
        <small>{{ modelMessage }}</small>
      </section>

      <section class="status-card">
        <span>摄像头状态</span>
        <strong>{{ cameraStatus === "ready" ? "运行中" : cameraStatus }}</strong>
        <small>{{ cameraMessage }}</small>
      </section>

      <section class="status-card">
        <span>RGB-D 区域采样</span>
        <strong>{{ depthStatus === "ready" ? "已连接" : depthStatus }}</strong>
        <small>{{ depthMessage }}</small>
      </section>

      <div class="metric-grid">
        <section v-for="metric in metricCards" :key="metric.label" class="metric-card">
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small>{{ metric.detail }}</small>
        </section>
      </div>

      <section class="depth-samples" aria-label="Depth 采样点">
        <div class="depth-samples-header">
          <span>采样点</span>
          <small>绿色为落在人像 mask 内</small>
        </div>
        <div v-if="depthSampleMetrics.length" class="depth-sample-list">
          <article
            v-for="sample in depthSampleMetrics"
            :key="sample.id"
            class="depth-sample"
            :class="{ missed: !sample.maskHit }"
          >
            <div>
              <strong>{{ sample.label }}</strong>
              <span>{{ sample.depthText }}</span>
            </div>
            <small>mask {{ sample.maskText }} / {{ sample.detail }}</small>
          </article>
        </div>
        <p v-else class="empty-text">启动 depth server 后显示采样点。</p>
      </section>
    </aside>
  </main>
</template>

<style scoped>
.segmentation-page {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 20px;
  width: 100%;
  min-height: 100vh;
  padding: 24px;
  background: #f8fafc;
  color: #102033;
}

.segmentation-stage,
.segmentation-panel {
  min-width: 0;
}

.stage-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.eyebrow {
  margin: 0 0 4px;
  color: #0f766e;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h2 {
  margin: 0;
  font-size: 1.55rem;
}

.primary-button {
  border: 0;
  border-radius: 8px;
  padding: 10px 16px;
  background: #0f766e;
  color: #ffffff;
  font-weight: 700;
  cursor: pointer;
}

.stage-canvas-wrap {
  position: relative;
  overflow: hidden;
  width: 100%;
  aspect-ratio: 16 / 9;
  border: 1px solid #d8e1ea;
  border-radius: 8px;
  background: #111827;
}

video {
  display: none;
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.stage-placeholder {
  position: absolute;
  inset: 0;
  display: grid;
  place-content: center;
  gap: 8px;
  padding: 24px;
  color: #f8fafc;
  text-align: center;
  background: rgba(15, 23, 42, 0.72);
}

.stage-placeholder strong {
  font-size: 1.35rem;
}

.stage-placeholder span {
  color: #cbd5e1;
}

.segmentation-panel {
  display: grid;
  align-content: start;
  gap: 14px;
}

.status-card,
.metric-card {
  display: grid;
  gap: 6px;
  min-width: 0;
  border: 1px solid #d8e1ea;
  border-radius: 8px;
  padding: 14px;
  background: #ffffff;
}

.status-card span,
.metric-card span {
  color: #64748b;
  font-size: 0.82rem;
}

.status-card strong,
.metric-card strong {
  color: #0f172a;
  font-size: 1.18rem;
  overflow-wrap: anywhere;
}

.status-card small,
.metric-card small {
  color: #64748b;
  line-height: 1.45;
}

.metric-grid {
  display: grid;
  gap: 14px;
}

.depth-samples {
  display: grid;
  gap: 10px;
  border: 1px solid #d8e1ea;
  border-radius: 8px;
  padding: 14px;
  background: #ffffff;
}

.depth-samples-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.depth-samples-header span {
  color: #0f172a;
  font-weight: 700;
}

.depth-samples-header small,
.empty-text {
  color: #64748b;
}

.depth-sample-list {
  display: grid;
  gap: 8px;
}

.depth-sample {
  display: grid;
  gap: 4px;
  border-left: 4px solid #22c55e;
  border-radius: 6px;
  padding: 8px 10px;
  background: #f8fafc;
}

.depth-sample.missed {
  border-left-color: #ef4444;
}

.depth-sample div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.depth-sample strong {
  color: #0f172a;
}

.depth-sample span {
  color: #0f766e;
  font-weight: 700;
}

.depth-sample small {
  color: #64748b;
  line-height: 1.35;
}

@media (max-width: 980px) {
  .segmentation-page {
    grid-template-columns: 1fr;
  }

  .segmentation-panel {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .metric-grid {
    grid-column: 1 / -1;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 620px) {
  .segmentation-page {
    padding: 16px;
  }

  .stage-header {
    align-items: stretch;
    flex-direction: column;
  }

  .segmentation-panel,
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
</style>
