<script setup lang="ts">
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

// ---------------------------------------------------------------------------
// Backend action server types
// ---------------------------------------------------------------------------

type BackendActionResult = {
  name: string;
  count: number;
  stage: string;
  score: number | null;
};

type BackendWsStatus = "disconnected" | "connecting" | "connected" | "error";

// ---------------------------------------------------------------------------

type CameraStatus = "idle" | "loading" | "ready" | "error";
type ModelStatus = "idle" | "loading" | "ready" | "error";
type PoseStatus = "idle" | "detecting" | "detected" | "missing";
type JumpingJackState = "unknown" | "closed" | "open";
type JumpingJackRuntimeStatus = "idle" | "running" | "paused";
type JumpingJackCounterPhase = "waiting_closed" | "ready" | "opened";
type FullBodyIntensityLevel = "静止" | "低强度" | "中强度" | "高强度";
type ConfidenceLevel = "高" | "中" | "低";

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

type FullBodyTrackedKey =
  | "nose"
  | "leftEar"
  | "rightEar"
  | "leftShoulder"
  | "rightShoulder"
  | "leftElbow"
  | "rightElbow"
  | "leftWrist"
  | "rightWrist"
  | "leftHip"
  | "rightHip"
  | "leftKnee"
  | "rightKnee"
  | "leftAnkle"
  | "rightAnkle";

type FullBodyMotionSample = {
  time: number;
  points: Partial<Record<FullBodyTrackedKey, LandmarkPoint>>;
};

type FullBodyMotionFeatures = {
  movementDistance: number;
  upperBodyDistance: number;
  lowerBodyDistance: number;
  torsoDistance: number;
  movementSpeed: number;
  movementAmplitude: number;
  activeFullBodyMs: number;
};

type BasicActionCandidate = {
  label: string;
  score: number;
  reason: string;
};

type DepthLookupPoint = {
  id: string;
  normX: number;
  normY: number;
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
  points?: DepthLookupResult[];
  error?: string;
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
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const SMOOTHING_WINDOW_SIZE = 5;
const FULL_BODY_HISTORY_MS = 1200;
const FULL_BODY_ACTIVE_DISTANCE_THRESHOLD = 0.04;
const DEPTH_SERVER_URL = "http://127.0.0.1:8765/depth";
const ACTION_SERVER_WS_URL = `ws://${window.location.hostname}:8766/ws`;
const DEPTH_REQUEST_INTERVAL_MS = 250;

// Camel-case key → uppercase landmark name sent to backend
const BACKEND_LANDMARK_KEY_MAP: Record<string, string> = {
  leftShoulder:  "LEFT_SHOULDER",
  rightShoulder: "RIGHT_SHOULDER",
  leftElbow:     "LEFT_ELBOW",
  rightElbow:    "RIGHT_ELBOW",
  leftWrist:     "LEFT_WRIST",
  rightWrist:    "RIGHT_WRIST",
  leftHip:       "LEFT_HIP",
  rightHip:      "RIGHT_HIP",
  leftKnee:      "LEFT_KNEE",
  rightKnee:     "RIGHT_KNEE",
  leftAnkle:     "LEFT_ANKLE",
  rightAnkle:    "RIGHT_ANKLE",
};
const DEPTH_FRONT_BACK_THRESHOLD_MM = 150;
const DEPTH_ACTION_MAX_SCORE_OFFSET_MM = 800;
const DEPTH_MAX_REASONABLE_OFFSET_MM = 2000;
const REQUIRED_LANDMARK_INDICES = [11, 12, 15, 16, 23, 24, 27, 28];
const MIN_LANDMARK_VISIBILITY = 0.5;
const LANDMARK_INDEX = {
  nose: 0,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;
const FULL_BODY_TRACKED_KEYS: FullBodyTrackedKey[] = [
  "leftShoulder",
  "rightShoulder",
  "leftElbow",
  "rightElbow",
  "leftWrist",
  "rightWrist",
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle",
];
const HEAD_TRACKED_KEYS: FullBodyTrackedKey[] = ["nose", "leftEar", "rightEar"];
const SAMPLE_TRACKED_KEYS: FullBodyTrackedKey[] = [
  ...FULL_BODY_TRACKED_KEYS,
  ...HEAD_TRACKED_KEYS,
];
const UPPER_BODY_TRACKED_KEYS: FullBodyTrackedKey[] = [
  "leftShoulder",
  "rightShoulder",
  "leftElbow",
  "rightElbow",
  "leftWrist",
  "rightWrist",
];
const LOWER_BODY_TRACKED_KEYS: FullBodyTrackedKey[] = [
  "leftHip",
  "rightHip",
  "leftKnee",
  "rightKnee",
  "leftAnkle",
  "rightAnkle",
];
const TORSO_TRACKED_KEYS: FullBodyTrackedKey[] = [
  "leftShoulder",
  "rightShoulder",
  "leftHip",
  "rightHip",
];
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
const cameraStatus = ref<CameraStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelStatus = ref<ModelStatus>("idle");
const modelMessage = ref("等待姿态模型加载");
const handModelStatus = ref<ModelStatus>("idle");
const handModelMessage = ref("等待手部模型加载");
const poseStatus = ref<PoseStatus>("idle");
const inferenceFps = ref(0);
const handCount = ref(0);
const landmarkCompleteness = ref(0);
const rawLandmarkState = ref("等待推理");
const smoothedLandmarkState = ref("等待平滑数据");
const feetState = ref<JumpingJackState>("unknown");
const armsState = ref<JumpingJackState>("unknown");
const instantJumpingJackState = ref<JumpingJackState>("unknown");
const jumpingJackState = ref<JumpingJackState>("unknown");
const jumpingJackRuntimeStatus = ref<JumpingJackRuntimeStatus>("idle");
const jumpingJackCounterPhase =
  ref<JumpingJackCounterPhase>("waiting_closed");
const jumpingJackCount = ref(0);
const jumpingJackElapsedMs = ref(0);
const jumpingJackCounterReason = ref("点击开始后，从双脚合拢和双臂回落开始记录。");
const weightKg = ref(60);
const actionStateReason = ref("等待完整关键点");
const actionDebounceProgress = ref("0/3");
const fullBodyJointState = ref("等待全身关键点");
const fullBodyMotionFeatures = ref<FullBodyMotionFeatures>({
  movementDistance: 0,
  upperBodyDistance: 0,
  lowerBodyDistance: 0,
  torsoDistance: 0,
  movementSpeed: 0,
  movementAmplitude: 0,
  activeFullBodyMs: 0,
});
const fullBodyActivityScore = ref(0);
const fullBodyIntensityLevel = ref<FullBodyIntensityLevel>("静止");
const fullBodyBasicActions = ref<BasicActionCandidate[]>([]);
const depthServiceStatus = ref("idle");
const depthServiceMessage = ref("等待 depth server");
const depthFrameSize = ref("未知");
const leftWristDepthMm = ref<number | null>(null);
const rightWristDepthMm = ref<number | null>(null);
const bodyDepthBaselineMm = ref<number | null>(null);
const bodyDepthSource = ref("无");
const leftWristDepthOffsetMm = ref<number | null>(null);
const rightWristDepthOffsetMm = ref<number | null>(null);
const leftWristDepthRelation = ref("无");
const rightWristDepthRelation = ref("无");
const leftWristDepthSource = ref("无");
const rightWristDepthSource = ref("无");

// Backend action server state
const backendWsStatus = ref<BackendWsStatus>("disconnected");
const backendActions = ref<BackendActionResult[]>([]);
const backendLatencyMs = ref<number | null>(null);

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let smoothedLandmarks: LandmarkPoint[] | null = null;
let latestHands: LandmarkPoint[][] = [];
let pendingJumpingJackState: JumpingJackState = "unknown";
let pendingJumpingJackFrames = 0;
let lastJumpingJackRuntimeAt = 0;
let lastDepthRequestAt = 0;
let isDepthRequesting = false;
let backendWs: WebSocket | null = null;
let backendWsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let backendFrameIndex = 0;
const landmarkHistory: LandmarkPoint[][] = [];
const fullBodyMotionHistory: FullBodyMotionSample[] = [];

const topFullBodyBasicActionsText = computed(() => {
  if (!fullBodyBasicActions.value.length) {
    return "暂无明显基本动作";
  }

  return fullBodyBasicActions.value
    .slice(0, 3)
    .map((candidate) => `${candidate.label} ${candidate.score}`)
    .join(" / ");
});

const currentFullBodyMet = computed(() => {
  if (fullBodyIntensityLevel.value === "高强度") {
    return 6.0;
  }

  if (fullBodyIntensityLevel.value === "中强度") {
    return 4.0;
  }

  if (fullBodyIntensityLevel.value === "低强度") {
    return 2.2;
  }

  return 1.0;
});

const fullBodyCalories = computed(() => {
  const activeMinutes = fullBodyMotionFeatures.value.activeFullBodyMs / 60_000;

  return (currentFullBodyMet.value * 3.5 * weightKg.value * activeMinutes) / 200;
});

const jumpingJackFrequencyPerMin = computed(() => {
  const elapsedMinutes = jumpingJackElapsedMs.value / 60_000;

  return elapsedMinutes > 0 ? jumpingJackCount.value / elapsedMinutes : 0;
});

const currentJumpingJackMet = computed(() => {
  const frequency = jumpingJackFrequencyPerMin.value;

  if (!jumpingJackCount.value || !jumpingJackElapsedMs.value) {
    return 0;
  }

  if (frequency > 50) {
    return 10.0;
  }

  if (frequency > 30) {
    return 8.0;
  }

  return 6.0;
});

const jumpingJackCalories = computed(() => {
  const elapsedMinutes = jumpingJackElapsedMs.value / 60_000;

  return (currentJumpingJackMet.value * 3.5 * weightKg.value * elapsedMinutes) / 200;
});

const fullBodyTrackedCompleteness = computed(() => {
  const latestSample = fullBodyMotionHistory[fullBodyMotionHistory.length - 1];

  if (!latestSample) {
    return 0;
  }

  const visibleCount = FULL_BODY_TRACKED_KEYS.filter((key) =>
    Boolean(latestSample.points[key]),
  ).length;

  return (visibleCount / FULL_BODY_TRACKED_KEYS.length) * 100;
});

const fullBodyConfidence = computed<ConfidenceLevel>(() => {
  if (poseStatus.value !== "detected" || modelStatus.value !== "ready") {
    return "低";
  }

  if (
    landmarkCompleteness.value >= 85 &&
    fullBodyTrackedCompleteness.value >= 80 &&
    fullBodyMotionFeatures.value.activeFullBodyMs >= 5000 &&
    fullBodyActivityScore.value >= 35
  ) {
    return "高";
  }

  if (
    landmarkCompleteness.value >= 60 &&
    fullBodyTrackedCompleteness.value >= 60 &&
    fullBodyMotionHistory.length >= 3
  ) {
    return "中";
  }

  return "低";
});

const fullBodyConfidenceReasons = computed(() => {
  const reasons: string[] = [];

  if (poseStatus.value !== "detected") {
    reasons.push("未稳定检测到完整人体");
  }

  if (modelStatus.value !== "ready") {
    reasons.push("姿态模型尚未就绪");
  }

  if (landmarkCompleteness.value < 60) {
    reasons.push("开合跳关键点完整率较低");
  } else if (landmarkCompleteness.value >= 85) {
    reasons.push("开合跳关键点较完整");
  }

  if (fullBodyTrackedCompleteness.value < 60) {
    reasons.push("全身跟踪关键点不足");
  } else if (fullBodyTrackedCompleteness.value >= 80) {
    reasons.push("全身关键点覆盖较好");
  }

  if (fullBodyMotionFeatures.value.activeFullBodyMs < 3000) {
    reasons.push("有效活动时间较短");
  }

  if (
    fullBodyActivityScore.value < 15 &&
    fullBodyMotionFeatures.value.activeFullBodyMs > 0
  ) {
    reasons.push("全身运动量较小");
  }

  if (fullBodyMotionFeatures.value.movementAmplitude < 0.04) {
    reasons.push("全身动作幅度较小");
  }

  if (!reasons.length) {
    reasons.push("关键点和运动量信号稳定");
  }

  return reasons;
});

const fullBodyConfidenceSummary = computed(() =>
  fullBodyConfidenceReasons.value.slice(0, 3).join("；"),
);

const jumpingJackQualityFeedback = computed(() => {
  const feedback: string[] = [];

  if (jumpingJackRuntimeStatus.value === "idle") {
    feedback.push("点击开始后再评估开合跳周期");
  }

  if (poseStatus.value !== "detected") {
    feedback.push("请保持全身入镜");
  }

  if (landmarkCompleteness.value < 75) {
    feedback.push("肩、腕、髋、踝关键点不够完整");
  }

  if (jumpingJackRuntimeStatus.value === "running") {
    if (feetState.value !== "open" && jumpingJackCounterPhase.value === "ready") {
      feedback.push("准备 open 阶段时双脚打开幅度可能不足");
    }

    if (armsState.value !== "open" && jumpingJackCounterPhase.value === "ready") {
      feedback.push("准备 open 阶段时手臂抬起或打开不足");
    }

    if (jumpingJackCounterPhase.value === "waiting_closed") {
      feedback.push("先回到双脚合拢和双臂回落的起始姿势");
    }

    if (jumpingJackCount.value < 2 && jumpingJackElapsedMs.value > 5000) {
      feedback.push("完整周期样本较少，频率和 cal 仅供参考");
    }
  }

  if (
    jumpingJackFrequencyPerMin.value > 0 &&
    (jumpingJackFrequencyPerMin.value < 10 || jumpingJackFrequencyPerMin.value > 80)
  ) {
    feedback.push("当前开合跳频率可能不稳定");
  }

  if (!feedback.length) {
    feedback.push("开合跳周期信号正常");
  }

  return feedback;
});

const jumpingJackQualitySummary = computed(() =>
  jumpingJackQualityFeedback.value.slice(0, 3).join("；"),
);

const depthDebugSummary = computed(() => {
  if (depthServiceStatus.value !== "ready") {
    return depthServiceMessage.value;
  }

  return `左手 ${formatNullableMm(leftWristDepthMm.value)} / 右手 ${formatNullableMm(
    rightWristDepthMm.value,
  )} / 身体 ${formatNullableMm(bodyDepthBaselineMm.value)} / ${leftWristDepthRelation.value} / ${rightWristDepthRelation.value}`;
});

const depthPrimaryMetrics = computed<MetricCard[]>(() => [
  {
    label: "左手腕 depth",
    value: formatNullableMm(leftWristDepthMm.value),
    detail: leftWristDepthRelation.value,
  },
  {
    label: "右手腕 depth",
    value: formatNullableMm(rightWristDepthMm.value),
    detail: rightWristDepthRelation.value,
  },
  {
    label: "身体基准 depth",
    value: formatNullableMm(bodyDepthBaselineMm.value),
    detail: "躯干参考",
  },
]);

const depthDetailMetrics = computed<MetricCard[]>(() => [
  {
    label: "左手 z 轴差值",
    value: formatSignedNullableMm(leftWristDepthOffsetMm.value),
    detail: leftWristDepthSource.value,
  },
  {
    label: "右手 z 轴差值",
    value: formatSignedNullableMm(rightWristDepthOffsetMm.value),
    detail: rightWristDepthSource.value,
  },
  {
    label: "Depth 帧尺寸",
    value: depthFrameSize.value,
    detail: depthServiceMessage.value,
  },
]);

const depthActionEvidenceMetrics = computed<MetricCard[]>(() => [
  getDepthActionEvidence("左手前后基础动作", leftWristDepthOffsetMm.value),
  getDepthActionEvidence("右手前后基础动作", rightWristDepthOffsetMm.value),
]);

const metrics = computed<MetricCard[]>(() => [
  {
    label: "开合跳次数",
    value: String(jumpingJackCount.value),
    detail: "组合动作计数",
  },
  {
    label: "开合跳时长",
    value: formatElapsedTime(jumpingJackElapsedMs.value),
    detail: jumpingJackRuntimeStatus.value,
  },
  {
    label: "开合跳频率",
    value: jumpingJackFrequencyPerMin.value.toFixed(1),
    detail: "次/分钟",
  },
  {
    label: "开合跳 MET",
    value: currentJumpingJackMet.value.toFixed(1),
    detail: "组合动作示例",
  },
  {
    label: "开合跳 cal",
    value: jumpingJackCalories.value.toFixed(2),
    detail: "组合动作估算",
  },
  {
    label: "组合动作阶段",
    value: jumpingJackCounterPhase.value,
    detail: jumpingJackCounterReason.value,
  },
  {
    label: "动作",
    value: jumpingJackState.value,
    detail: `脚 ${feetState.value} / 手 ${armsState.value}`,
  },
  {
    label: "全身活动时长",
    value: formatElapsedTime(fullBodyMotionFeatures.value.activeFullBodyMs),
    detail: "有效活动",
  },
  {
    label: "全身 MET",
    value: currentFullBodyMet.value.toFixed(1),
    detail: fullBodyIntensityLevel.value,
  },
  {
    label: "全身 cal",
    value: fullBodyCalories.value.toFixed(2),
    detail: "通用活动估算",
  },
  {
    label: "全身估算置信度",
    value: fullBodyConfidence.value,
    detail: fullBodyConfidenceSummary.value,
  },
  {
    label: "动作质量反馈",
    value: jumpingJackQualitySummary.value,
    detail: "开合跳示例",
  },
  {
    label: "全身活动分数",
    value: String(fullBodyActivityScore.value),
    detail: fullBodyIntensityLevel.value,
  },
  {
    label: "基本动作候选",
    value: topFullBodyBasicActionsText.value,
    detail: "非唯一动作名",
  },
  {
    label: "全身位移",
    value: fullBodyMotionFeatures.value.movementDistance.toFixed(3),
    detail: "最近帧",
  },
  {
    label: "全身速度",
    value: fullBodyMotionFeatures.value.movementSpeed.toFixed(2),
    detail: "归一化/秒",
  },
  {
    label: "全身幅度",
    value: fullBodyMotionFeatures.value.movementAmplitude.toFixed(3),
    detail: "1.2秒窗口",
  },
]);

const modelDetails = computed(() => [
  { label: "tasks-vision", value: MEDIAPIPE_TASKS_VERSION },
  { label: "wasm", value: MEDIAPIPE_WASM_URL },
  { label: "pose model", value: POSE_LANDMARKER_MODEL_URL },
  { label: "hand model", value: HAND_LANDMARKER_MODEL_URL },
]);

const poseDebugDetails = computed(() => [
  { label: "人体检测", value: poseStatus.value },
  { label: "检测到手数", value: String(handCount.value) },
  { label: "手部模型", value: handModelStatus.value },
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
  { label: "开合跳记录", value: jumpingJackRuntimeStatus.value },
  { label: "开合跳次数", value: String(jumpingJackCount.value) },
  { label: "开合跳时长", value: formatElapsedTime(jumpingJackElapsedMs.value) },
  { label: "开合跳频率", value: jumpingJackFrequencyPerMin.value.toFixed(1) },
  { label: "开合跳 MET", value: currentJumpingJackMet.value.toFixed(1) },
  { label: "开合跳 cal", value: jumpingJackCalories.value.toFixed(2) },
  { label: "组合动作阶段", value: jumpingJackCounterPhase.value },
  { label: "组合动作计数", value: jumpingJackCounterReason.value },
  { label: "防抖进度", value: actionDebounceProgress.value },
  { label: "状态判断", value: actionStateReason.value },
  { label: "全身关键点", value: fullBodyJointState.value },
  { label: "全身活动分数", value: String(fullBodyActivityScore.value) },
  { label: "全身强度档位", value: fullBodyIntensityLevel.value },
  { label: "全身 MET", value: currentFullBodyMet.value.toFixed(1) },
  { label: "全身 cal", value: fullBodyCalories.value.toFixed(2) },
  { label: "全身跟踪完整率", value: `${fullBodyTrackedCompleteness.value.toFixed(0)}%` },
  { label: "全身估算置信度", value: fullBodyConfidence.value },
  { label: "置信度原因", value: fullBodyConfidenceSummary.value },
  { label: "质量反馈", value: jumpingJackQualitySummary.value },
  { label: "Depth 服务", value: depthServiceStatus.value },
  { label: "Depth 摘要", value: depthDebugSummary.value },
  { label: "Depth 帧尺寸", value: depthFrameSize.value },
  { label: "左手腕 depth", value: formatNullableMm(leftWristDepthMm.value) },
  { label: "右手腕 depth", value: formatNullableMm(rightWristDepthMm.value) },
  { label: "身体基准 depth", value: formatNullableMm(bodyDepthBaselineMm.value) },
  { label: "身体基准来源", value: bodyDepthSource.value },
  { label: "左手 z 轴差值", value: formatSignedNullableMm(leftWristDepthOffsetMm.value) },
  { label: "右手 z 轴差值", value: formatSignedNullableMm(rightWristDepthOffsetMm.value) },
  { label: "左手前后状态", value: leftWristDepthRelation.value },
  { label: "右手前后状态", value: rightWristDepthRelation.value },
  { label: "左手采样来源", value: leftWristDepthSource.value },
  { label: "右手采样来源", value: rightWristDepthSource.value },
  { label: "基本动作候选", value: topFullBodyBasicActionsText.value },
  { label: "全身位移", value: fullBodyMotionFeatures.value.movementDistance.toFixed(3) },
  { label: "上肢位移", value: fullBodyMotionFeatures.value.upperBodyDistance.toFixed(3) },
  { label: "下肢位移", value: fullBodyMotionFeatures.value.lowerBodyDistance.toFixed(3) },
  { label: "躯干位移", value: fullBodyMotionFeatures.value.torsoDistance.toFixed(3) },
  { label: "全身速度", value: fullBodyMotionFeatures.value.movementSpeed.toFixed(2) },
  { label: "全身幅度", value: fullBodyMotionFeatures.value.movementAmplitude.toFixed(3) },
  { label: "全身活动时长", value: formatElapsedTime(fullBodyMotionFeatures.value.activeFullBodyMs) },
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

// ---------------------------------------------------------------------------
// Backend WebSocket
// ---------------------------------------------------------------------------

function connectBackendWs() {
  if (backendWs && backendWs.readyState <= WebSocket.OPEN) {
    return;
  }
  backendWsStatus.value = "connecting";
  const ws = new WebSocket(ACTION_SERVER_WS_URL);

  ws.onopen = () => {
    backendWsStatus.value = "connected";
    backendFrameIndex = 0;
  };

  ws.onmessage = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data as string) as {
        actions?: BackendActionResult[];
        latency_ms?: number;
        error?: string;
      };
      if (data.error) return;
      if (data.actions) backendActions.value = data.actions;
      if (data.latency_ms != null) backendLatencyMs.value = data.latency_ms;
    } catch {
      // ignore malformed frames
    }
  };

  ws.onerror = () => {
    backendWsStatus.value = "error";
  };

  ws.onclose = () => {
    backendWs = null;
    if (backendWsStatus.value !== "disconnected") {
      // Auto-reconnect after 2 s
      backendWsStatus.value = "disconnected";
      backendWsReconnectTimer = setTimeout(connectBackendWs, 2000);
    }
  };

  backendWs = ws;
}

function disconnectBackendWs() {
  if (backendWsReconnectTimer !== null) {
    clearTimeout(backendWsReconnectTimer);
    backendWsReconnectTimer = null;
  }
  backendWsStatus.value = "disconnected";
  backendWs?.close();
  backendWs = null;
  backendActions.value = [];
  backendLatencyMs.value = null;
}

function sendLandmarksToBackend(landmarks: LandmarkPoint[], timestampMs: number) {
  if (!backendWs || backendWs.readyState !== WebSocket.OPEN) return;

  // Build named-key landmark dict for the backend
  const named: Record<string, { x: number; y: number; z?: number; visibility?: number }> = {};
  for (const [camel, backendName] of Object.entries(BACKEND_LANDMARK_KEY_MAP)) {
    const idx = LANDMARK_INDEX[camel as keyof typeof LANDMARK_INDEX];
    const lm = landmarks[idx];
    if (lm) {
      named[backendName] = {
        x: lm.x,
        y: lm.y,
        z: lm.z ?? 0,
        visibility: lm.visibility ?? 0,
      };
    }
  }

  const msg = JSON.stringify({
    frame_index: backendFrameIndex++,
    timestamp_ms: timestampMs,
    landmarks: named,
  });
  backendWs.send(msg);
}

// ---------------------------------------------------------------------------

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
  latestHands = [];
  handCount.value = 0;
  resetJumpingJackAnalysis("等待完整关键点");
  resetJumpingJackCounter();
  resetFullBodyMotionFeatures();
  disconnectBackendWs();
}

function disposePoseLandmarker() {
  poseLandmarker?.close();
  poseLandmarker = null;
}

function disposeHandLandmarker() {
  handLandmarker?.close();
  handLandmarker = null;
}

async function loadHandLandmarker() {
  if (handLandmarker || handModelStatus.value === "loading") {
    return;
  }

  handModelStatus.value = "loading";
  handModelMessage.value = "正在加载 MediaPipe Hand Landmarker";

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

    handModelStatus.value = "ready";
    handModelMessage.value = "手部模型已就绪，将叠加 21 点手部骨架。";
  } catch (error) {
    disposeHandLandmarker();
    handModelStatus.value = "error";
    handModelMessage.value =
      error instanceof Error
        ? `手部模型加载失败：${error.message}`
        : "手部模型加载失败，请检查网络和模型资源路径。";
  }
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

function resetJumpingJackCounter() {
  jumpingJackRuntimeStatus.value = "idle";
  jumpingJackCounterPhase.value = "waiting_closed";
  jumpingJackCount.value = 0;
  jumpingJackElapsedMs.value = 0;
  lastJumpingJackRuntimeAt = 0;
  jumpingJackCounterReason.value = "点击开始后，从双脚合拢和双臂回落开始记录。";
}

function startJumpingJackRecording() {
  if (jumpingJackRuntimeStatus.value === "running") {
    return;
  }

  jumpingJackRuntimeStatus.value = "running";
  lastJumpingJackRuntimeAt = performance.now();

  if (jumpingJackCounterPhase.value === "waiting_closed") {
    jumpingJackCounterReason.value = "等待 closed 起始姿势，再进入 ready。";
  } else {
    jumpingJackCounterReason.value = "记录中，等待完整 closed → open → closed 周期。";
  }
}

function pauseJumpingJackRecording() {
  if (jumpingJackRuntimeStatus.value !== "running") {
    return;
  }

  jumpingJackRuntimeStatus.value = "paused";
  lastJumpingJackRuntimeAt = 0;
  jumpingJackCounterReason.value = "已暂停，组合动作状态机保持当前阶段。";
}

function resetJumpingJackRecording() {
  resetJumpingJackCounter();
  resetJumpingJackAnalysis("等待完整关键点");
}

function updateJumpingJackRuntime(now: number) {
  if (jumpingJackRuntimeStatus.value !== "running") {
    lastJumpingJackRuntimeAt = 0;
    return;
  }

  if (!lastJumpingJackRuntimeAt) {
    lastJumpingJackRuntimeAt = now;
    return;
  }

  jumpingJackElapsedMs.value += now - lastJumpingJackRuntimeAt;
  lastJumpingJackRuntimeAt = now;
}

function updateJumpingJackCounter(state: JumpingJackState) {
  if (jumpingJackRuntimeStatus.value !== "running") {
    return;
  }

  if (state === "unknown") {
    jumpingJackCounterReason.value = "当前状态 unknown，组合动作状态机暂不推进。";
    return;
  }

  if (jumpingJackCounterPhase.value === "waiting_closed") {
    if (state === "closed") {
      jumpingJackCounterPhase.value = "ready";
      jumpingJackCounterReason.value = "已捕获 closed 起始姿势，等待 open。";
    } else {
      jumpingJackCounterReason.value = "请先回到 closed 起始姿势。";
    }
    return;
  }

  if (jumpingJackCounterPhase.value === "ready") {
    if (state === "open") {
      jumpingJackCounterPhase.value = "opened";
      jumpingJackCounterReason.value = "已捕获 open，等待回到 closed 完成 1 次。";
    }
    return;
  }

  if (jumpingJackCounterPhase.value === "opened" && state === "closed") {
    jumpingJackCount.value += 1;
    jumpingJackCounterPhase.value = "ready";
    jumpingJackCounterReason.value = "完成 closed → open → closed，计数 +1。";
  }
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

function formatElapsedTime(ms: number) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60)
    .toString()
    .padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");

  return `${minutes}:${seconds}`;
}

function formatNullableMm(value: number | null) {
  return value === null ? "无" : `${Math.round(value)}mm`;
}

function formatSignedNullableMm(value: number | null) {
  if (value === null) {
    return "无";
  }

  const rounded = Math.round(value);

  return `${rounded > 0 ? "+" : ""}${rounded}mm`;
}

function getDepthRelation(offsetMm: number | null) {
  if (offsetMm === null) {
    return "无";
  }

  if (Math.abs(offsetMm) >= DEPTH_MAX_REASONABLE_OFFSET_MM) {
    return "疑似采到背景";
  }

  if (offsetMm <= -DEPTH_FRONT_BACK_THRESHOLD_MM) {
    return "在身体前";
  }

  if (offsetMm >= DEPTH_FRONT_BACK_THRESHOLD_MM) {
    return "在身体后";
  }

  return "接近身体平面";
}

function getDepthActionEvidence(label: string, offsetMm: number | null): MetricCard {
  if (offsetMm === null) {
    return {
      label,
      value: "无",
      detail: "等待 wrist/body depth",
    };
  }

  if (Math.abs(offsetMm) >= DEPTH_MAX_REASONABLE_OFFSET_MM) {
    return {
      label,
      value: "暂不采信",
      detail: `疑似采到背景 / ${formatSignedNullableMm(offsetMm)}`,
    };
  }

  const magnitude = Math.abs(offsetMm);
  const score = clampScore(
    ((magnitude - DEPTH_FRONT_BACK_THRESHOLD_MM) /
      (DEPTH_ACTION_MAX_SCORE_OFFSET_MM - DEPTH_FRONT_BACK_THRESHOLD_MM)) *
      100,
  );

  if (offsetMm <= -DEPTH_FRONT_BACK_THRESHOLD_MM) {
    return {
      label,
      value: `手前移 ${score}`,
      detail: `${formatSignedNullableMm(offsetMm)}，depth 证据`,
    };
  }

  if (offsetMm >= DEPTH_FRONT_BACK_THRESHOLD_MM) {
    return {
      label,
      value: `手后移 ${score}`,
      detail: `${formatSignedNullableMm(offsetMm)}，depth 证据`,
    };
  }

  return {
    label,
    value: "接近平面",
    detail: `${formatSignedNullableMm(offsetMm)}，低于 ${DEPTH_FRONT_BACK_THRESHOLD_MM}mm 阈值`,
  };
}

function getPointDistance(left: LandmarkPoint, right: LandmarkPoint) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function getVisibleTrackedPoint(
  landmarks: LandmarkPoint[],
  key: FullBodyTrackedKey,
) {
  const landmark = landmarks[LANDMARK_INDEX[key]];

  return isVisibleLandmark(landmark) ? landmark : null;
}

function getFullBodyMotionSample(
  landmarks: LandmarkPoint[],
  now: number,
): FullBodyMotionSample {
  const points: FullBodyMotionSample["points"] = {};

  SAMPLE_TRACKED_KEYS.forEach((key) => {
    const landmark = getVisibleTrackedPoint(landmarks, key);

    if (landmark) {
      points[key] = landmark;
    }
  });

  return { time: now, points };
}

function getAverageTrackedDistance(
  previous: FullBodyMotionSample,
  current: FullBodyMotionSample,
  keys: FullBodyTrackedKey[],
) {
  const distances = keys
    .map((key) => {
      const previousPoint = previous.points[key];
      const currentPoint = current.points[key];

      return previousPoint && currentPoint
        ? getPointDistance(previousPoint, currentPoint)
        : null;
    })
    .filter((distance): distance is number => distance !== null);

  if (!distances.length) {
    return 0;
  }

  return distances.reduce((sum, distance) => sum + distance, 0) / distances.length;
}

function getTrackedAmplitude(
  samples: FullBodyMotionSample[],
  keys: FullBodyTrackedKey[],
) {
  const amplitudes = keys.map((key) => {
    const points = samples
      .map((sample) => sample.points[key])
      .filter((point): point is LandmarkPoint => Boolean(point));

    if (points.length < 2) {
      return 0;
    }

    const xValues = points.map((point) => point.x);
    const yValues = points.map((point) => point.y);
    const xRange = Math.max(...xValues) - Math.min(...xValues);
    const yRange = Math.max(...yValues) - Math.min(...yValues);

    return Math.hypot(xRange, yRange);
  });

  return Math.max(...amplitudes, 0);
}

function clampScore(value: number) {
  return Math.min(100, Math.max(0, Math.round(value)));
}

function scoreFromRange(value: number, min: number, max: number) {
  if (max <= min) {
    return value >= max ? 100 : 0;
  }

  return clampScore(((value - min) / (max - min)) * 100);
}

function scoreFromInverseRange(value: number, min: number, max: number) {
  if (max <= min) {
    return value <= min ? 100 : 0;
  }

  return clampScore(((max - value) / (max - min)) * 100);
}

function getFullBodyIntensityLevel(score: number): FullBodyIntensityLevel {
  if (score < 15) {
    return "静止";
  }

  if (score < 40) {
    return "低强度";
  }

  if (score < 70) {
    return "中强度";
  }

  return "高强度";
}

function getAverageTrackedY(
  sample: FullBodyMotionSample,
  keys: FullBodyTrackedKey[],
) {
  const points = keys
    .map((key) => sample.points[key])
    .filter((point): point is LandmarkPoint => Boolean(point));

  if (!points.length) {
    return null;
  }

  return points.reduce((sum, point) => sum + point.y, 0) / points.length;
}

function getAverageTrackedX(
  sample: FullBodyMotionSample,
  keys: FullBodyTrackedKey[],
) {
  const points = keys
    .map((key) => sample.points[key])
    .filter((point): point is LandmarkPoint => Boolean(point));

  if (!points.length) {
    return null;
  }

  return points.reduce((sum, point) => sum + point.x, 0) / points.length;
}

function getTrackedDistance(
  sample: FullBodyMotionSample,
  firstKey: FullBodyTrackedKey,
  secondKey: FullBodyTrackedKey,
) {
  const first = sample.points[firstKey];
  const second = sample.points[secondKey];

  return first && second ? getPointDistance(first, second) : null;
}

function getTorsoHeight(sample: FullBodyMotionSample) {
  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const hipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);

  if (shoulderY === null || hipY === null) {
    return null;
  }

  return Math.max(Math.abs(hipY - shoulderY), 0.05);
}

function getLimbLiftScore(
  sample: FullBodyMotionSample,
  hipKey: FullBodyTrackedKey,
  kneeKey: FullBodyTrackedKey,
) {
  const hip = sample.points[hipKey];
  const knee = sample.points[kneeKey];
  const torsoHeight = getTorsoHeight(sample);

  if (!hip || !knee || torsoHeight === null) {
    return 0;
  }

  const kneeGapRatio = Math.abs(knee.y - hip.y) / torsoHeight;

  return scoreFromInverseRange(kneeGapRatio, 0.18, 0.62);
}

function getHipVerticalDelta(
  previous: FullBodyMotionSample | null,
  current: FullBodyMotionSample,
) {
  if (!previous) {
    return 0;
  }

  const previousHipY = getAverageTrackedY(previous, ["leftHip", "rightHip"]);
  const currentHipY = getAverageTrackedY(current, ["leftHip", "rightHip"]);

  if (previousHipY === null || currentHipY === null) {
    return 0;
  }

  return currentHipY - previousHipY;
}

function getTrackedDistanceRatio(
  sample: FullBodyMotionSample,
  firstKey: FullBodyTrackedKey,
  secondKey: FullBodyTrackedKey,
  baseDistance: number,
) {
  const distance = getTrackedDistance(sample, firstKey, secondKey);

  return distance && baseDistance > 0 ? distance / baseDistance : null;
}

function getLowerBodyMovementScore(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  if (!previous) {
    return 0;
  }

  const distance = getAverageTrackedDistance(previous, sample, LOWER_BODY_TRACKED_KEYS);

  return scoreFromRange(distance, 0.018, 0.07);
}

function getSquatPostureScore(sample: FullBodyMotionSample) {
  const torsoHeight = getTorsoHeight(sample);
  const hipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);
  const kneeY = getAverageTrackedY(sample, ["leftKnee", "rightKnee"]);
  const ankleY = getAverageTrackedY(sample, ["leftAnkle", "rightAnkle"]);

  if (torsoHeight === null || hipY === null || kneeY === null || ankleY === null) {
    return 0;
  }

  const kneeToHipRatio = (kneeY - hipY) / torsoHeight;
  const ankleToHipRatio = (ankleY - hipY) / torsoHeight;
  const kneeBendScore = scoreFromInverseRange(kneeToHipRatio, 0.55, 1.25);
  const hipDropScore = scoreFromInverseRange(ankleToHipRatio, 1.1, 2.25);

  return clampScore(kneeBendScore * 0.65 + hipDropScore * 0.35);
}

function getTrackedDelta(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
  key: FullBodyTrackedKey,
) {
  const current = sample.points[key];
  const last = previous?.points[key];

  if (!current || !last) {
    return null;
  }

  return {
    x: current.x - last.x,
    y: current.y - last.y,
    distance: getPointDistance(current, last),
  };
}

function getElbowAngleScore(
  sample: FullBodyMotionSample,
  side: "left" | "right",
  target: "extend" | "bend",
) {
  const shoulder = sample.points[side === "left" ? "leftShoulder" : "rightShoulder"];
  const elbow = sample.points[side === "left" ? "leftElbow" : "rightElbow"];
  const wrist = sample.points[side === "left" ? "leftWrist" : "rightWrist"];

  if (!shoulder || !elbow || !wrist) {
    return 0;
  }

  const shoulderVector = { x: shoulder.x - elbow.x, y: shoulder.y - elbow.y };
  const wristVector = { x: wrist.x - elbow.x, y: wrist.y - elbow.y };
  const shoulderLength = Math.hypot(shoulderVector.x, shoulderVector.y);
  const wristLength = Math.hypot(wristVector.x, wristVector.y);

  if (shoulderLength <= 0 || wristLength <= 0) {
    return 0;
  }

  const cosine =
    (shoulderVector.x * wristVector.x + shoulderVector.y * wristVector.y) /
    (shoulderLength * wristLength);
  const angle = (Math.acos(Math.min(Math.max(cosine, -1), 1)) * 180) / Math.PI;

  return target === "extend"
    ? scoreFromRange(angle, 135, 170)
    : scoreFromInverseRange(angle, 75, 135);
}

function getArmLiftScore(sample: FullBodyMotionSample, side: "left" | "right") {
  const shoulder = sample.points[side === "left" ? "leftShoulder" : "rightShoulder"];
  const elbow = sample.points[side === "left" ? "leftElbow" : "rightElbow"];
  const wrist = sample.points[side === "left" ? "leftWrist" : "rightWrist"];
  const torsoHeight = getTorsoHeight(sample);

  if (!shoulder || !wrist || torsoHeight === null) {
    return 0;
  }

  const wristLiftRatio = (shoulder.y - wrist.y) / torsoHeight;
  const elbowLiftRatio = elbow ? (shoulder.y - elbow.y) / torsoHeight : 0;

  return clampScore(
    scoreFromRange(wristLiftRatio, -0.02, 0.42) * 0.75 +
      scoreFromRange(elbowLiftRatio, -0.08, 0.26) * 0.25,
  );
}

function getArmDropScore(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  const torsoHeight = getTorsoHeight(sample);

  if (!previous || torsoHeight === null) {
    return 0;
  }

  const deltas = ["leftWrist", "rightWrist"]
    .map((key) => getTrackedDelta(sample, previous, key as FullBodyTrackedKey))
    .filter((delta): delta is { x: number; y: number; distance: number } => Boolean(delta));

  if (!deltas.length) {
    return 0;
  }

  const downwardRatio =
    deltas.reduce((total, delta) => total + Math.max(delta.y, 0), 0) /
    deltas.length /
    torsoHeight;

  return scoreFromRange(downwardRatio, 0.04, 0.22);
}

function getHorizontalHandSwingScore(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
  shoulderWidth: number,
) {
  if (!previous || shoulderWidth <= 0) {
    return 0;
  }

  const deltas = ["leftWrist", "rightWrist"]
    .map((key) => getTrackedDelta(sample, previous, key as FullBodyTrackedKey))
    .filter((delta): delta is { x: number; y: number; distance: number } => Boolean(delta));

  if (!deltas.length) {
    return 0;
  }

  const horizontalRatio =
    deltas.reduce((total, delta) => total + Math.abs(delta.x), 0) /
    deltas.length /
    shoulderWidth;

  return scoreFromRange(horizontalRatio, 0.08, 0.34);
}

function getDepthMotionScore(offsetMm: number | null, direction: "front" | "back") {
  if (
    offsetMm === null ||
    depthServiceStatus.value !== "ready" ||
    Math.abs(offsetMm) >= DEPTH_MAX_REASONABLE_OFFSET_MM
  ) {
    return 0;
  }

  const signedOffset = direction === "front" ? -offsetMm : offsetMm;

  return scoreFromRange(
    signedOffset,
    DEPTH_FRONT_BACK_THRESHOLD_MM,
    DEPTH_ACTION_MAX_SCORE_OFFSET_MM,
  );
}

function getHandForwardScore() {
  return Math.max(
    getDepthMotionScore(leftWristDepthOffsetMm.value, "front"),
    getDepthMotionScore(rightWristDepthOffsetMm.value, "front"),
  );
}

function getHandBackwardScore() {
  return Math.max(
    getDepthMotionScore(leftWristDepthOffsetMm.value, "back"),
    getDepthMotionScore(rightWristDepthOffsetMm.value, "back"),
  );
}

function getTorsoLeanScores(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
  shoulderWidth: number,
) {
  const shoulderX = getAverageTrackedX(sample, ["leftShoulder", "rightShoulder"]);
  const hipX = getAverageTrackedX(sample, ["leftHip", "rightHip"]);
  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const hipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);
  const lastShoulderY = previous
    ? getAverageTrackedY(previous, ["leftShoulder", "rightShoulder"])
    : null;
  const lastHipY = previous ? getAverageTrackedY(previous, ["leftHip", "rightHip"]) : null;

  const lateralRatio =
    shoulderX !== null && hipX !== null && shoulderWidth > 0
      ? (shoulderX - hipX) / shoulderWidth
      : 0;
  const shoulderHipDelta =
    shoulderY !== null && hipY !== null && lastShoulderY !== null && lastHipY !== null
      ? (shoulderY - lastShoulderY) - (hipY - lastHipY)
      : 0;
  const torsoHeight = getTorsoHeight(sample);
  const pitchRatio = torsoHeight !== null ? shoulderHipDelta / torsoHeight : 0;
  const leftShoulder = sample.points.leftShoulder;
  const rightShoulder = sample.points.rightShoulder;
  const shoulderZDelta =
    leftShoulder?.z !== undefined && rightShoulder?.z !== undefined
      ? leftShoulder.z - rightShoulder.z
      : 0;

  return {
    leftLean: scoreFromRange(-lateralRatio, 0.18, 0.55),
    rightLean: scoreFromRange(lateralRatio, 0.18, 0.55),
    forwardLean: scoreFromRange(pitchRatio, 0.06, 0.24),
    backwardLean: scoreFromRange(-pitchRatio, 0.06, 0.24),
    leftTurn: scoreFromRange(shoulderZDelta, 0.08, 0.24),
    rightTurn: scoreFromRange(-shoulderZDelta, 0.08, 0.24),
  };
}

function getHeadMotionScores(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  const noseDelta = getTrackedDelta(sample, previous, "nose");
  const earDistance = getTrackedDistance(sample, "leftEar", "rightEar");
  const earCenterX = getAverageTrackedX(sample, ["leftEar", "rightEar"]);
  const nose = sample.points.nose;
  const headBase = Math.max(earDistance ?? 0, MIN_SHOULDER_WIDTH);
  const noseOffsetRatio =
    nose && earCenterX !== null && headBase > 0 ? (nose.x - earCenterX) / headBase : 0;
  const horizontalRatio = noseDelta ? Math.abs(noseDelta.x) / headBase : 0;
  const verticalRatio = noseDelta ? Math.abs(noseDelta.y) / headBase : 0;

  return {
    nod: scoreFromRange(verticalRatio, 0.08, 0.28),
    shake: scoreFromRange(horizontalRatio, 0.08, 0.28),
    leftTurn: scoreFromRange(-noseOffsetRatio, 0.18, 0.5),
    rightTurn: scoreFromRange(noseOffsetRatio, 0.18, 0.5),
  };
}

function getFullBodyBasicActionCandidates(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  const shoulderDistance = getTrackedDistance(sample, "leftShoulder", "rightShoulder");
  const ankleDistance = getTrackedDistance(sample, "leftAnkle", "rightAnkle");
  const kneeDistance = getTrackedDistance(sample, "leftKnee", "rightKnee");
  const wristDistance = getTrackedDistance(sample, "leftWrist", "rightWrist");
  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const wristY = getAverageTrackedY(sample, ["leftWrist", "rightWrist"]);
  const torsoHeight = getTorsoHeight(sample);
  const shoulderWidth = Math.max(shoulderDistance ?? 0, MIN_SHOULDER_WIDTH);
  const ankleRatio =
    shoulderDistance && ankleDistance ? ankleDistance / shoulderWidth : 0;
  const kneeRatio =
    shoulderDistance && kneeDistance ? kneeDistance / shoulderWidth : 0;
  const previousShoulderWidth = previous
    ? Math.max(
        getTrackedDistance(previous, "leftShoulder", "rightShoulder") ?? 0,
        MIN_SHOULDER_WIDTH,
      )
    : shoulderWidth;
  const previousAnkleRatio = previous
    ? getTrackedDistanceRatio(previous, "leftAnkle", "rightAnkle", previousShoulderWidth)
    : null;
  const ankleRatioDelta = previousAnkleRatio === null ? 0 : ankleRatio - previousAnkleRatio;
  const lowerBodyMovementScore = getLowerBodyMovementScore(sample, previous);
  const footClosingScore = scoreFromRange(-ankleRatioDelta, 0.04, 0.22);
  const footClosedPostureScore = scoreFromInverseRange(ankleRatio, 0.6, 0.95);
  const kneeClosedPostureScore = scoreFromInverseRange(kneeRatio, 0.75, 1.2);
  const footClosedScore =
    Math.min(footClosedPostureScore, kneeClosedPostureScore + 20) *
    (Math.max(footClosingScore, lowerBodyMovementScore * 0.55) / 100);
  const wristRatio =
    shoulderDistance && wristDistance ? wristDistance / shoulderWidth : 0;
  const armLiftRatio =
    shoulderY !== null && wristY !== null && torsoHeight !== null
      ? (shoulderY - wristY) / torsoHeight
      : 0;
  const leftArmLiftScore = getArmLiftScore(sample, "left");
  const rightArmLiftScore = getArmLiftScore(sample, "right");
  const bothArmsLiftScore =
    Math.min(leftArmLiftScore, rightArmLiftScore) * 0.75 +
    Math.max(leftArmLiftScore, rightArmLiftScore) * 0.25;
  const armDropScore = getArmDropScore(sample, previous);
  const handForwardScore = getHandForwardScore();
  const handBackwardScore = getHandBackwardScore();
  const horizontalHandSwingScore = getHorizontalHandSwingScore(sample, previous, shoulderWidth);
  const upperBodyContextScore = Math.max(
    leftArmLiftScore,
    rightArmLiftScore,
    armDropScore,
    handForwardScore,
    handBackwardScore,
    horizontalHandSwingScore,
  );
  const rawArmExtensionScore = Math.max(
    getElbowAngleScore(sample, "left", "extend"),
    getElbowAngleScore(sample, "right", "extend"),
  );
  const armExtensionScore =
    rawArmExtensionScore * (Math.max(upperBodyContextScore, 20) / 100);
  const armBendScore = Math.max(
    getElbowAngleScore(sample, "left", "bend"),
    getElbowAngleScore(sample, "right", "bend"),
  );
  const torsoScores = getTorsoLeanScores(sample, previous, shoulderWidth);
  const headScores = getHeadMotionScores(sample, previous);
  const hipDelta = getHipVerticalDelta(previous, sample);
  const squatPostureScore = getSquatPostureScore(sample);
  const previousSquatPostureScore = previous ? getSquatPostureScore(previous) : 0;
  const squatDescentScore =
    scoreFromRange(hipDelta, 0.012, 0.055) * (squatPostureScore / 100);
  const squatHoldScore = squatPostureScore >= 68 ? squatPostureScore * 0.75 : 0;
  const standUpScore =
    scoreFromRange(-hipDelta, 0.012, 0.055) *
    (Math.max(previousSquatPostureScore, squatPostureScore) / 100);
  const leftLegLiftScore = getLimbLiftScore(sample, "leftHip", "leftKnee");
  const rightLegLiftScore = getLimbLiftScore(sample, "rightHip", "rightKnee");
  const candidates: BasicActionCandidate[] = [
    {
      label: "双脚打开",
      score: scoreFromRange(ankleRatio, 1.05, 1.45),
      reason: "脚踝间距相对肩宽变大",
    },
    {
      label: "双脚合拢",
      score: footClosedScore,
      reason: "脚踝和膝盖距离收窄，并出现合拢变化",
    },
    {
      label: "左手上抬",
      score: leftArmLiftScore,
      reason: "左腕和左肘相对左肩上抬",
    },
    {
      label: "右手上抬",
      score: rightArmLiftScore,
      reason: "右腕和右肘相对右肩上抬",
    },
    {
      label: "双臂上抬",
      score: Math.max(
        bothArmsLiftScore,
        scoreFromRange(armLiftRatio, 0.05, 0.45),
        scoreFromRange(wristRatio, 0.9, 1.35) * 0.7,
      ),
      reason: "手腕高于肩部或双手横向打开",
    },
    {
      label: "手臂回落",
      score: armDropScore,
      reason: "手腕相对上一帧向下移动",
    },
    {
      label: "手向前移动",
      score: handForwardScore,
      reason: "RGB-D 显示手腕接近摄像头",
    },
    {
      label: "手向后移动",
      score: handBackwardScore,
      reason: "RGB-D 显示手腕远离摄像头",
    },
    {
      label: "手水平摆动",
      score: horizontalHandSwingScore,
      reason: "手腕横向位移明显",
    },
    {
      label: "手臂伸展",
      score: armExtensionScore,
      reason: "肩肘腕夹角接近伸直，并伴随上肢活动证据",
    },
    {
      label: "手臂弯曲",
      score: armBendScore,
      reason: "肩肘腕夹角变小",
    },
    {
      label: "左抬腿",
      score: leftLegLiftScore,
      reason: "左膝相对左髋上抬",
    },
    {
      label: "右抬腿",
      score: rightLegLiftScore,
      reason: "右膝相对右髋上抬",
    },
    {
      label: "膝盖上抬",
      score: Math.max(leftLegLiftScore, rightLegLiftScore),
      reason: "任一膝盖相对髋部上抬",
    },
    {
      label: "下蹲",
      score: Math.max(squatDescentScore, squatHoldScore),
      reason: "髋部下移并伴随膝髋结构变化",
    },
    {
      label: "起身",
      score: standUpScore,
      reason: "从下蹲结构中髋部上移",
    },
    {
      label: "身体前倾",
      score: torsoScores.forwardLean,
      reason: "肩部相对髋部向下移动",
    },
    {
      label: "身体后仰",
      score: torsoScores.backwardLean,
      reason: "肩部相对髋部向上移动",
    },
    {
      label: "身体左倾",
      score: torsoScores.leftLean,
      reason: "肩部中心相对髋部中心左移",
    },
    {
      label: "身体右倾",
      score: torsoScores.rightLean,
      reason: "肩部中心相对髋部中心右移",
    },
    {
      label: "身体左转",
      score: torsoScores.leftTurn,
      reason: "左右肩 z 差显示身体旋转",
    },
    {
      label: "身体右转",
      score: torsoScores.rightTurn,
      reason: "左右肩 z 差显示身体旋转",
    },
    {
      label: "点头",
      score: headScores.nod,
      reason: "鼻尖纵向位移明显",
    },
    {
      label: "摇头",
      score: headScores.shake,
      reason: "鼻尖横向位移明显",
    },
    {
      label: "左转头",
      score: headScores.leftTurn,
      reason: "鼻尖相对双耳中心左移",
    },
    {
      label: "右转头",
      score: headScores.rightTurn,
      reason: "鼻尖相对双耳中心右移",
    },
  ];

  return candidates
    .map((candidate) => ({
      ...candidate,
      score: clampScore(candidate.score),
    }))
    .filter((candidate) => candidate.score >= 35)
    .sort((left, right) => right.score - left.score);
}

function updateFullBodyActivityClassification(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  const features = fullBodyMotionFeatures.value;
  const distanceScore = scoreFromRange(features.movementDistance, 0.015, 0.12);
  const speedScore = scoreFromRange(features.movementSpeed, 0.12, 0.8);
  const amplitudeScore = scoreFromRange(features.movementAmplitude, 0.04, 0.32);
  const lowerBodyScore = scoreFromRange(features.lowerBodyDistance, 0.01, 0.08);
  const completenessScore = landmarkCompleteness.value;
  const activityScore = clampScore(
    distanceScore * 0.25 +
      speedScore * 0.25 +
      amplitudeScore * 0.25 +
      lowerBodyScore * 0.15 +
      completenessScore * 0.1,
  );

  fullBodyActivityScore.value = activityScore;
  fullBodyIntensityLevel.value = getFullBodyIntensityLevel(activityScore);
  fullBodyBasicActions.value = getFullBodyBasicActionCandidates(sample, previous);
}

function resetFullBodyMotionFeatures() {
  fullBodyMotionHistory.length = 0;
  fullBodyJointState.value = "等待全身关键点";
  fullBodyMotionFeatures.value = {
    movementDistance: 0,
    upperBodyDistance: 0,
    lowerBodyDistance: 0,
    torsoDistance: 0,
    movementSpeed: 0,
    movementAmplitude: 0,
    activeFullBodyMs: 0,
  };
  fullBodyActivityScore.value = 0;
  fullBodyIntensityLevel.value = "静止";
  fullBodyBasicActions.value = [];
}

function resetDepthDebug(reason = "等待完整关键点") {
  depthServiceStatus.value = "idle";
  depthServiceMessage.value = reason;
  depthFrameSize.value = "未知";
  leftWristDepthMm.value = null;
  rightWristDepthMm.value = null;
  bodyDepthBaselineMm.value = null;
  bodyDepthSource.value = "无";
  leftWristDepthOffsetMm.value = null;
  rightWristDepthOffsetMm.value = null;
  leftWristDepthRelation.value = "无";
  rightWristDepthRelation.value = "无";
  leftWristDepthSource.value = "无";
  rightWristDepthSource.value = "无";
}

function getDepthPoint(
  landmarks: LandmarkPoint[],
  id: keyof typeof LANDMARK_INDEX,
): DepthLookupPoint | null {
  const landmark = landmarks[LANDMARK_INDEX[id]];

  if (!isVisibleLandmark(landmark)) {
    return null;
  }

  return {
    id,
    normX: landmark.x,
    normY: landmark.y,
  };
}

function getAverageDepthPoint(
  landmarks: LandmarkPoint[],
  id: string,
  keys: Array<keyof typeof LANDMARK_INDEX>,
): DepthLookupPoint | null {
  const points = keys
    .map((key) => landmarks[LANDMARK_INDEX[key]])
    .filter((landmark): landmark is LandmarkPoint => isVisibleLandmark(landmark));

  if (!points.length) {
    return null;
  }

  return {
    id,
    normX: points.reduce((total, point) => total + point.x, 0) / points.length,
    normY: points.reduce((total, point) => total + point.y, 0) / points.length,
  };
}

function getValidDepth(points: DepthLookupResult[], id: string) {
  const point = points.find((item) => item.id === id);

  return point?.valid && point.medianDepthMm > 0 ? point.medianDepthMm : null;
}

function getBodyBaseline(points: DepthLookupResult[]) {
  const candidates = [
    { id: "torsoCenter", label: "躯干中心" },
    { id: "shoulderCenter", label: "肩部中心" },
    { id: "hipCenter", label: "髋部中心" },
    { id: "leftShoulder", label: "左肩" },
    { id: "rightShoulder", label: "右肩" },
    { id: "leftHip", label: "左髋" },
    { id: "rightHip", label: "右髋" },
  ]
    .map((candidate) => ({
      ...candidate,
      depth: getValidDepth(points, candidate.id),
    }))
    .filter((candidate): candidate is { id: string; label: string; depth: number } =>
      candidate.depth !== null,
    );

  if (!candidates.length) {
    return { depth: null, source: "无" };
  }

  const depths = candidates.map((candidate) => candidate.depth).sort((a, b) => a - b);
  const middle = Math.floor(depths.length / 2);
  const baseline =
    depths.length % 2 === 0
      ? (depths[middle - 1] + depths[middle]) / 2
      : depths[middle];

  return {
    depth: baseline,
    source: candidates.map((candidate) => `${candidate.label}:${candidate.depth}`).join(" / "),
  };
}

function getWristDepth(points: DepthLookupResult[], id: string) {
  const point = points.find((item) => item.id === id);

  if (!point?.valid) {
    return { depth: null, source: "无" };
  }

  if (point.foregroundDepthMm && point.foregroundDepthMm > 0) {
    const source = point.foregroundCandidate
      ? `前景候选 r${point.validRatio ?? "?"}`
      : `前景窗口 r${point.validRatio ?? "?"}`;

    return { depth: point.foregroundDepthMm, source };
  }

  if (point.medianDepthMm > 0) {
    return { depth: point.medianDepthMm, source: `中心中位数 r${point.validRatio ?? "?"}` };
  }

  return { depth: null, source: "无有效深度" };
}

async function updateDepthDebug(
  landmarks: LandmarkPoint[] | null,
  now: number,
) {
  if (!landmarks?.length) {
    resetDepthDebug("未检测到完整人体");
    return;
  }

  if (isDepthRequesting || now - lastDepthRequestAt < DEPTH_REQUEST_INTERVAL_MS) {
    return;
  }

  const points = [
    getDepthPoint(landmarks, "leftWrist"),
    getDepthPoint(landmarks, "rightWrist"),
    getDepthPoint(landmarks, "leftShoulder"),
    getDepthPoint(landmarks, "rightShoulder"),
    getDepthPoint(landmarks, "leftHip"),
    getDepthPoint(landmarks, "rightHip"),
    getAverageDepthPoint(landmarks, "shoulderCenter", ["leftShoulder", "rightShoulder"]),
    getAverageDepthPoint(landmarks, "hipCenter", ["leftHip", "rightHip"]),
    getAverageDepthPoint(landmarks, "torsoCenter", [
      "leftShoulder",
      "rightShoulder",
      "leftHip",
      "rightHip",
    ]),
  ].filter((point): point is DepthLookupPoint => Boolean(point));

  if (!points.length) {
    resetDepthDebug("缺少可查询 depth 的关键点");
    return;
  }

  isDepthRequesting = true;
  lastDepthRequestAt = now;

  try {
    const response = await fetch(DEPTH_SERVER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ points }),
    });
    const data = (await response.json()) as DepthLookupResponse;

    if (!response.ok || !data.ok || !data.points) {
      throw new Error(data.error || `depth server ${response.status}`);
    }

    const baseline = getBodyBaseline(data.points);
    const leftWristDepth = getWristDepth(data.points, "leftWrist");
    const rightWristDepth = getWristDepth(data.points, "rightWrist");

    depthServiceStatus.value = "ready";
    depthServiceMessage.value = "depth server 已连接";
    depthFrameSize.value =
      data.width && data.height ? `${data.width}x${data.height}` : "未知";
    bodyDepthBaselineMm.value = baseline.depth;
    bodyDepthSource.value = baseline.source;
    leftWristDepthMm.value = leftWristDepth.depth;
    rightWristDepthMm.value = rightWristDepth.depth;
    leftWristDepthSource.value = leftWristDepth.source;
    rightWristDepthSource.value = rightWristDepth.source;
    leftWristDepthOffsetMm.value =
      leftWristDepth.depth !== null && baseline.depth !== null
        ? leftWristDepth.depth - baseline.depth
        : null;
    rightWristDepthOffsetMm.value =
      rightWristDepth.depth !== null && baseline.depth !== null
        ? rightWristDepth.depth - baseline.depth
        : null;
    leftWristDepthRelation.value = getDepthRelation(leftWristDepthOffsetMm.value);
    rightWristDepthRelation.value = getDepthRelation(rightWristDepthOffsetMm.value);
  } catch (error) {
    depthServiceStatus.value = "error";
    depthServiceMessage.value =
      error instanceof Error ? error.message : "depth server 请求失败";
  } finally {
    isDepthRequesting = false;
  }
}

function updateFullBodyMotionFeatures(
  landmarks: LandmarkPoint[] | null,
  now: number,
) {
  if (!landmarks?.length) {
    resetFullBodyMotionFeatures();
    return;
  }

  const sample = getFullBodyMotionSample(landmarks, now);
  const visibleCount = FULL_BODY_TRACKED_KEYS.filter((key) =>
    Boolean(sample.points[key]),
  ).length;

  fullBodyJointState.value = `${visibleCount}/${FULL_BODY_TRACKED_KEYS.length}`;
  fullBodyMotionHistory.push(sample);

  while (
    fullBodyMotionHistory.length &&
    now - fullBodyMotionHistory[0].time > FULL_BODY_HISTORY_MS
  ) {
    fullBodyMotionHistory.shift();
  }

  if (fullBodyMotionHistory.length < 2) {
    fullBodyMotionFeatures.value = {
      ...fullBodyMotionFeatures.value,
      movementDistance: 0,
      upperBodyDistance: 0,
      lowerBodyDistance: 0,
      torsoDistance: 0,
      movementSpeed: 0,
      movementAmplitude: 0,
    };
    updateFullBodyActivityClassification(sample, null);
    return;
  }

  const previous = fullBodyMotionHistory[fullBodyMotionHistory.length - 2];
  const elapsedSeconds = Math.max((sample.time - previous.time) / 1000, 0.001);
  const upperBodyDistance = getAverageTrackedDistance(
    previous,
    sample,
    UPPER_BODY_TRACKED_KEYS,
  );
  const lowerBodyDistance = getAverageTrackedDistance(
    previous,
    sample,
    LOWER_BODY_TRACKED_KEYS,
  );
  const torsoDistance = getAverageTrackedDistance(
    previous,
    sample,
    TORSO_TRACKED_KEYS,
  );
  const movementDistance =
    upperBodyDistance * 0.35 + lowerBodyDistance * 0.45 + torsoDistance * 0.2;
  const movementSpeed = movementDistance / elapsedSeconds;
  const movementAmplitude = getTrackedAmplitude(
    fullBodyMotionHistory,
    FULL_BODY_TRACKED_KEYS,
  );
  const nextActiveMs =
    movementDistance >= FULL_BODY_ACTIVE_DISTANCE_THRESHOLD
      ? fullBodyMotionFeatures.value.activeFullBodyMs + elapsedSeconds * 1000
      : fullBodyMotionFeatures.value.activeFullBodyMs;

  fullBodyMotionFeatures.value = {
    movementDistance,
    upperBodyDistance,
    lowerBodyDistance,
    torsoDistance,
    movementSpeed,
    movementAmplitude,
    activeFullBodyMs: nextActiveMs,
  };
  updateFullBodyActivityClassification(sample, previous);
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
    updateJumpingJackCounter(nextState);
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

function drawHandsOverlay(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  if (!latestHands.length) {
    return;
  }

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

async function runPoseInference(now: number) {
  const video = videoRef.value;

  updateJumpingJackRuntime(now);

  if (
    (!poseLandmarker && !handLandmarker) ||
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
    if (handLandmarker) {
      const handResult = handLandmarker.detectForVideo(video, now);
      latestHands = handResult.landmarks as LandmarkPoint[][];
      handCount.value = latestHands.length;
    } else {
      latestHands = [];
      handCount.value = 0;
    }

    if (!poseLandmarker) {
      updateInferenceFps(now);
      return;
    }

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
      resetFullBodyMotionFeatures();
      resetDepthDebug("未检测到完整人体");
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
    updateFullBodyMotionFeatures(smoothedLandmarks, now);
    void updateDepthDebug(smoothedLandmarks, now);
    if (smoothedLandmarks) sendLandmarksToBackend(smoothedLandmarks, now);
  } catch (error) {
    poseStatus.value = "missing";
    rawLandmarkState.value =
      error instanceof Error ? error.message : "姿态推理失败";
    resetJumpingJackAnalysis("姿态推理失败");
    resetFullBodyMotionFeatures();
    resetDepthDebug("姿态推理失败");
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
  drawHandsOverlay(context, canvas);

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
  void loadHandLandmarker();
  connectBackendWs();
});

onBeforeUnmount(() => {
  stopCamera();
  disposePoseLandmarker();
  disposeHandLandmarker();
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
            <span>开合跳</span>
            <strong>{{ jumpingJackState }}</strong>
            <small>{{ jumpingJackRuntimeStatus }} / {{ jumpingJackCounterPhase }}</small>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button type="button" @click="startJumpingJackRecording">开始</button>
        <button type="button" class="secondary" @click="pauseJumpingJackRecording">暂停</button>
        <button type="button" class="secondary" @click="resetJumpingJackRecording">重置</button>
      </div>
    </section>

    <aside class="side-panel" aria-label="实时指标">
      <!-- Backend action recognition panel -->
      <section class="backend-action-panel" aria-label="后端动作识别">
        <div class="backend-action-header">
          <div>
            <p class="eyebrow">后端算法层</p>
            <h2>动作识别</h2>
          </div>
          <span class="status-pill" :class="`status-${backendWsStatus === 'connected' ? 'ready' : backendWsStatus === 'connecting' ? 'loading' : 'error'}`">
            {{ backendWsStatus }}
          </span>
        </div>

        <div v-if="backendActions.length" class="backend-action-grid">
          <section
            v-for="action in backendActions"
            :key="action.name"
            class="backend-action-card"
          >
            <span>{{ action.name }}</span>
            <strong class="backend-count">{{ action.count }}</strong>
            <small>{{ action.stage }} · {{ action.score != null ? action.score.toFixed(2) : '--' }}</small>
          </section>
        </div>
        <p v-else class="backend-action-empty">
          {{ backendWsStatus === 'connected' ? '等待第一帧…' : '后端未连接' }}
        </p>

        <p v-if="backendLatencyMs != null" class="backend-latency">
          后端延迟 {{ backendLatencyMs.toFixed(1) }} ms
        </p>
      </section>

      <section class="depth-focus-panel" aria-label="Depth 测试参数">
        <div class="depth-focus-header">
          <div>
            <p class="eyebrow">RGB-D Depth</p>
            <h2>深度调试</h2>
          </div>
          <span class="status-pill" :class="`status-${depthServiceStatus}`">
            {{ depthServiceStatus }}
          </span>
        </div>

        <section class="depth-hero-card">
          <span>Depth 摘要</span>
          <strong>{{ depthDebugSummary }}</strong>
          <small>z 轴差值只表示摄像头纵深方向差，不等于手臂长度。</small>
        </section>

        <div class="depth-primary-grid">
          <section
            v-for="metric in depthPrimaryMetrics"
            :key="metric.label"
            class="depth-metric-card"
          >
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.detail }}</small>
          </section>
        </div>

        <div class="depth-detail-grid">
          <section
            v-for="metric in depthDetailMetrics"
            :key="metric.label"
            class="depth-detail-card"
          >
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.detail }}</small>
          </section>
        </div>

        <div class="depth-action-grid">
          <section
            v-for="metric in depthActionEvidenceMetrics"
            :key="metric.label"
            class="depth-action-card"
          >
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.detail }}</small>
          </section>
        </div>

        <section class="depth-source-card">
          <span>身体基准来源</span>
          <strong>{{ bodyDepthSource }}</strong>
        </section>
      </section>

      <details class="debug-panel">
        <summary>
          <span>运动估算指标</span>
          <strong class="debug-status status-ready">折叠</strong>
        </summary>
        <div class="debug-body">
          <label class="weight-field">
            <span>体重 kg</span>
            <input v-model.number="weightKg" type="number" min="20" max="200" />
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
        </div>
      </details>

      <details class="debug-panel">
        <summary>
          <span>视觉模型</span>
          <strong :class="`debug-status status-${modelStatus}`">
            {{ modelStatus }}
          </strong>
        </summary>
        <div class="debug-body">
          <p>{{ modelMessage }}</p>
          <p>{{ handModelMessage }}</p>
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
