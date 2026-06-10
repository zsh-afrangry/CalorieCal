<script setup lang="ts">
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type CameraStatus = "idle" | "loading" | "ready" | "error";
type ModelStatus = "idle" | "loading" | "ready" | "error";
type PoseStatus = "idle" | "detected" | "missing";
type JumpingJackState = "unknown" | "closed" | "open";
type FullBodyIntensityLevel = "静止" | "低强度" | "中强度" | "高强度";
type ConfidenceLevel = "高" | "中" | "低";
type WaveCompositeDirection = -1 | 0 | 1;
type PunchCompositePhase = "idle" | "extended" | "recovering";
type HighKneeCompositeSide = "unknown" | "left" | "right";
type SquatCompositePhase = "idle" | "down" | "rising";

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

type DemoCalorieSample = {
  time: number;
  calories: number;
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

type DemoMetric = {
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "active" | "good" | "warning" | "danger";
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
  error?: string;
  points?: DepthLookupResult[];
};

type DepthServiceStatus = "idle" | "ready" | "error";
type DepthQualityState = "unavailable" | "ready" | "unstable" | "background";

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
const DEPTH_REQUEST_INTERVAL_MS = 250;
const DEPTH_FRONT_BACK_THRESHOLD_MM = 150;
const DEPTH_ACTION_MAX_SCORE_OFFSET_MM = 800;
const DEPTH_MAX_REASONABLE_OFFSET_MM = 2000;
const DEPTH_BASELINE_HOLD_MS = 500;
const DEPTH_BASELINE_JUMP_MM = 900;
const DEMO_WEIGHT_KG = 60;
const DEMO_CALORIE_WINDOW_MS = 1000;
const DEMO_CALORIE_MAX_DELTA_MS = 250;
const MIN_LANDMARK_VISIBILITY = 0.5;
const REQUIRED_LANDMARK_INDICES = [11, 12, 15, 16, 23, 24, 27, 28];
const DISPLAY_FACE_LANDMARK_INDICES = [0, 3, 6, 9, 10];
const MIN_SHOULDER_WIDTH = 0.04;
const FOOT_OPEN_SHOULDER_RATIO = 1.15;
const FOOT_CLOSED_SHOULDER_RATIO = 0.85;
const WRIST_OPEN_SHOULDER_RATIO = 1.15;
const WRIST_CLOSED_SHOULDER_RATIO = 1.0;
const ARM_OPEN_MAX_TORSO_RATIO = 0.25;
const ARM_CLOSED_MIN_TORSO_RATIO = 0.38;
const ACTION_DEBOUNCE_FRAMES = 3;
const COMPOSITE_ACTION_HOLD_MS = 900;
const FULL_BODY_COMPOSITE_MIN_ACTION_SCORE = 55;
const FULL_BODY_COMPOSITE_MIN_UPPER_DISTANCE = 0.018;
const FULL_BODY_COMPOSITE_MIN_LOWER_DISTANCE = 0.025;
const REST_COMPOSITE_ENTER_HOLD_MS = 600;
const REST_COMPOSITE_EXIT_HOLD_MS = 280;
const REST_COMPOSITE_MAX_ACTIVITY_SCORE = 28;
const REST_COMPOSITE_MAX_MOVEMENT_DISTANCE = 0.022;
const REST_COMPOSITE_MAX_UPPER_DISTANCE = 0.018;
const REST_COMPOSITE_MAX_LOWER_DISTANCE = 0.018;
const REST_COMPOSITE_EXIT_ACTIVITY_SCORE = 38;
const REST_COMPOSITE_EXIT_ACTION_SCORE = 58;
const REST_COMPOSITE_EXIT_UPPER_DISTANCE = 0.026;
const REST_COMPOSITE_EXIT_LOWER_DISTANCE = 0.03;
const REST_COMPOSITE_EXIT_TORSO_DISTANCE = 0.02;
const WAVE_COMPOSITE_MIN_SWING_SCORE = 50;
const WAVE_COMPOSITE_MIN_UPPER_DISTANCE = 0.012;
const WAVE_COMPOSITE_DIRECTION_RATIO = 0.028;
const WAVE_COMPOSITE_TURN_WINDOW_MS = 1300;
const WAVE_COMPOSITE_HOLD_MS = 850;
const WAVE_COMPOSITE_DEPTH_CONFLICT_SCORE = 72;
const WAVE_COMPOSITE_EXTENSION_CONFLICT_SCORE = 68;
const PUNCH_COMPOSITE_MIN_FORWARD_SCORE = 55;
const PUNCH_COMPOSITE_MIN_EXTENSION_SCORE = 48;
const PUNCH_COMPOSITE_MIN_FIST_SCORE = 45;
const PUNCH_COMPOSITE_MIN_BODY_LEVEL_SCORE = 52;
const PUNCH_COMPOSITE_MIN_UPPER_DISTANCE = 0.012;
const PUNCH_COMPOSITE_RECOVERY_SCORE = 45;
const PUNCH_COMPOSITE_HOLD_MS = 900;
const JUMPING_JACK_BASIC_MIN_ARM_SCORE = 55;
const JUMPING_JACK_BASIC_MIN_FOOT_SCORE = 45;
const JUMPING_JACK_BASIC_MIN_UPPER_DISTANCE = 0.018;
const JUMPING_JACK_BASIC_MIN_LOWER_DISTANCE = 0.018;
const HIGH_KNEE_COMPOSITE_MIN_SCORE = 58;
const HIGH_KNEE_COMPOSITE_MIN_LOWER_DISTANCE = 0.012;
const HIGH_KNEE_COMPOSITE_ALTERNATE_WINDOW_MS = 1800;
const HIGH_KNEE_COMPOSITE_HOLD_MS = 950;
const SQUAT_COMPOSITE_MIN_DOWN_SCORE = 55;
const SQUAT_COMPOSITE_MIN_UP_SCORE = 45;
const SQUAT_COMPOSITE_MIN_LOWER_DISTANCE = 0.01;
const SQUAT_COMPOSITE_HOLD_MS = 1100;
const UPPER_BODY_BASIC_ACTION_LABELS = [
  "左手上抬",
  "右手上抬",
  "双臂上抬",
  "手臂回落",
  "手向前移动",
  "手向后移动",
  "手水平摆动",
  "手臂伸展",
  "手臂弯曲",
];
const LOWER_BODY_BASIC_ACTION_LABELS = [
  "双脚打开",
  "双脚合拢",
  "左抬腿",
  "右抬腿",
  "膝盖上抬",
  "下蹲",
  "起身",
];
const TORSO_OR_HEAD_BASIC_ACTION_LABELS = [
  "身体前倾",
  "身体后仰",
  "身体左倾",
  "身体右倾",
  "身体左转",
  "身体右转",
  "点头",
  "摇头",
  "左转头",
  "右转头",
];
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
const LEFT_BODY_SEGMENTS = [
  [11, 13],
  [13, 15],
  [23, 25],
  [25, 27],
] as const;
const RIGHT_BODY_SEGMENTS = [
  [12, 14],
  [14, 16],
  [24, 26],
  [26, 28],
] as const;
const CENTER_BODY_SEGMENTS = [
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
] as const;
const FOOT_SEGMENTS = [
  [27, 29],
  [29, 31],
  [28, 30],
  [30, 32],
] as const;
const EMPHASIZED_JOINT_INDICES = new Set([11, 12, 23, 24, 25, 26, 27, 28]);
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
const depthServiceStatus = ref<DepthServiceStatus>("idle");
const depthServiceMessage = ref("当前使用 2D 估算");
const depthFrameSize = ref("未知");
const leftWristDepthOffsetMm = ref<number | null>(null);
const rightWristDepthOffsetMm = ref<number | null>(null);
const depthQualityState = ref<DepthQualityState>("unavailable");
const depthQualityReason = ref("当前使用 2D 估算");
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
const demoElapsedMs = ref(0);
const demoWindowCalories = ref(0);
const fullBodyActivityScore = ref(0);
const fullBodyIntensityLevel = ref<FullBodyIntensityLevel>("静止");
const fullBodyBasicActions = ref<BasicActionCandidate[]>([]);
const feetState = ref<JumpingJackState>("unknown");
const armsState = ref<JumpingJackState>("unknown");
const jumpingJackState = ref<JumpingJackState>("unknown");
const compositeActionLabel = ref("未识别组合动作");
const compositeActionDetail = ref("等待组合动作证据");
const lastCompositeEvidenceAt = ref(0);

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let handLandmarker: HandLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let lastDemoElapsedAt = 0;
let lastDepthRequestAt = 0;
let isDepthRequesting = false;
let lastStableBodyDepthMm: number | null = null;
let lastStableBodyDepthAt = 0;
let smoothedLandmarks: LandmarkPoint[] | null = null;
let latestHands: LandmarkPoint[][] = [];
let pendingJumpingJackState: JumpingJackState = "unknown";
let pendingJumpingJackFrames = 0;
let restCompositeActive = false;
let restCompositeCandidateStartedAt = 0;
let restCompositeExitStartedAt = 0;
let waveCompositeDirection: WaveCompositeDirection = 0;
let waveCompositeTurnTimes: number[] = [];
let lastWaveCompositeEvidenceAt = 0;
let lastWaveCompositeSampleAt = 0;
let punchCompositePhase: PunchCompositePhase = "idle";
let lastPunchCompositeEvidenceAt = 0;
let highKneeCompositeSide: HighKneeCompositeSide = "unknown";
let highKneeCompositeSideChanges: number[] = [];
let lastHighKneeCompositeEvidenceAt = 0;
let squatCompositePhase: SquatCompositePhase = "idle";
let lastSquatCompositeEvidenceAt = 0;
const landmarkHistory: LandmarkPoint[][] = [];
const fullBodyMotionHistory: FullBodyMotionSample[] = [];
const demoCalorieHistory: DemoCalorieSample[] = [];

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
  return demoWindowCalories.value;
});

const fullBodyIntensityTone = computed<DemoMetric["tone"]>(() => {
  if (fullBodyIntensityLevel.value === "高强度") {
    return "danger";
  }

  if (fullBodyIntensityLevel.value === "中强度") {
    return "warning";
  }

  if (fullBodyIntensityLevel.value === "低强度") {
    return "active";
  }

  return "neutral";
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

  const hasMediumBaseQuality =
    landmarkCompleteness.value >= 60 && fullBodyTrackedCompleteness.value >= 60;

  if (!hasMediumBaseQuality) {
    return "低";
  }

  if (
    landmarkCompleteness.value >= 85 &&
    fullBodyTrackedCompleteness.value >= 80 &&
    fullBodyMotionHistory.length >= 3
  ) {
    return depthServiceStatus.value === "ready" && depthQualityState.value !== "ready"
      ? "中"
      : "高";
  }

  return "中";
});

const fullBodyConfidenceSummary = computed(() => {
  const reasons: string[] = [];

  if (poseStatus.value !== "detected") {
    reasons.push("未稳定检测到完整人体");
  }

  if (modelStatus.value !== "ready") {
    reasons.push("姿态模型尚未就绪");
  }

  if (landmarkCompleteness.value < 60) {
    reasons.push("关键点完整率较低");
  }

  if (fullBodyTrackedCompleteness.value < 60) {
    reasons.push("全身跟踪关键点不足");
  }

  if (
    fullBodyMotionFeatures.value.activeFullBodyMs < 1000 &&
    fullBodyActivityScore.value > 0
  ) {
    reasons.push("有效活动时间较短");
  }

  if (depthServiceStatus.value === "ready" && depthQualityState.value === "ready") {
    reasons.push("前后方向证据已接入");
  } else if (depthServiceStatus.value === "ready") {
    reasons.push(depthQualityReason.value);
  } else {
    reasons.push("当前使用 2D 估算");
  }

  if (!reasons.length) {
    reasons.push("人体入镜和运动信号稳定");
  }

  return reasons.slice(0, 2).join("；");
});

const fullBodyConfidenceTone = computed<DemoMetric["tone"]>(() => {
  if (fullBodyConfidence.value === "高") {
    return "good";
  }

  if (fullBodyConfidence.value === "中") {
    return "warning";
  }

  return "danger";
});

const depthStatusTone = computed<DemoMetric["tone"]>(() =>
  depthServiceStatus.value === "ready" ? "good" : "neutral",
);

const depthQualityPrompt = computed(() => {
  if (depthServiceStatus.value !== "ready") {
    return "";
  }

  if (depthQualityState.value === "unstable") {
    return "请正对摄像头，估算更稳定。";
  }

  if (depthQualityState.value === "background") {
    return "请站到画面中央，估算更稳定。";
  }

  return "";
});

const cameraStatusLabel = computed(() => {
  if (cameraStatus.value === "loading") {
    return "摄像头启动中";
  }

  if (cameraStatus.value === "ready") {
    return "摄像头已就绪";
  }

  if (cameraStatus.value === "error") {
    return "摄像头未就绪";
  }

  return "等待摄像头";
});

const modelStatusLabel = computed(() =>
  modelStatus.value === "ready" ? "人体识别已就绪" : "人体识别准备中",
);

const handStatusLabel = computed(() =>
  handModelStatus.value === "ready" ? "手部识别已就绪" : "手部识别准备中",
);

const depthStatusLabel = computed(() =>
  depthServiceStatus.value === "ready" ? "RGB-D 已接入" : "当前使用 2D 估算",
);

const cameraOverlayTitle = computed(() => {
  if (cameraStatus.value === "loading") {
    return "正在启动摄像头";
  }

  if (cameraStatus.value === "error") {
    return "摄像头未就绪";
  }

  return "准备开始识别";
});

const demoCenterPrompt = computed(() => {
  if (cameraStatus.value !== "ready") {
    return null;
  }

  if (modelStatus.value !== "ready") {
    return {
      title: "正在准备识别",
      detail: "请稍候片刻",
    };
  }

  if (poseStatus.value !== "detected") {
    return {
      title: "请站到画面中央",
      detail: "让头部、躯干和四肢尽量完整入镜",
    };
  }

  return null;
});

const systemStatusText = computed(() => {
  if (cameraStatus.value !== "ready") {
    return cameraMessage.value;
  }

  if (modelStatus.value !== "ready") {
    return modelMessage.value;
  }

  if (poseStatus.value === "missing") {
    return "请站到画面中央。";
  }

  if (fullBodyConfidence.value === "低") {
    return "请正对摄像头，保持身体完整入镜。";
  }

  if (depthQualityPrompt.value) {
    return depthQualityPrompt.value;
  }

  if (poseStatus.value === "detected") {
    return "正在实时估算全身运动强度。";
  }

  return "等待人体入镜。";
});

const motionSuggestion = computed(() => {
  if (poseStatus.value !== "detected") {
    return "人体进入画面后开始估算";
  }

  if (fullBodyConfidence.value === "低") {
    return "请让肩、髋、膝、踝尽量完整入镜";
  }

  if (depthQualityPrompt.value) {
    return depthQualityPrompt.value;
  }

  if (fullBodyIntensityLevel.value === "静止") {
    return "当前运动量较低";
  }

  if (fullBodyIntensityLevel.value === "高强度") {
    return "当前全身参与度较高";
  }

  return "当前运动信号正常";
});

const primaryMetrics = computed<DemoMetric[]>(() => [
  {
    label: "当前基本动作",
    value: topFullBodyBasicActionsText.value,
    detail: "可同时存在多个基础动作证据",
    tone: fullBodyBasicActions.value.length ? "active" : "neutral",
  },
  {
    label: "当前强度",
    value: fullBodyIntensityLevel.value,
    detail: `活动分数 ${fullBodyActivityScore.value}`,
    tone: fullBodyIntensityTone.value,
  },
  {
    label: "RGB-D 状态",
    value: depthServiceStatus.value === "ready" ? "已接入" : "2D 估算",
    detail:
      depthServiceStatus.value === "ready"
        ? `${depthServiceMessage.value} / ${depthFrameSize.value}`
        : depthServiceMessage.value,
    tone: depthStatusTone.value,
  },
]);

const motionMetrics = computed<DemoMetric[]>(() => [
  {
    label: "全身位移",
    value: fullBodyMotionFeatures.value.movementDistance.toFixed(3),
    detail: "最近帧归一化位移",
  },
  {
    label: "全身速度",
    value: fullBodyMotionFeatures.value.movementSpeed.toFixed(2),
    detail: "归一化 / 秒",
  },
  {
    label: "全身幅度",
    value: fullBodyMotionFeatures.value.movementAmplitude.toFixed(3),
    detail: "1.2 秒窗口",
  },
  {
    label: "有效活动",
    value: formatElapsedTime(fullBodyMotionFeatures.value.activeFullBodyMs),
    detail: "超过阈值后累计",
  },
  {
    label: "识别质量",
    value: fullBodyConfidence.value,
    detail: fullBodyConfidenceSummary.value,
    tone: fullBodyConfidenceTone.value,
  },
  {
    label: "全身关键点",
    value: fullBodyJointState.value,
    detail: `完整率 ${landmarkCompleteness.value.toFixed(0)}%`,
  },
]);

const depthActionMetrics = computed<DemoMetric[]>(() => [
  getDepthActionEvidence("左手前后", leftWristDepthOffsetMm.value),
  getDepthActionEvidence("右手前后", rightWristDepthOffsetMm.value),
]);

function getCameraErrorMessage(error: unknown) {
  if (!(error instanceof DOMException)) {
    return "请检查摄像头连接。";
  }

  if (error.name === "NotAllowedError") {
    return "请允许浏览器使用摄像头。";
  }

  if (error.name === "NotFoundError") {
    return "未找到摄像头。";
  }

  if (error.name === "NotReadableError") {
    return "摄像头可能被其他应用占用。";
  }

  return "摄像头暂时不可用。";
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
  landmarkHistory.length = 0;
  smoothedLandmarks = null;
  latestHands = [];
  handCount.value = 0;
  demoElapsedMs.value = 0;
  demoWindowCalories.value = 0;
  lastDemoElapsedAt = 0;
  demoCalorieHistory.length = 0;
  resetJumpingJackAnalysis("等待完整关键点");
  resetFullBodyMotionFeatures();
  resetDepthServiceStatus();
}

function pauseDemoElapsed() {
  lastDemoElapsedAt = 0;
  demoWindowCalories.value = 0;
  demoCalorieHistory.length = 0;
}

function updateDemoElapsed(now: number) {
  if (lastDemoElapsedAt > 0) {
    const elapsedMs = Math.min(now - lastDemoElapsedAt, DEMO_CALORIE_MAX_DELTA_MS);
    const elapsedMinutes = elapsedMs / 60_000;
    const calories =
      ((currentFullBodyMet.value * 3.5 * DEMO_WEIGHT_KG * elapsedMinutes) / 200) * 1000;

    demoElapsedMs.value += elapsedMs;
    demoCalorieHistory.push({ time: now, calories });

    while (
      demoCalorieHistory.length &&
      now - demoCalorieHistory[0].time > DEMO_CALORIE_WINDOW_MS
    ) {
      demoCalorieHistory.shift();
    }

    demoWindowCalories.value = demoCalorieHistory.reduce(
      (total, sample) => total + sample.calories,
      0,
    );
  }

  lastDemoElapsedAt = now;
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
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });

    handModelStatus.value = "ready";
    handModelMessage.value = "手部模型已就绪";
  } catch (error) {
    handModelStatus.value = "error";
    handModelMessage.value =
      error instanceof Error ? error.message : "手部模型加载失败";
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
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    modelStatus.value = "ready";
    modelMessage.value = "姿态模型已就绪";
  } catch (error) {
    modelStatus.value = "error";
    modelMessage.value =
      error instanceof Error ? error.message : "姿态模型加载失败";
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

  while (landmarkHistory.length > SMOOTHING_WINDOW_SIZE) {
    landmarkHistory.shift();
  }

  const landmarkCount = landmarks.length;

  smoothedLandmarks = Array.from({ length: landmarkCount }, (_, index) => {
    const points = landmarkHistory
      .map((sample) => sample[index])
      .filter((point): point is LandmarkPoint => Boolean(point));
    const total = points.reduce<{ x: number; y: number; z: number; visibility: number }>(
      (acc, point) => ({
        x: acc.x + point.x,
        y: acc.y + point.y,
        z: acc.z + (point.z ?? 0),
        visibility: acc.visibility + (point.visibility ?? 1),
      }),
      { x: 0, y: 0, z: 0, visibility: 0 },
    );

    return {
      x: total.x / points.length,
      y: total.y / points.length,
      z: total.z / points.length,
      visibility: total.visibility / points.length,
    };
  });
}

function updateLandmarkCompleteness(landmarks: LandmarkPoint[]) {
  const visibleCount = REQUIRED_LANDMARK_INDICES.filter((index) =>
    isVisibleLandmark(landmarks[index]),
  ).length;

  landmarkCompleteness.value =
    (visibleCount / REQUIRED_LANDMARK_INDICES.length) * 100;
}

function updateInferenceFps(now: number) {
  inferenceFrameCount += 1;

  if (!lastFpsSampleAt) {
    lastFpsSampleAt = now;
    return;
  }

  const elapsedMs = now - lastFpsSampleAt;

  if (elapsedMs >= 1000) {
    inferenceFps.value = (inferenceFrameCount * 1000) / elapsedMs;
    inferenceFrameCount = 0;
    lastFpsSampleAt = now;
  }
}

function resetJumpingJackAnalysis(reason: string) {
  feetState.value = "unknown";
  armsState.value = "unknown";
  jumpingJackState.value = "unknown";
  pendingJumpingJackState = "unknown";
  pendingJumpingJackFrames = 0;

  if (poseStatus.value === "detected") {
    updateCompositeActionDisplay(performance.now(), reason);
    return;
  }

  compositeActionLabel.value = "未识别组合动作";
  compositeActionDetail.value = reason;
}

function isVisibleLandmark(landmark: LandmarkPoint | undefined) {
  return Boolean(landmark) && (landmark?.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY;
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

function getPointDistance(left: LandmarkPoint, right: LandmarkPoint) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function resetDepthServiceStatus(reason = "当前使用 2D 估算") {
  depthServiceStatus.value = "idle";
  depthServiceMessage.value = reason;
  depthFrameSize.value = "未知";
  leftWristDepthOffsetMm.value = null;
  rightWristDepthOffsetMm.value = null;
  depthQualityState.value = "unavailable";
  depthQualityReason.value = reason;
  lastStableBodyDepthMm = null;
  lastStableBodyDepthAt = 0;
}

function getDepthActionEvidence(label: string, offsetMm: number | null): DemoMetric {
  if (offsetMm === null || depthServiceStatus.value !== "ready") {
    return {
      label,
      value: "2D 估算",
      detail: "当前使用 2D 估算",
      tone: "neutral",
    };
  }

  if (Math.abs(offsetMm) >= DEPTH_MAX_REASONABLE_OFFSET_MM) {
    return {
      label,
      value: "暂不采信",
      detail: "depth 采样疑似背景",
      tone: "warning",
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
      detail: "RGB-D 前后方向证据",
      tone: "active",
    };
  }

  if (offsetMm >= DEPTH_FRONT_BACK_THRESHOLD_MM) {
    return {
      label,
      value: `手后移 ${score}`,
      detail: "RGB-D 前后方向证据",
      tone: "active",
    };
  }

  return {
    label,
    value: "接近平面",
    detail: "前后位移低于阈值",
    tone: "neutral",
  };
}

function getValidDepth(points: DepthLookupResult[], id: string) {
  const point = points.find((item) => item.id === id);

  return point?.valid && point.medianDepthMm > 0 ? point.medianDepthMm : null;
}

function getBodyBaselineDepth(points: DepthLookupResult[]) {
  const depths = [
    "torsoCenter",
    "shoulderCenter",
    "hipCenter",
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
  ]
    .map((id) => getValidDepth(points, id))
    .filter((depth): depth is number => depth !== null)
    .sort((a, b) => a - b);

  if (!depths.length) {
    return null;
  }

  const middle = Math.floor(depths.length / 2);

  return depths.length % 2 === 0
    ? (depths[middle - 1] + depths[middle]) / 2
    : depths[middle];
}

function getEffectiveBodyDepth(bodyDepth: number | null, now: number) {
  if (bodyDepth === null) {
    const canHold =
      lastStableBodyDepthMm !== null &&
      now - lastStableBodyDepthAt <= DEPTH_BASELINE_HOLD_MS;

    return {
      depth: canHold ? lastStableBodyDepthMm : null,
      state: canHold ? "held" : "missing",
    };
  }

  if (
    lastStableBodyDepthMm !== null &&
    Math.abs(bodyDepth - lastStableBodyDepthMm) >= DEPTH_BASELINE_JUMP_MM
  ) {
    const canHold = now - lastStableBodyDepthAt <= DEPTH_BASELINE_HOLD_MS;

    return {
      depth: canHold ? lastStableBodyDepthMm : null,
      state: canHold ? "held" : "jump",
    };
  }

  lastStableBodyDepthMm = bodyDepth;
  lastStableBodyDepthAt = now;

  return {
    depth: bodyDepth,
    state: "stable",
  };
}

function getTorsoReferenceVisibleCount(landmarks: LandmarkPoint[]) {
  return (["leftShoulder", "rightShoulder", "leftHip", "rightHip"] as const).filter(
    (key) => isVisibleLandmark(landmarks[LANDMARK_INDEX[key]]),
  ).length;
}

function getWristDepth(points: DepthLookupResult[], id: string) {
  const point = points.find((item) => item.id === id);

  if (!point?.valid) {
    return null;
  }

  if (point.foregroundDepthMm && point.foregroundDepthMm > 0) {
    return point.foregroundDepthMm;
  }

  return point.medianDepthMm > 0 ? point.medianDepthMm : null;
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

async function updateDepthServiceStatus(
  landmarks: LandmarkPoint[] | null,
  now: number,
) {
  if (!landmarks?.length) {
    resetDepthServiceStatus("等待人体关键点 / 使用 2D 估算");
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
    getAverageDepthPoint(landmarks, "shoulderCenter", [
      "leftShoulder",
      "rightShoulder",
    ]),
    getAverageDepthPoint(landmarks, "hipCenter", ["leftHip", "rightHip"]),
    getAverageDepthPoint(landmarks, "torsoCenter", [
      "leftShoulder",
      "rightShoulder",
      "leftHip",
      "rightHip",
    ]),
  ].filter((point): point is DepthLookupPoint => Boolean(point));

  if (!points.length) {
    resetDepthServiceStatus("缺少 depth 查询点 / 使用 2D 估算");
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

    if (!response.ok || !data.ok) {
      throw new Error(data.error || `depth server ${response.status}`);
    }

    depthServiceStatus.value = "ready";
    depthServiceMessage.value = "RGB-D 已接入";
    depthFrameSize.value =
      data.width && data.height ? `${data.width}x${data.height}` : "未知";
    const depthPoints = data.points ?? [];
    const torsoReferenceVisibleCount = getTorsoReferenceVisibleCount(landmarks);
    const bodyDepth = getBodyBaselineDepth(depthPoints);
    const effectiveBodyDepth = getEffectiveBodyDepth(bodyDepth, now);
    const leftWristDepth = getWristDepth(depthPoints, "leftWrist");
    const rightWristDepth = getWristDepth(depthPoints, "rightWrist");
    leftWristDepthOffsetMm.value =
      effectiveBodyDepth.depth !== null && leftWristDepth !== null
        ? leftWristDepth - effectiveBodyDepth.depth
        : null;
    rightWristDepthOffsetMm.value =
      effectiveBodyDepth.depth !== null && rightWristDepth !== null
        ? rightWristDepth - effectiveBodyDepth.depth
        : null;

    if (
      torsoReferenceVisibleCount < 3 ||
      effectiveBodyDepth.depth === null ||
      (leftWristDepth === null && rightWristDepth === null)
    ) {
      depthQualityState.value = "unstable";
      depthQualityReason.value =
        torsoReferenceVisibleCount < 3
          ? "躯干参考不稳定"
          : "身体基准不稳定";
    } else if (effectiveBodyDepth.state === "held") {
      depthQualityState.value = "unstable";
      depthQualityReason.value = "短暂沿用身体基准";
    } else if (effectiveBodyDepth.state === "jump") {
      depthQualityState.value = "unstable";
      depthQualityReason.value = "身体基准跳变过大";
    } else if (
      [leftWristDepthOffsetMm.value, rightWristDepthOffsetMm.value].some(
        (offset) => offset !== null && Math.abs(offset) >= DEPTH_MAX_REASONABLE_OFFSET_MM,
      )
    ) {
      depthQualityState.value = "background";
      depthQualityReason.value = "depth 采样疑似背景";
    } else {
      depthQualityState.value = "ready";
      depthQualityReason.value = "前后方向证据已接入";
    }
  } catch {
    depthServiceStatus.value = "error";
    depthServiceMessage.value = "当前使用 2D 估算";
    depthFrameSize.value = "未知";
    leftWristDepthOffsetMm.value = null;
    rightWristDepthOffsetMm.value = null;
    depthQualityState.value = "unavailable";
    depthQualityReason.value = "当前使用 2D 估算";
  } finally {
    isDepthRequesting = false;
  }
}

function getVisibleTrackedPoint(
  landmarks: LandmarkPoint[],
  key: FullBodyTrackedKey,
) {
  const point = landmarks[LANDMARK_INDEX[key]];

  return isVisibleLandmark(point) ? point : null;
}

function getFullBodyMotionSample(
  landmarks: LandmarkPoint[],
  now: number,
): FullBodyMotionSample {
  const points = SAMPLE_TRACKED_KEYS.reduce<
    Partial<Record<FullBodyTrackedKey, LandmarkPoint>>
  >((acc, key) => {
    const point = getVisibleTrackedPoint(landmarks, key);

    if (point) {
      acc[key] = point;
    }

    return acc;
  }, {});

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

  return distances.reduce((total, distance) => total + distance, 0) / distances.length;
}

function getTrackedAmplitude(
  history: FullBodyMotionSample[],
  keys: FullBodyTrackedKey[],
) {
  const ranges = keys
    .map((key) => {
      const points = history
        .map((sample) => sample.points[key])
        .filter((point): point is LandmarkPoint => Boolean(point));

      if (points.length < 2) {
        return null;
      }

      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);

      return Math.hypot(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
      );
    })
    .filter((range): range is number => range !== null);

  if (!ranges.length) {
    return 0;
  }

  return ranges.reduce((total, range) => total + range, 0) / ranges.length;
}

function clampScore(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 100));
}

function scoreFromRange(value: number, min: number, max: number) {
  if (value <= min) {
    return 0;
  }

  if (value >= max) {
    return 100;
  }

  return ((value - min) / (max - min)) * 100;
}

function scoreFromInverseRange(value: number, min: number, max: number) {
  if (value <= min) {
    return 100;
  }

  if (value >= max) {
    return 0;
  }

  return ((max - value) / (max - min)) * 100;
}

function getFullBodyIntensityLevel(score: number): FullBodyIntensityLevel {
  if (score >= 70) {
    return "高强度";
  }

  if (score >= 40) {
    return "中强度";
  }

  if (score >= 15) {
    return "低强度";
  }

  return "静止";
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

  return points.reduce((total, point) => total + point.y, 0) / points.length;
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

  return points.reduce((total, point) => total + point.x, 0) / points.length;
}

function getTrackedDistance(
  sample: FullBodyMotionSample,
  leftKey: FullBodyTrackedKey,
  rightKey: FullBodyTrackedKey,
) {
  const left = sample.points[leftKey];
  const right = sample.points[rightKey];

  return left && right ? getPointDistance(left, right) : null;
}

function getTorsoHeight(sample: FullBodyMotionSample) {
  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const hipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);

  if (shoulderY === null || hipY === null) {
    return 0;
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

  if (!hip || !knee || torsoHeight <= 0) {
    return 0;
  }

  const kneeGapRatio = Math.abs(knee.y - hip.y) / torsoHeight;

  return scoreFromInverseRange(kneeGapRatio, 0.18, 0.62);
}

function getHipVerticalDelta(
  sample: FullBodyMotionSample,
  previous: FullBodyMotionSample | null,
) {
  if (!previous) {
    return 0;
  }

  const currentHipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);
  const previousHipY = getAverageTrackedY(previous, ["leftHip", "rightHip"]);

  if (currentHipY === null || previousHipY === null) {
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

  if (torsoHeight <= 0 || hipY === null || kneeY === null || ankleY === null) {
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

  if (!shoulder || !wrist || torsoHeight <= 0) {
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

  if (!previous || torsoHeight <= 0) {
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

  const horizontalRatios = deltas.map((delta) => Math.abs(delta.x) / shoulderWidth);
  const maxHorizontalRatio = Math.max(...horizontalRatios);
  const averageHorizontalRatio =
    horizontalRatios.reduce((total, ratio) => total + ratio, 0) / horizontalRatios.length;

  return clampScore(
    scoreFromRange(maxHorizontalRatio, 0.055, 0.26) * 0.72 +
      scoreFromRange(averageHorizontalRatio, 0.05, 0.24) * 0.28,
  );
}

function getHandStaticFistScore(hand: LandmarkPoint[]) {
  const wrist = hand[0];
  const indexMcp = hand[5];
  const middleMcp = hand[9];
  const ringMcp = hand[13];
  const pinkyMcp = hand[17];

  if (!wrist || !indexMcp || !middleMcp || !ringMcp || !pinkyMcp) {
    return 0;
  }

  const palmWidth = Math.max(getPointDistance(indexMcp, pinkyMcp), 0.015);
  const palmCenter = {
    x: (wrist.x + indexMcp.x + middleMcp.x + ringMcp.x + pinkyMcp.x) / 5,
    y: (wrist.y + indexMcp.y + middleMcp.y + ringMcp.y + pinkyMcp.y) / 5,
  };
  const fingerPairs = [
    [8, 5],
    [12, 9],
    [16, 13],
    [20, 17],
  ] as const;
  const tipToMcpRatios = fingerPairs
    .map(([tipIndex, mcpIndex]) => {
      const tip = hand[tipIndex];
      const mcp = hand[mcpIndex];

      return tip && mcp ? getPointDistance(tip, mcp) / palmWidth : null;
    })
    .filter((ratio): ratio is number => ratio !== null);
  const tipToPalmRatios = [8, 12, 16, 20]
    .map((tipIndex) => {
      const tip = hand[tipIndex];

      return tip ? getPointDistance(tip, palmCenter) / palmWidth : null;
    })
    .filter((ratio): ratio is number => ratio !== null);

  if (!tipToMcpRatios.length || !tipToPalmRatios.length) {
    return 0;
  }

  const averageTipToMcp =
    tipToMcpRatios.reduce((total, ratio) => total + ratio, 0) / tipToMcpRatios.length;
  const averageTipToPalm =
    tipToPalmRatios.reduce((total, ratio) => total + ratio, 0) / tipToPalmRatios.length;

  return clampScore(
    scoreFromInverseRange(averageTipToMcp, 0.45, 0.95) * 0.65 +
      scoreFromInverseRange(averageTipToPalm, 0.75, 1.45) * 0.35,
  );
}

function getPunchFistScore() {
  if (!latestHands.length) {
    return 0;
  }

  return Math.max(...latestHands.map(getHandStaticFistScore));
}

function getPunchBodyLevelScore(sample: FullBodyMotionSample | undefined) {
  if (!sample) {
    return 0;
  }

  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const hipY = getAverageTrackedY(sample, ["leftHip", "rightHip"]);
  const torsoCenterX = getAverageTrackedX(sample, ["leftShoulder", "rightShoulder", "leftHip", "rightHip"]);
  const shoulderWidth = Math.max(
    getTrackedDistance(sample, "leftShoulder", "rightShoulder") ?? 0,
    MIN_SHOULDER_WIDTH,
  );
  const torsoHeight = getTorsoHeight(sample);

  if (shoulderY === null || hipY === null || torsoCenterX === null || torsoHeight <= 0) {
    return 0;
  }

  const scores = (["leftWrist", "rightWrist"] as FullBodyTrackedKey[])
    .map((key) => {
      const wrist = sample.points[key];

      if (!wrist) {
        return null;
      }

      const wristLiftRatio = (shoulderY - wrist.y) / torsoHeight;
      const verticalScore =
        wristLiftRatio >= -0.25 && wristLiftRatio <= 0.28
          ? 100
          : wristLiftRatio > 0.28
            ? scoreFromInverseRange(wristLiftRatio, 0.28, 0.7)
            : scoreFromRange(wristLiftRatio, -0.65, -0.25);
      const lateralRatio = Math.abs(wrist.x - torsoCenterX) / shoulderWidth;
      const lateralScore = scoreFromInverseRange(lateralRatio, 0.95, 1.55);

      return clampScore(verticalScore * 0.78 + lateralScore * 0.22);
    })
    .filter((score): score is number => score !== null);

  return scores.length ? Math.max(...scores) : 0;
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
  const pitchRatio = torsoHeight > 0 ? shoulderHipDelta / torsoHeight : 0;
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
  const torsoHeight = getTorsoHeight(sample);
  const shoulderY = getAverageTrackedY(sample, ["leftShoulder", "rightShoulder"]);
  const wristY = getAverageTrackedY(sample, ["leftWrist", "rightWrist"]);
  const hipDelta = getHipVerticalDelta(sample, previous);
  const shoulderWidth = Math.max(shoulderDistance ?? 0, MIN_SHOULDER_WIDTH);
  const ankleRatio = ankleDistance ? ankleDistance / shoulderWidth : 0;
  const kneeRatio = kneeDistance ? kneeDistance / shoulderWidth : 0;
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
  const wristRatio = wristDistance ? wristDistance / shoulderWidth : 0;
  const wristLiftRatio =
    wristY !== null && shoulderY !== null && torsoHeight > 0
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
      reason: "脚踝距离超过肩宽",
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
        scoreFromRange(wristLiftRatio, 0.05, 0.45),
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
  const activityScore = clampScore(
    distanceScore * 0.25 +
      speedScore * 0.25 +
      amplitudeScore * 0.25 +
      lowerBodyScore * 0.15 +
      landmarkCompleteness.value * 0.1,
  );

  fullBodyActivityScore.value = activityScore;
  fullBodyIntensityLevel.value = getFullBodyIntensityLevel(activityScore);
  fullBodyBasicActions.value = getFullBodyBasicActionCandidates(sample, previous);
}

function resetFullBodyMotionFeatures() {
  fullBodyMotionHistory.length = 0;
  resetRestCompositeState();
  resetMotionCompositeStates();
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

function getTopBasicActionLabels() {
  return fullBodyBasicActions.value.slice(0, 3).map((candidate) => candidate.label);
}

function getBasicActionScore(labels: string[]) {
  return fullBodyBasicActions.value
    .filter((candidate) => labels.includes(candidate.label))
    .reduce((maxScore, candidate) => Math.max(maxScore, candidate.score), 0);
}

function resetRestCompositeState() {
  restCompositeActive = false;
  restCompositeCandidateStartedAt = 0;
  restCompositeExitStartedAt = 0;
}

function updateRestCompositeState(
  now: number,
  isRestCandidate: boolean,
  hasExitEvidence: boolean,
) {
  if (isRestCandidate) {
    restCompositeExitStartedAt = 0;

    if (!restCompositeCandidateStartedAt) {
      restCompositeCandidateStartedAt = now;
    }

    if (
      restCompositeActive ||
      now - restCompositeCandidateStartedAt >= REST_COMPOSITE_ENTER_HOLD_MS
    ) {
      restCompositeActive = true;
      return true;
    }

    return false;
  }

  restCompositeCandidateStartedAt = 0;

  if (!restCompositeActive) {
    return false;
  }

  if (!hasExitEvidence) {
    restCompositeExitStartedAt = 0;
    return true;
  }

  if (!restCompositeExitStartedAt) {
    restCompositeExitStartedAt = now;
    return true;
  }

  if (now - restCompositeExitStartedAt < REST_COMPOSITE_EXIT_HOLD_MS) {
    return true;
  }

  resetRestCompositeState();
  return false;
}

function resetWaveCompositeState() {
  waveCompositeDirection = 0;
  waveCompositeTurnTimes = [];
  lastWaveCompositeEvidenceAt = 0;
  lastWaveCompositeSampleAt = 0;
}

function resetPunchCompositeState() {
  punchCompositePhase = "idle";
  lastPunchCompositeEvidenceAt = 0;
}

function resetHighKneeCompositeState() {
  highKneeCompositeSide = "unknown";
  highKneeCompositeSideChanges = [];
  lastHighKneeCompositeEvidenceAt = 0;
}

function resetSquatCompositeState() {
  squatCompositePhase = "idle";
  lastSquatCompositeEvidenceAt = 0;
}

function resetMotionCompositeStates() {
  resetWaveCompositeState();
  resetPunchCompositeState();
  resetHighKneeCompositeState();
  resetSquatCompositeState();
}

function getDominantWristHorizontalDirection(
  previous: FullBodyMotionSample,
  current: FullBodyMotionSample,
) {
  const shoulderWidth = Math.max(
    getTrackedDistance(current, "leftShoulder", "rightShoulder") ?? 0,
    MIN_SHOULDER_WIDTH,
  );
  const deltas = (["leftWrist", "rightWrist"] as FullBodyTrackedKey[])
    .map((key) => getTrackedDelta(current, previous, key))
    .filter((delta): delta is { x: number; y: number; distance: number } => Boolean(delta))
    .sort((left, right) => Math.abs(right.x) - Math.abs(left.x));
  const dominantDelta = deltas[0];

  if (!dominantDelta || shoulderWidth <= 0) {
    return { direction: 0 as WaveCompositeDirection, horizontalRatio: 0 };
  }

  const horizontalRatio = Math.abs(dominantDelta.x) / shoulderWidth;
  const direction: WaveCompositeDirection =
    horizontalRatio >= WAVE_COMPOSITE_DIRECTION_RATIO
      ? dominantDelta.x > 0
        ? 1
        : -1
      : 0;

  return { direction, horizontalRatio };
}

function updateWaveCompositeState(now: number) {
  const current = fullBodyMotionHistory[fullBodyMotionHistory.length - 1];
  const previous = fullBodyMotionHistory[fullBodyMotionHistory.length - 2];
  const swingScore = getBasicActionScore(["手水平摆动"]);
  const depthConflictScore = Math.max(
    getBasicActionScore(["手向前移动"]),
    getBasicActionScore(["手向后移动"]),
  );
  const armExtensionScore = getBasicActionScore(["手臂伸展"]);
  const fistScore = getPunchFistScore();
  const bodyLevelScore = getPunchBodyLevelScore(current);
  const features = fullBodyMotionFeatures.value;
  const visibleWristCount = current
    ? (current.points.leftWrist ? 1 : 0) + (current.points.rightWrist ? 1 : 0)
    : 0;
  let horizontalRatio = 0;

  if (previous && current && current.time !== lastWaveCompositeSampleAt) {
    if (lastWaveCompositeSampleAt && current.time - lastWaveCompositeSampleAt > 650) {
      waveCompositeDirection = 0;
      waveCompositeTurnTimes = [];
    }

    const wristMotion = getDominantWristHorizontalDirection(previous, current);
    horizontalRatio = wristMotion.horizontalRatio;
    lastWaveCompositeSampleAt = current.time;

    if (wristMotion.direction !== 0) {
      if (
        waveCompositeDirection !== 0 &&
        wristMotion.direction !== waveCompositeDirection
      ) {
        waveCompositeTurnTimes.push(now);
      }

      waveCompositeDirection = wristMotion.direction;
    }
  }

  waveCompositeTurnTimes = waveCompositeTurnTimes.filter(
    (time) => now - time <= WAVE_COMPOSITE_TURN_WINDOW_MS,
  );

  const turnCount = waveCompositeTurnTimes.length;
  const hasPunchLikeConflict =
    depthConflictScore >= WAVE_COMPOSITE_DEPTH_CONFLICT_SCORE &&
    armExtensionScore >= WAVE_COMPOSITE_EXTENSION_CONFLICT_SCORE &&
    fistScore >= PUNCH_COMPOSITE_MIN_FIST_SCORE &&
    bodyLevelScore >= PUNCH_COMPOSITE_MIN_BODY_LEVEL_SCORE;
  const hasWaveEvidence =
    visibleWristCount > 0 &&
    swingScore >= WAVE_COMPOSITE_MIN_SWING_SCORE &&
    features.upperBodyDistance >= WAVE_COMPOSITE_MIN_UPPER_DISTANCE &&
    (turnCount >= 1 || horizontalRatio >= WAVE_COMPOSITE_DIRECTION_RATIO * 1.8) &&
    !(hasPunchLikeConflict && turnCount < 2 && swingScore < 82);

  if (hasWaveEvidence) {
    lastWaveCompositeEvidenceAt = now;
  }

  const active = now - lastWaveCompositeEvidenceAt <= WAVE_COMPOSITE_HOLD_MS;
  const reasonParts = [
    `手水平摆动 ${swingScore}`,
    `${turnCount} 次换向`,
    `腕点可见 ${visibleWristCount}/2`,
  ];

  if (hasPunchLikeConflict) {
    reasonParts.push(`疑似出拳: 握拳 ${fistScore}，水平前侧 ${bodyLevelScore}`);
  }

  return {
    active,
    reason: reasonParts.join("，"),
  };
}

function updatePunchCompositeState(now: number) {
  const current = fullBodyMotionHistory[fullBodyMotionHistory.length - 1];
  const forwardScore = getBasicActionScore(["手向前移动"]);
  const backwardScore = getBasicActionScore(["手向后移动"]);
  const extensionScore = getBasicActionScore(["手臂伸展"]);
  const bendScore = getBasicActionScore(["手臂弯曲"]);
  const swingScore = getBasicActionScore(["手水平摆动"]);
  const fistScore = getPunchFistScore();
  const bodyLevelScore = getPunchBodyLevelScore(current);
  const features = fullBodyMotionFeatures.value;
  const hasExtendEvidence =
    forwardScore >= PUNCH_COMPOSITE_MIN_FORWARD_SCORE &&
    extensionScore >= PUNCH_COMPOSITE_MIN_EXTENSION_SCORE &&
    fistScore >= PUNCH_COMPOSITE_MIN_FIST_SCORE &&
    bodyLevelScore >= PUNCH_COMPOSITE_MIN_BODY_LEVEL_SCORE &&
    features.upperBodyDistance >= PUNCH_COMPOSITE_MIN_UPPER_DISTANCE &&
    swingScore < 78;
  const hasRecoveryEvidence =
    punchCompositePhase !== "idle" &&
    (backwardScore >= PUNCH_COMPOSITE_RECOVERY_SCORE ||
      bendScore >= PUNCH_COMPOSITE_RECOVERY_SCORE);

  if (hasExtendEvidence) {
    punchCompositePhase = "extended";
    lastPunchCompositeEvidenceAt = now;
  } else if (hasRecoveryEvidence) {
    punchCompositePhase = "recovering";
    lastPunchCompositeEvidenceAt = now;
  }

  const active = now - lastPunchCompositeEvidenceAt <= PUNCH_COMPOSITE_HOLD_MS;

  if (!active) {
    punchCompositePhase = "idle";
  }

  return {
    active,
    reason:
      punchCompositePhase === "recovering"
        ? `手向前 ${forwardScore}，握拳 ${fistScore}，回收 ${Math.max(backwardScore, bendScore)}`
        : `手向前 ${forwardScore}，伸展 ${extensionScore}，握拳 ${fistScore}，水平前侧 ${bodyLevelScore}`,
  };
}

function updateJumpingJackCompositeState(now: number) {
  const armsOpenScore = getBasicActionScore(["双臂上抬"]);
  const armsCloseScore = getBasicActionScore(["手臂回落"]);
  const feetOpenScore = getBasicActionScore(["双脚打开"]);
  const feetCloseScore = getBasicActionScore(["双脚合拢"]);
  const features = fullBodyMotionFeatures.value;
  const hasOpenEvidence =
    armsOpenScore >= JUMPING_JACK_BASIC_MIN_ARM_SCORE &&
    feetOpenScore >= JUMPING_JACK_BASIC_MIN_FOOT_SCORE &&
    features.upperBodyDistance >= JUMPING_JACK_BASIC_MIN_UPPER_DISTANCE &&
    features.lowerBodyDistance >= JUMPING_JACK_BASIC_MIN_LOWER_DISTANCE;
  const hasCloseEvidence =
    armsCloseScore >= JUMPING_JACK_BASIC_MIN_ARM_SCORE &&
    feetCloseScore >= JUMPING_JACK_BASIC_MIN_FOOT_SCORE &&
    features.upperBodyDistance >= JUMPING_JACK_BASIC_MIN_UPPER_DISTANCE &&
    features.lowerBodyDistance >= JUMPING_JACK_BASIC_MIN_LOWER_DISTANCE;

  if (hasOpenEvidence || hasCloseEvidence) {
    lastCompositeEvidenceAt.value = now;
  }

  return {
    active:
      lastCompositeEvidenceAt.value > 0 &&
      now - lastCompositeEvidenceAt.value <= COMPOSITE_ACTION_HOLD_MS,
    reason: hasOpenEvidence
      ? `双臂上抬 ${armsOpenScore}，双脚打开 ${feetOpenScore}`
      : hasCloseEvidence
        ? `手臂回落 ${armsCloseScore}，双脚合拢 ${feetCloseScore}`
        : `脚 ${feetState.value} / 手 ${armsState.value}`,
  };
}

function updateHighKneeCompositeState(now: number) {
  const leftScore = getBasicActionScore(["左抬腿"]);
  const rightScore = getBasicActionScore(["右抬腿"]);
  const kneeLiftScore = getBasicActionScore(["膝盖上抬"]);
  const features = fullBodyMotionFeatures.value;
  const dominantSide: HighKneeCompositeSide =
    leftScore >= HIGH_KNEE_COMPOSITE_MIN_SCORE && leftScore >= rightScore
      ? "left"
      : rightScore >= HIGH_KNEE_COMPOSITE_MIN_SCORE
        ? "right"
        : "unknown";

  if (
    dominantSide !== "unknown" &&
    features.lowerBodyDistance >= HIGH_KNEE_COMPOSITE_MIN_LOWER_DISTANCE
  ) {
    if (highKneeCompositeSide !== "unknown" && dominantSide !== highKneeCompositeSide) {
      highKneeCompositeSideChanges.push(now);
    }

    highKneeCompositeSide = dominantSide;
    lastHighKneeCompositeEvidenceAt = now;
  }

  highKneeCompositeSideChanges = highKneeCompositeSideChanges.filter(
    (time) => now - time <= HIGH_KNEE_COMPOSITE_ALTERNATE_WINDOW_MS,
  );

  const active =
    now - lastHighKneeCompositeEvidenceAt <= HIGH_KNEE_COMPOSITE_HOLD_MS &&
    kneeLiftScore >= HIGH_KNEE_COMPOSITE_MIN_SCORE;

  if (!active && now - lastHighKneeCompositeEvidenceAt > HIGH_KNEE_COMPOSITE_HOLD_MS) {
    highKneeCompositeSide = "unknown";
  }

  return {
    active,
    reason: `${dominantSide === "left" ? "左" : dominantSide === "right" ? "右" : "膝"}膝上抬 ${kneeLiftScore}，左右交替 ${highKneeCompositeSideChanges.length} 次`,
  };
}

function updateSquatCompositeState(now: number) {
  const squatScore = getBasicActionScore(["下蹲"]);
  const standUpScore = getBasicActionScore(["起身"]);
  const features = fullBodyMotionFeatures.value;
  const hasDownEvidence =
    squatScore >= SQUAT_COMPOSITE_MIN_DOWN_SCORE &&
    features.lowerBodyDistance >= SQUAT_COMPOSITE_MIN_LOWER_DISTANCE;
  const hasRiseEvidence =
    squatCompositePhase !== "idle" &&
    standUpScore >= SQUAT_COMPOSITE_MIN_UP_SCORE &&
    features.lowerBodyDistance >= SQUAT_COMPOSITE_MIN_LOWER_DISTANCE;

  if (hasDownEvidence) {
    squatCompositePhase = "down";
    lastSquatCompositeEvidenceAt = now;
  } else if (hasRiseEvidence) {
    squatCompositePhase = "rising";
    lastSquatCompositeEvidenceAt = now;
  }

  const active = now - lastSquatCompositeEvidenceAt <= SQUAT_COMPOSITE_HOLD_MS;

  if (!active) {
    squatCompositePhase = "idle";
  }

  return {
    active,
    reason:
      squatCompositePhase === "rising"
        ? `下蹲 ${squatScore}，起身 ${standUpScore}`
        : `下蹲 ${squatScore}，下肢位移 ${features.lowerBodyDistance.toFixed(3)}`,
  };
}

function getCompositeDetail(reason = "") {
  const actionText = topFullBodyBasicActionsText.value;

  if (!fullBodyBasicActions.value.length) {
    return reason || "等待稳定基础动作证据";
  }

  return reason ? `${actionText} / ${reason}` : `基础动作: ${actionText}`;
}

function updateCompositeActionDisplay(now: number, reason = "") {
  const isRecentComposite =
    lastCompositeEvidenceAt.value > 0 &&
    now - lastCompositeEvidenceAt.value <= COMPOSITE_ACTION_HOLD_MS;

  if (jumpingJackState.value === "open" || isRecentComposite) {
    resetRestCompositeState();
    resetMotionCompositeStates();
    compositeActionLabel.value = "开合跳";
    compositeActionDetail.value = `脚 ${feetState.value} / 手 ${armsState.value}`;
    return;
  }

  if (poseStatus.value !== "detected") {
    resetRestCompositeState();
    resetMotionCompositeStates();
    compositeActionLabel.value = "未识别组合动作";
    compositeActionDetail.value = reason || "等待人体入镜";
    return;
  }

  const actionLabels = getTopBasicActionLabels();
  const hasUpperBodyAction = actionLabels.some((label) =>
    UPPER_BODY_BASIC_ACTION_LABELS.includes(label),
  );
  const hasLowerBodyAction = actionLabels.some((label) =>
    LOWER_BODY_BASIC_ACTION_LABELS.includes(label),
  );
  const hasTorsoOrHeadAction = actionLabels.some((label) =>
    TORSO_OR_HEAD_BASIC_ACTION_LABELS.includes(label),
  );
  const upperBodyCompositeScore = getBasicActionScore(UPPER_BODY_BASIC_ACTION_LABELS);
  const lowerBodyCompositeScore = getBasicActionScore(LOWER_BODY_BASIC_ACTION_LABELS);
  const torsoOrHeadCompositeScore = getBasicActionScore(TORSO_OR_HEAD_BASIC_ACTION_LABELS);
  const features = fullBodyMotionFeatures.value;
  const hasStableFullBodyEvidence =
    upperBodyCompositeScore >= FULL_BODY_COMPOSITE_MIN_ACTION_SCORE &&
    lowerBodyCompositeScore >= FULL_BODY_COMPOSITE_MIN_ACTION_SCORE &&
    features.upperBodyDistance >= FULL_BODY_COMPOSITE_MIN_UPPER_DISTANCE &&
    features.lowerBodyDistance >= FULL_BODY_COMPOSITE_MIN_LOWER_DISTANCE;
  const jumpingJackComposite = updateJumpingJackCompositeState(now);
  const squatComposite = updateSquatCompositeState(now);
  const highKneeComposite = updateHighKneeCompositeState(now);
  const punchComposite = updatePunchCompositeState(now);
  const waveComposite = updateWaveCompositeState(now);
  const isRestCandidate =
    fullBodyActivityScore.value <= REST_COMPOSITE_MAX_ACTIVITY_SCORE &&
    features.movementDistance <= REST_COMPOSITE_MAX_MOVEMENT_DISTANCE &&
    features.upperBodyDistance <= REST_COMPOSITE_MAX_UPPER_DISTANCE &&
    features.lowerBodyDistance <= REST_COMPOSITE_MAX_LOWER_DISTANCE;
  const hasRestExitEvidence =
    fullBodyActivityScore.value >= REST_COMPOSITE_EXIT_ACTIVITY_SCORE ||
    (upperBodyCompositeScore >= REST_COMPOSITE_EXIT_ACTION_SCORE &&
      features.upperBodyDistance >= REST_COMPOSITE_EXIT_UPPER_DISTANCE) ||
    (lowerBodyCompositeScore >= REST_COMPOSITE_EXIT_ACTION_SCORE &&
      features.lowerBodyDistance >= REST_COMPOSITE_EXIT_LOWER_DISTANCE) ||
    (torsoOrHeadCompositeScore >= REST_COMPOSITE_EXIT_ACTION_SCORE &&
      features.torsoDistance >= REST_COMPOSITE_EXIT_TORSO_DISTANCE);
  const shouldShowRest = updateRestCompositeState(
    now,
    isRestCandidate,
    hasRestExitEvidence,
  );

  if (shouldShowRest) {
    compositeActionLabel.value = "静息";
    compositeActionDetail.value = reason
      ? `${reason} / 低运动量持续稳定`
      : "检测到人体，低运动量持续稳定";
    return;
  }

  if (hasStableFullBodyEvidence) {
    resetRestCompositeState();
  }

  if (jumpingJackComposite.active) {
    resetRestCompositeState();
    resetPunchCompositeState();
    resetWaveCompositeState();
    compositeActionLabel.value = "开合跳";
    compositeActionDetail.value = getCompositeDetail(jumpingJackComposite.reason);
    return;
  }

  if (squatComposite.active) {
    resetRestCompositeState();
    compositeActionLabel.value = "深蹲";
    compositeActionDetail.value = getCompositeDetail(squatComposite.reason);
    return;
  }

  if (highKneeComposite.active) {
    resetRestCompositeState();
    compositeActionLabel.value = "高抬腿";
    compositeActionDetail.value = getCompositeDetail(highKneeComposite.reason);
    return;
  }

  if (punchComposite.active) {
    resetRestCompositeState();
    resetWaveCompositeState();
    compositeActionLabel.value = "出拳";
    compositeActionDetail.value = getCompositeDetail(punchComposite.reason);
    return;
  }

  if (hasStableFullBodyEvidence) {
    compositeActionLabel.value = "全身活动";
    compositeActionDetail.value = getCompositeDetail(
      reason || "上肢和下肢都有稳定运动证据",
    );
    return;
  }

  if (waveComposite.active) {
    compositeActionLabel.value = "挥手";
    compositeActionDetail.value = getCompositeDetail(waveComposite.reason);
    return;
  }

  if (hasUpperBodyAction) {
    compositeActionLabel.value = "上肢活动";
    compositeActionDetail.value = getCompositeDetail(reason);
    return;
  }

  if (hasLowerBodyAction) {
    compositeActionLabel.value = "下肢活动";
    compositeActionDetail.value = getCompositeDetail(reason);
    return;
  }

  if (hasTorsoOrHeadAction) {
    compositeActionLabel.value = "姿态活动";
    compositeActionDetail.value = getCompositeDetail(reason);
    return;
  }

  if (poseStatus.value === "detected" && fullBodyIntensityLevel.value === "静止") {
    compositeActionLabel.value = "静息";
    compositeActionDetail.value = reason ? `${reason} / 当前运动量较低` : "检测到人体，当前运动量较低";
    return;
  }

  if (fullBodyActivityScore.value >= 25) {
    compositeActionLabel.value = "活动中";
    compositeActionDetail.value = reason || "根据运动量判断为活动状态";
    return;
  }

  compositeActionLabel.value = "未识别组合动作";
  compositeActionDetail.value = reason || "当前主要看基础动作和运动量";
}

function updateDebouncedJumpingJackState(
  nextState: JumpingJackState,
  now: number,
) {
  if (nextState === "open") {
    lastCompositeEvidenceAt.value = now;
  }

  if (nextState === "unknown") {
    jumpingJackState.value = "unknown";
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
    updateCompositeActionDisplay(now);
    return;
  }

  if (nextState === jumpingJackState.value) {
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
    updateCompositeActionDisplay(now);
    return;
  }

  if (nextState !== pendingJumpingJackState) {
    pendingJumpingJackState = nextState;
    pendingJumpingJackFrames = 1;
  } else {
    pendingJumpingJackFrames += 1;
  }

  if (pendingJumpingJackFrames >= ACTION_DEBOUNCE_FRAMES) {
    jumpingJackState.value = nextState;
    pendingJumpingJackState = "unknown";
    pendingJumpingJackFrames = 0;
  }

  updateCompositeActionDisplay(now);
}

function updateJumpingJackAnalysis(
  landmarks: LandmarkPoint[] | null,
  now: number,
) {
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
    resetJumpingJackAnalysis("人体距离过远或未正对镜头");
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
  updateDebouncedJumpingJackState(nextActionState, now);
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

function shouldDrawPosePoint(index: number) {
  return index >= 11 || DISPLAY_FACE_LANDMARK_INDICES.includes(index);
}

function drawBodySegmentGroup(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[],
  segments: readonly (readonly [number, number])[],
  color: string,
  width: number,
) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.lineCap = "round";
  context.lineJoin = "round";

  segments.forEach(([fromIndex, toIndex]) => {
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

  context.restore();
}

function drawTorsoPanel(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[],
) {
  const torsoKeys = [
    LANDMARK_INDEX.leftShoulder,
    LANDMARK_INDEX.rightShoulder,
    LANDMARK_INDEX.rightHip,
    LANDMARK_INDEX.leftHip,
  ];
  const torsoPoints = torsoKeys
    .map((index) => landmarks[index])
    .filter((landmark): landmark is LandmarkPoint => isVisibleLandmark(landmark));

  if (torsoPoints.length < 3) {
    return;
  }

  const points = torsoKeys
    .map((index) => landmarks[index])
    .filter((landmark): landmark is LandmarkPoint => isVisibleLandmark(landmark))
    .map((landmark) => toCanvasPoint(landmark, canvas.width, canvas.height));

  context.save();
  context.fillStyle = "rgba(45, 212, 191, 0.18)";
  context.strokeStyle = "rgba(20, 184, 166, 0.92)";
  context.lineWidth = Math.max(3, canvas.width / 320);
  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y);
    } else {
      context.lineTo(point.x, point.y);
    }
  });
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
}

function drawPoseJoint(
  context: CanvasRenderingContext2D,
  point: { x: number; y: number },
  radius: number,
  fill: string,
) {
  context.beginPath();
  context.fillStyle = fill;
  context.strokeStyle = "rgba(15, 23, 42, 0.76)";
  context.lineWidth = 2;
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
}

function drawPoseOverlay(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[] | null,
) {
  if (!landmarks?.length) {
    return;
  }

  drawTorsoPanel(context, canvas, landmarks);
  drawBodySegmentGroup(
    context,
    canvas,
    landmarks,
    CENTER_BODY_SEGMENTS,
    "rgba(45, 212, 191, 0.92)",
    Math.max(5, canvas.width / 220),
  );
  drawBodySegmentGroup(
    context,
    canvas,
    landmarks,
    LEFT_BODY_SEGMENTS,
    "rgba(96, 165, 250, 0.96)",
    Math.max(8, canvas.width / 150),
  );
  drawBodySegmentGroup(
    context,
    canvas,
    landmarks,
    RIGHT_BODY_SEGMENTS,
    "rgba(250, 204, 21, 0.96)",
    Math.max(8, canvas.width / 150),
  );
  drawBodySegmentGroup(
    context,
    canvas,
    landmarks,
    FOOT_SEGMENTS,
    "rgba(226, 232, 240, 0.86)",
    Math.max(4, canvas.width / 260),
  );

  landmarks.forEach((landmark, index) => {
    if (!isVisibleLandmark(landmark) || !shouldDrawPosePoint(index)) {
      return;
    }

    const point = toCanvasPoint(landmark, canvas.width, canvas.height);
    const isRequired = REQUIRED_LANDMARK_INDICES.includes(index);
    const isFace = DISPLAY_FACE_LANDMARK_INDICES.includes(index);
    const isEmphasized = EMPHASIZED_JOINT_INDICES.has(index);

    drawPoseJoint(
      context,
      point,
      isEmphasized ? Math.max(8, canvas.width / 170) : isRequired ? 7 : 5,
      isEmphasized ? "#ffffff" : isRequired ? "#facc15" : isFace ? "#fb7185" : "#ffffff",
    );
  });
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
  context.lineWidth = Math.max(2, canvas.width / 420);

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
      context.lineWidth = 1.5;
      context.arc(
        point.x * canvas.width,
        point.y * canvas.height,
        Math.max(2.5, canvas.width / 520),
        0,
        Math.PI * 2,
      );
      context.fill();
      context.stroke();
    });
  });

  context.restore();
}

async function runPoseInference(now: number) {
  const video = videoRef.value;

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
      latestHands = (handResult.landmarks as LandmarkPoint[][]).map(toLandmarkPoints);
      handCount.value = latestHands.length;
    } else {
      latestHands = [];
      handCount.value = 0;
    }

    if (!poseLandmarker) {
      updateInferenceFps(now);
      pauseDemoElapsed();
      return;
    }

    const result = poseLandmarker.detectForVideo(video, now);
    const landmarks = result.landmarks[0] as LandmarkPoint[] | undefined;

    updateInferenceFps(now);

    if (!landmarks?.length) {
      poseStatus.value = "missing";
      landmarkCompleteness.value = 0;
      landmarkHistory.length = 0;
      smoothedLandmarks = null;
      pauseDemoElapsed();
      resetJumpingJackAnalysis("未检测到完整人体");
      resetFullBodyMotionFeatures();
      resetDepthServiceStatus("等待人体关键点 / 使用 2D 估算");
      return;
    }

    poseStatus.value = "detected";
    updateDemoElapsed(now);
    updateLandmarkCompleteness(landmarks);
    updateSmoothedLandmarks(landmarks);
    updateJumpingJackAnalysis(smoothedLandmarks, now);
    updateFullBodyMotionFeatures(smoothedLandmarks, now);
    updateCompositeActionDisplay(now);
    void updateDepthServiceStatus(smoothedLandmarks, now);
  } catch {
    poseStatus.value = "missing";
    pauseDemoElapsed();
    resetJumpingJackAnalysis("姿态推理失败");
    resetFullBodyMotionFeatures();
    resetDepthServiceStatus("姿态推理失败 / 使用 2D 估算");
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
    cameraMessage.value = "摄像头已就绪";
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
});

onBeforeUnmount(() => {
  stopCamera();
  disposePoseLandmarker();
  disposeHandLandmarker();
});
</script>

<template>
  <main class="demo-shell">
    <section class="demo-stage-panel" aria-label="演示版摄像头识别区域">
      <div class="demo-stage-toolbar">
        <div>
          <p class="eyebrow">CalorieCal Demo</p>
          <h1>运动强度演示</h1>
        </div>
        <span class="status-pill" :class="`status-${cameraStatus}`">
          {{ cameraStatusLabel }}
        </span>
      </div>

      <div class="demo-camera-stage">
        <div class="demo-camera-frame">
          <video ref="videoRef" aria-label="摄像头原始画面" playsinline muted />
          <canvas ref="canvasRef" aria-label="运动识别画布渲染层" />

          <div v-if="cameraStatus !== 'ready'" class="camera-overlay">
            <span>{{ cameraOverlayTitle }}</span>
            <p>{{ cameraMessage }}</p>
          </div>

          <div v-else class="demo-live-banner" aria-live="polite">
            {{ systemStatusText }}
          </div>

          <div v-if="demoCenterPrompt" class="demo-center-prompt" aria-live="polite">
            <strong>{{ demoCenterPrompt.title }}</strong>
            <span>{{ demoCenterPrompt.detail }}</span>
          </div>

          <div
            v-if="cameraStatus === 'ready' && poseStatus === 'detected'"
            class="demo-action-badge"
          >
            <span>{{ fullBodyIntensityLevel }}</span>
            <strong>{{ compositeActionLabel }}</strong>
            <small>{{ topFullBodyBasicActionsText }}</small>
          </div>
        </div>
      </div>
    </section>

    <aside class="demo-side-panel" aria-label="演示版实时指标">
      <section class="demo-hero-metric">
        <span>当前组合动作</span>
        <strong>{{ compositeActionLabel }}</strong>
        <small>{{ compositeActionDetail }}</small>
      </section>

      <section class="demo-hero-metric accent">
        <span>估算消耗</span>
        <strong>{{ fullBodyCalories.toFixed(2) }} cal</strong>
        <small>最近 1 秒实时估算</small>
      </section>

      <div class="demo-metrics-grid">
        <section
          v-for="metric in primaryMetrics"
          :key="metric.label"
          class="demo-metric-card"
          :class="metric.tone ? `tone-${metric.tone}` : undefined"
        >
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small>{{ metric.detail }}</small>
        </section>
      </div>

      <section class="demo-section">
        <div class="demo-section-header">
          <span>运动量</span>
          <small>{{ motionSuggestion }}</small>
        </div>
        <div class="demo-metrics-grid compact">
          <section
            v-for="metric in motionMetrics"
            :key="metric.label"
            class="demo-metric-card"
            :class="metric.tone ? `tone-${metric.tone}` : undefined"
          >
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.detail }}</small>
          </section>
        </div>
      </section>

      <section class="demo-section">
        <div class="demo-section-header">
          <span>前后方向</span>
          <small>{{ depthServiceStatus === "ready" ? "RGB-D 证据" : "当前使用 2D 估算" }}</small>
        </div>
        <div class="demo-metrics-grid compact">
          <section
            v-for="metric in depthActionMetrics"
            :key="metric.label"
            class="demo-metric-card"
            :class="metric.tone ? `tone-${metric.tone}` : undefined"
          >
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
            <small>{{ metric.detail }}</small>
          </section>
        </div>
      </section>

      <section class="demo-status-strip">
        <span>{{ modelStatusLabel }}</span>
        <span>{{ handStatusLabel }}</span>
        <span>{{ depthStatusLabel }}</span>
        <span>{{ inferenceFps.toFixed(1) }} FPS</span>
        <span>{{ handCount }} hand</span>
      </section>
    </aside>
  </main>
</template>

<style scoped>
.demo-shell {
  align-items: start;
  display: grid;
  gap: 20px;
  grid-template-columns: minmax(0, 1fr) 360px;
  width: 100%;
}

.demo-stage-panel,
.demo-side-panel {
  background: #ffffff;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}

.demo-stage-panel {
  display: flex;
  flex-direction: column;
  min-height: calc(100vh - 48px);
  overflow: hidden;
}

.demo-stage-toolbar {
  align-items: center;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 20px 24px;
}

.eyebrow {
  color: #0f766e;
  font-size: 13px;
  font-weight: 900;
  margin: 0 0 5px;
}

.demo-stage-toolbar h1 {
  color: #0f172a;
  font-size: 34px;
  line-height: 1.05;
  margin: 0;
}

.status-pill {
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  color: #334155;
  font-size: 15px;
  font-weight: 900;
  padding: 9px 14px;
  white-space: nowrap;
}

.status-ready {
  background: #ecfdf5;
  border-color: #86efac;
  color: #166534;
}

.status-loading,
.status-idle {
  background: #eff6ff;
  border-color: #93c5fd;
  color: #1d4ed8;
}

.status-error {
  background: #fff7ed;
  border-color: #fdba74;
  color: #9a3412;
}

.demo-camera-stage {
  align-items: center;
  background:
    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(20, 83, 45, 0.9)),
    #0f172a;
  display: flex;
  flex: 1;
  justify-content: center;
  min-height: 540px;
  padding: 24px;
}

.demo-camera-frame {
  aspect-ratio: 16 / 9;
  border: 1px solid rgba(148, 163, 184, 0.5);
  border-radius: 8px;
  max-width: 1180px;
  overflow: hidden;
  position: relative;
  width: 100%;
}

.demo-camera-frame video,
.demo-camera-frame canvas {
  height: 100%;
  inset: 0;
  object-fit: cover;
  position: absolute;
  transform: scaleX(-1);
  width: 100%;
}

.demo-camera-frame video {
  opacity: 0;
}

.demo-camera-frame canvas {
  background: #020617;
}

.demo-live-banner {
  background: rgba(15, 23, 42, 0.74);
  border: 1px solid rgba(148, 163, 184, 0.38);
  border-radius: 6px;
  bottom: 16px;
  color: #f8fafc;
  font-size: 15px;
  font-weight: 800;
  left: 16px;
  max-width: calc(100% - 32px);
  padding: 10px 12px;
  position: absolute;
  transition: background-color 160ms ease, border-color 160ms ease;
}

.camera-overlay,
.demo-center-prompt {
  align-items: center;
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid rgba(226, 232, 240, 0.28);
  border-radius: 8px;
  color: #f8fafc;
  display: flex;
  flex-direction: column;
  gap: 8px;
  left: 50%;
  max-width: min(480px, calc(100% - 48px));
  padding: 22px 24px;
  position: absolute;
  text-align: center;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 3;
}

.camera-overlay span,
.demo-center-prompt strong {
  color: #ffffff;
  font-size: 26px;
  font-weight: 900;
  line-height: 1.15;
}

.camera-overlay p,
.demo-center-prompt span {
  color: #cbd5e1;
  font-size: 16px;
  font-weight: 800;
  line-height: 1.45;
  margin: 0;
}

.demo-action-badge {
  animation: subtleRise 260ms ease-out;
  background: rgba(15, 23, 42, 0.78);
  border: 1px solid rgba(250, 204, 21, 0.58);
  border-radius: 8px;
  color: #e2e8f0;
  display: grid;
  gap: 4px;
  max-width: min(360px, calc(100% - 32px));
  padding: 14px 16px;
  position: absolute;
  right: 16px;
  text-align: right;
  top: 16px;
  z-index: 2;
}

.demo-action-badge span,
.demo-action-badge small {
  color: #cbd5e1;
  font-size: 13px;
  font-weight: 800;
}

.demo-action-badge strong {
  color: #ffffff;
  font-size: 30px;
  line-height: 1.1;
}

.demo-side-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
  max-height: calc(100vh - 48px);
  overflow-y: auto;
  padding: 18px;
  position: sticky;
  top: 24px;
}

.demo-hero-metric,
.demo-section {
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  display: grid;
  gap: 10px;
  padding: 18px;
}

.demo-hero-metric {
  background: #f8fafc;
  min-height: 132px;
}

.demo-hero-metric.accent {
  background: #ecfdf5;
  border-color: #99f6e4;
}

.demo-hero-metric span,
.demo-metric-card span,
.demo-section-header span {
  color: #475569;
  font-size: 14px;
  font-weight: 900;
}

.demo-hero-metric strong {
  color: #0f172a;
  font-size: 38px;
  line-height: 1.1;
  overflow-wrap: anywhere;
}

.demo-hero-metric small,
.demo-metric-card small,
.demo-section-header small {
  color: #64748b;
  font-size: 13px;
  font-weight: 750;
  line-height: 1.35;
}

.demo-metrics-grid {
  display: grid;
  gap: 12px;
}

.demo-metrics-grid.compact {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.demo-metric-card {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: grid;
  gap: 7px;
  min-height: 122px;
  padding: 15px;
  transition:
    background-color 180ms ease,
    border-color 180ms ease,
    box-shadow 180ms ease;
}

.demo-metric-card strong {
  color: #0f172a;
  font-size: 25px;
  line-height: 1.14;
  overflow-wrap: anywhere;
}

.demo-metric-card.tone-active {
  background: #eff6ff;
  border-color: #93c5fd;
}

.demo-metric-card.tone-good {
  background: #ecfdf5;
  border-color: #86efac;
}

.demo-metric-card.tone-warning {
  background: #fffbeb;
  border-color: #fcd34d;
}

.demo-metric-card.tone-danger {
  background: #fff7ed;
  border-color: #fdba74;
}

.demo-metric-card.tone-active strong {
  color: #1d4ed8;
}

.demo-metric-card.tone-good strong {
  color: #166534;
}

.demo-metric-card.tone-warning strong {
  color: #92400e;
}

.demo-metric-card.tone-danger strong {
  color: #9a3412;
}

.demo-section-header {
  align-items: start;
  display: flex;
  gap: 10px;
  justify-content: space-between;
}

.demo-section-header small {
  max-width: 170px;
  text-align: right;
}

.demo-status-strip {
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding: 12px;
}

.demo-status-strip span {
  color: #64748b;
  font-size: 12px;
  font-weight: 800;
}

@keyframes subtleRise {
  from {
    opacity: 0.82;
    transform: translateY(-4px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (min-width: 1500px) {
  .demo-shell {
    gap: 24px;
    grid-template-columns: minmax(0, 1fr) 420px;
  }

  .demo-stage-toolbar h1 {
    font-size: 40px;
  }

  .demo-action-badge strong {
    font-size: 36px;
  }

  .demo-hero-metric strong {
    font-size: 44px;
  }

  .demo-metric-card strong {
    font-size: 29px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .demo-action-badge {
    animation: none;
  }

  .demo-live-banner,
  .demo-metric-card {
    transition: none;
  }
}

@media (max-width: 1100px), (orientation: portrait) {
  .demo-shell {
    align-items: stretch;
    gap: 12px;
    grid-template-areas:
      "toolbar toolbar"
      "hero-main hero-kcal"
      "primary primary"
      "camera camera"
      "motion motion"
      "depth depth"
      "status status";
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .demo-stage-panel,
  .demo-side-panel {
    background: transparent;
    border: 0;
    box-shadow: none;
    display: contents;
  }

  .demo-stage-toolbar {
    background: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 8px;
    grid-area: toolbar;
    padding: 16px 20px;
  }

  .demo-stage-toolbar h1 {
    font-size: 30px;
  }

  .demo-camera-stage {
    border-radius: 8px;
    grid-area: camera;
    min-height: min(78vh, 1320px);
    min-height: min(78dvh, 1320px);
    padding: 18px;
  }

  .demo-camera-frame {
    aspect-ratio: 9 / 16;
    max-height: min(72vh, 1280px);
    max-height: min(72dvh, 1280px);
    max-width: 100%;
    width: min(100%, 720px);
  }

  .demo-camera-frame video,
  .demo-camera-frame canvas {
    object-fit: contain;
  }

  .demo-side-panel > .demo-hero-metric:first-of-type {
    grid-area: hero-main;
  }

  .demo-side-panel > .demo-hero-metric.accent {
    grid-area: hero-kcal;
  }

  .demo-side-panel > .demo-metrics-grid {
    grid-area: primary;
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .demo-side-panel > .demo-section:nth-of-type(3) {
    grid-area: motion;
  }

  .demo-side-panel > .demo-section:nth-of-type(4) {
    grid-area: depth;
  }

  .demo-status-strip {
    grid-area: status;
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .demo-hero-metric {
    min-height: 112px;
    padding: 16px;
  }

  .demo-hero-metric strong {
    font-size: 32px;
  }

  .demo-metric-card {
    min-height: 108px;
    padding: 13px;
  }

  .demo-section .demo-metrics-grid.compact {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .demo-section-header small {
    max-width: 360px;
  }
}

@media (max-width: 640px) {
  .demo-shell {
    grid-template-areas:
      "toolbar"
      "hero-main"
      "hero-kcal"
      "primary"
      "camera"
      "motion"
      "depth"
      "status";
    grid-template-columns: 1fr;
  }

  .demo-camera-stage {
    min-height: 420px;
    padding: 12px;
  }

  .demo-camera-frame {
    aspect-ratio: 3 / 4;
    width: 100%;
  }

  .demo-side-panel > .demo-metrics-grid,
  .demo-section .demo-metrics-grid.compact,
  .demo-metrics-grid.compact,
  .demo-status-strip {
    grid-template-columns: 1fr;
  }

  .demo-action-badge {
    left: 12px;
    right: 12px;
    text-align: left;
  }
}
</style>
