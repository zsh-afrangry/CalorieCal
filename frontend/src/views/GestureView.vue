<script setup lang="ts">
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

type RuntimeStatus = "idle" | "loading" | "ready" | "error";
type AppRuntimeStatus = "idle" | "loading" | "ready" | "running" | "paused" | "error";
type LandmarkPoint = { x: number; y: number; z?: number; visibility?: number };
type GestureLabel =
  | "等待检测"
  | "静止"
  | "挥手"
  | "出拳"
  | "招手"
  | "握拳"
  | "手向前"
  | "手向后";
type WaveRuntimeStatus = "idle" | "recording" | "ended";
type WaveDirection = "left" | "right";
type ConfidenceLevel = "高" | "中" | "低";
type HandMotionSample = {
  time: number;
  centerX: number;
  centerY: number;
  scale: number;
  fingerCurl: number;
};
type MotionEvidence = {
  horizontalSwingScore: number;
  depthMoveScore: number;
  fingerCurlScore: number;
  waveCycleScore: number;
  stabilityScore: number;
};
type GestureCandidate = {
  label: GestureLabel;
  score: number;
  reason: string;
};
type WaveCounterSignal = {
  active: boolean;
  score: number;
  reason: string;
};
type PoseDetectionStatus = "idle" | "detected" | "missing";
type UpperBodySideSample = {
  shoulder: LandmarkPoint | null;
  elbow: LandmarkPoint | null;
  wrist: LandmarkPoint | null;
  handCenter: LandmarkPoint | null;
};
type UpperBodyMotionSample = {
  time: number;
  left: UpperBodySideSample;
  right: UpperBodySideSample;
};
type UpperBodyMotionFeatures = {
  movementDistance: number;
  movementSpeed: number;
  movementAmplitude: number;
  activeUpperBodyMs: number;
};
type UpperBodyIntensityLevel = "静止" | "低强度" | "中强度" | "高强度";
type BasicActionCandidate = {
  label: string;
  score: number;
  reason: string;
};

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const MOTION_HISTORY_MS = 1200;
const MIN_MOTION_DELTA = 0.015;
const WAVE_DIRECTION_DELTA = 0.012;
const WAVE_TURN_COOLDOWN_MS = 260;
const WAVE_TURNS_PER_COUNT = 2;
const WAVE_SIGNAL_MIN_SCORE = 55;
const WAVE_SIGNAL_GRACE_FRAMES = 4;
const UPPER_BODY_HISTORY_MS = 1200;
const UPPER_BODY_ACTIVE_DISTANCE_THRESHOLD = 0.025;
const UPPER_BODY_INTENSITY_HOLD_MS = 450;
const MIN_POSE_VISIBILITY = 0.45;
const POSE_LANDMARK_INDEX = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
} as const;
const EMPTY_MOTION_EVIDENCE: MotionEvidence = {
  horizontalSwingScore: 0,
  depthMoveScore: 0,
  fingerCurlScore: 0,
  waveCycleScore: 0,
  stabilityScore: 0,
};
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
const poseModelStatus = ref<RuntimeStatus>("idle");
const poseStatus = ref<PoseDetectionStatus>("idle");
const cameraMessage = ref("等待摄像头启动");
const modelMessage = ref("等待手势模型加载");
const poseModelMessage = ref("等待姿态模型加载");
const handCount = ref(0);
const inferenceFps = ref(0);
const lastHandedness = ref("无");
const upperBodyJointState = ref("等待姿态关键点");
const upperBodyMotionFeatures = ref<UpperBodyMotionFeatures>({
  movementDistance: 0,
  movementSpeed: 0,
  movementAmplitude: 0,
  activeUpperBodyMs: 0,
});
const upperBodyActivityScore = ref(0);
const upperBodyIntensityLevel = ref<UpperBodyIntensityLevel>("静止");
const lastActiveUpperBodyScore = ref(0);
const lastActiveUpperBodyLevel = ref<UpperBodyIntensityLevel>("静止");
const lastActiveUpperBodyAt = ref(0);
const upperBodyBasicActions = ref<BasicActionCandidate[]>([]);
const currentGesture = ref<GestureLabel>("等待检测");
const motionDirection = ref("等待检测");
const instantMotion = ref("等待检测");
const handScaleTrend = ref("等待检测");
const gestureReason = ref("等待手部入镜");
const motionEvidence = ref<MotionEvidence>({ ...EMPTY_MOTION_EVIDENCE });
const gestureCandidates = ref<GestureCandidate[]>([]);
const waveCounterSignalReason = ref("等待挥手证据");
const waveRuntimeStatus = ref<WaveRuntimeStatus>("idle");
const waveCount = ref(0);
const waveElapsedMs = ref(0);
const weightKg = ref(60);
const waveCounterReason = ref("点击开始后连续挥手，结束只停止记录。");
const lastWaveCountedAtText = ref("无");
const waveCycleGestureFrames = ref(0);
const waveCycleTurnCount = ref(0);

let stream: MediaStream | null = null;
let animationFrameId: number | null = null;
let handLandmarker: HandLandmarker | null = null;
let poseLandmarker: PoseLandmarker | null = null;
let isInferencing = false;
let lastInferenceStartedAt = 0;
let lastFpsSampleAt = 0;
let inferenceFrameCount = 0;
let latestHands: LandmarkPoint[][] = [];
let latestPoseLandmarks: LandmarkPoint[] | null = null;
const handMotionHistory: HandMotionSample[] = [];
const upperBodyMotionHistory: UpperBodyMotionSample[] = [];
let waveStartedAt = 0;
let waveAccumulatedMs = 0;
let lastWaveDirection: WaveDirection | null = null;
let lastWaveTurnAt = 0;
let waveLowSignalFrames = 0;

const topGestureCandidatesText = computed(() => {
  if (!gestureCandidates.value.length) {
    return "等待检测";
  }

  return gestureCandidates.value
    .slice(0, 3)
    .map((candidate) => `${candidate.label} ${candidate.score}`)
    .join(" / ");
});

const topUpperBodyBasicActionsText = computed(() => {
  if (!upperBodyBasicActions.value.length) {
    return "暂无基本动作";
  }

  return upperBodyBasicActions.value
    .slice(0, 3)
    .map((candidate) => `${candidate.label} ${candidate.score}`)
    .join(" / ");
});

const debugDetails = computed(() => [
  { label: "运行状态", value: appRuntimeLabel.value },
  { label: "运行说明", value: appRuntimeReason.value },
  { label: "手部数量", value: String(handCount.value) },
  { label: "姿态检测", value: poseStatus.value },
  { label: "上肢关键点", value: upperBodyJointState.value },
  { label: "当前动作", value: currentGesture.value },
  { label: "挥手计数状态", value: waveRuntimeLabel.value },
  { label: "挥手次数", value: String(waveCount.value) },
  { label: "挥手频率", value: `${waveFrequencyPerMin.value.toFixed(1)} 次/分钟` },
  { label: "挥手 MET", value: currentWaveMet.value.toFixed(1) },
  { label: "挥手 cal", value: waveCalories.value.toFixed(2) },
  { label: "估算置信度", value: waveConfidence.value },
  { label: "置信度原因", value: waveConfidenceSummary.value },
  { label: "本段挥手帧", value: String(waveCycleGestureFrames.value) },
  { label: "当前周期换向", value: String(waveCycleTurnCount.value) },
  { label: "最近计数", value: lastWaveCountedAtText.value },
  { label: "即时运动", value: instantMotion.value },
  { label: "运动方向", value: motionDirection.value },
  { label: "尺度变化", value: handScaleTrend.value },
  { label: "判断原因", value: gestureReason.value },
  { label: "候选状态", value: topGestureCandidatesText.value },
  { label: "横向摆动证据", value: String(motionEvidence.value.horizontalSwingScore) },
  { label: "前后移动证据", value: String(motionEvidence.value.depthMoveScore) },
  { label: "手指弯曲证据", value: String(motionEvidence.value.fingerCurlScore) },
  { label: "挥手周期证据", value: String(motionEvidence.value.waveCycleScore) },
  { label: "关键点稳定证据", value: String(motionEvidence.value.stabilityScore) },
  { label: "计数信号", value: waveCounterSignalReason.value },
  {
    label: "上肢位移",
    value: upperBodyMotionFeatures.value.movementDistance.toFixed(3),
  },
  {
    label: "上肢速度",
    value: upperBodyMotionFeatures.value.movementSpeed.toFixed(3),
  },
  {
    label: "上肢幅度",
    value: upperBodyMotionFeatures.value.movementAmplitude.toFixed(3),
  },
  {
    label: "上肢有效时长",
    value: formatElapsedTime(upperBodyMotionFeatures.value.activeUpperBodyMs),
  },
  { label: "上肢活动分数", value: String(upperBodyActivityScore.value) },
  { label: "上肢强度档位", value: upperBodyIntensityLevel.value },
  { label: "上肢 MET", value: currentUpperBodyMet.value.toFixed(1) },
  { label: "上肢 cal", value: upperBodyCalories.value.toFixed(2) },
  { label: "上肢关键点完整率", value: `${upperBodyJointCompleteness.value.toFixed(0)}%` },
  { label: "上肢估算置信度", value: upperBodyConfidence.value },
  { label: "上肢置信度原因", value: upperBodyConfidenceSummary.value },
  { label: "基本动作候选", value: topUpperBodyBasicActionsText.value },
  { label: "实际推理 FPS", value: inferenceFps.value.toFixed(1) },
  { label: "左右手", value: lastHandedness.value },
  { label: "手部模型", value: HAND_LANDMARKER_MODEL_URL },
  { label: "姿态模型", value: POSE_LANDMARKER_MODEL_URL },
]);

const waveFrequencyPerMin = computed(() => {
  const elapsedMinutes = waveElapsedMs.value / 60_000;
  return elapsedMinutes > 0 ? waveCount.value / elapsedMinutes : 0;
});

const currentWaveMet = computed(() => {
  const frequency = waveFrequencyPerMin.value;

  if (frequency > 45) {
    return 3.0;
  }

  if (frequency > 20) {
    return 2.3;
  }

  return 1.8;
});

const waveCalories = computed(() => {
  const elapsedMinutes = waveElapsedMs.value / 60_000;
  return ((currentWaveMet.value * 3.5 * weightKg.value * elapsedMinutes) / 200) * 1000;
});

const currentUpperBodyMet = computed(() => {
  if (upperBodyIntensityLevel.value === "高强度") {
    return 3.0;
  }

  if (upperBodyIntensityLevel.value === "中强度") {
    return 2.2;
  }

  if (upperBodyIntensityLevel.value === "低强度") {
    return 1.5;
  }

  return 1.0;
});

const upperBodyCalories = computed(() => {
  const activeMinutes = upperBodyMotionFeatures.value.activeUpperBodyMs / 60_000;
  return ((currentUpperBodyMet.value * 3.5 * weightKg.value * activeMinutes) / 200) * 1000;
});

const upperBodyJointCompleteness = computed(() => {
  const latest = upperBodyMotionHistory[upperBodyMotionHistory.length - 1];

  if (!latest) {
    return 0;
  }

  const joints = [
    latest.left.shoulder,
    latest.left.elbow,
    latest.left.wrist,
    latest.left.handCenter,
    latest.right.shoulder,
    latest.right.elbow,
    latest.right.wrist,
    latest.right.handCenter,
  ];

  return (joints.filter(Boolean).length / joints.length) * 100;
});

const upperBodyConfidence = computed<ConfidenceLevel>(() => {
  if (
    cameraStatus.value !== "ready" ||
    poseModelStatus.value !== "ready" ||
    poseStatus.value !== "detected" ||
    upperBodyJointCompleteness.value < 45
  ) {
    return "低";
  }

  if (
    upperBodyJointCompleteness.value >= 75 &&
    upperBodyMotionFeatures.value.activeUpperBodyMs >= 5000 &&
    upperBodyActivityScore.value >= 35 &&
    upperBodyMotionFeatures.value.movementAmplitude >= 0.08
  ) {
    return "高";
  }

  return "中";
});

const upperBodyConfidenceReasons = computed(() => {
  const reasons: string[] = [];

  if (cameraStatus.value !== "ready") {
    reasons.push("摄像头未就绪");
  }

  if (poseModelStatus.value !== "ready") {
    reasons.push("姿态模型未就绪");
  }

  if (poseStatus.value !== "detected") {
    reasons.push("未稳定检测到上肢姿态");
  }

  if (upperBodyJointCompleteness.value < 45) {
    reasons.push("肩肘腕或手部关键点不足");
  } else if (upperBodyJointCompleteness.value >= 75) {
    reasons.push("上肢关键点较完整");
  }

  if (upperBodyMotionFeatures.value.activeUpperBodyMs < 3000) {
    reasons.push("有效活动时间较短");
  }

  if (upperBodyActivityScore.value < 15) {
    reasons.push("上肢运动量较低");
  } else if (upperBodyActivityScore.value >= 35) {
    reasons.push("上肢运动量稳定");
  }

  if (upperBodyMotionFeatures.value.movementAmplitude < 0.04) {
    reasons.push("运动幅度较小");
  }

  return reasons.length ? reasons : ["等待上肢活动数据"];
});

const upperBodyConfidenceSummary = computed(() => {
  return upperBodyConfidenceReasons.value.slice(0, 3).join("；");
});

const waveConfidence = computed<ConfidenceLevel>(() => {
  if (
    cameraStatus.value !== "ready" ||
    modelStatus.value !== "ready" ||
    poseModelStatus.value !== "ready" ||
    handCount.value <= 0 ||
    waveCount.value <= 0
  ) {
    return "低";
  }

  if (
    waveCount.value >= 3 &&
    waveElapsedMs.value >= 10_000 &&
    waveCycleGestureFrames.value >= 8
  ) {
    return "高";
  }

  return "中";
});

const waveConfidenceReasons = computed(() => {
  const reasons: string[] = [];

  if (cameraStatus.value !== "ready") {
    reasons.push("摄像头未就绪");
  }

  if (modelStatus.value !== "ready") {
    reasons.push("手部模型未就绪");
  }

  if (poseModelStatus.value !== "ready") {
    reasons.push("姿态模型未就绪");
  }

  if (handCount.value <= 0) {
    reasons.push("未检测到手部");
  }

  if (waveCount.value <= 0) {
    reasons.push("尚无有效挥手周期");
  }

  if (waveElapsedMs.value > 0 && waveElapsedMs.value < 3000) {
    reasons.push("记录时间较短");
  }

  if (currentGesture.value === "挥手") {
    reasons.push("当前稳定识别为挥手");
  } else if (waveRuntimeStatus.value === "recording") {
    reasons.push(`当前更像${currentGesture.value}`);
  }

  if (waveCycleGestureFrames.value >= 8) {
    reasons.push("本段挥手帧较稳定");
  }

  if (waveCount.value >= 3) {
    reasons.push("已有多个挥手周期");
  }

  return reasons.length ? reasons : ["等待开始记录"];
});

const waveConfidenceSummary = computed(() => {
  return waveConfidenceReasons.value.slice(0, 3).join("；");
});

function updateWeight(event: Event) {
  const value = Number((event.target as HTMLInputElement).value);

  if (Number.isFinite(value) && value >= 20 && value <= 200) {
    weightKg.value = value;
  }
}

const waveRuntimeLabel = computed(() => {
  if (waveRuntimeStatus.value === "recording") {
    return "记录中";
  }

  if (waveRuntimeStatus.value === "ended") {
    return "已结束";
  }

  return "未开始";
});

const appRuntimeStatus = computed<AppRuntimeStatus>(() => {
  if (cameraStatus.value === "error" || modelStatus.value === "error") {
    return "error";
  }

  if (poseModelStatus.value === "error") {
    return "error";
  }

  if (
    cameraStatus.value === "loading" ||
    modelStatus.value === "loading" ||
    poseModelStatus.value === "loading"
  ) {
    return "loading";
  }

  if (
    cameraStatus.value !== "ready" ||
    modelStatus.value !== "ready" ||
    poseModelStatus.value !== "ready"
  ) {
    return "idle";
  }

  if (waveRuntimeStatus.value === "recording") {
    return "running";
  }

  if (waveRuntimeStatus.value === "ended") {
    return "paused";
  }

  return "ready";
});

const appRuntimeLabel = computed(() => {
  const labels: Record<AppRuntimeStatus, string> = {
    idle: "未开始",
    loading: "加载中",
    ready: "已就绪",
    running: "记录中",
    paused: "已结束",
    error: "出错",
  };

  return labels[appRuntimeStatus.value];
});

const appRuntimeReason = computed(() => {
  if (cameraStatus.value === "error") {
    return cameraMessage.value;
  }

  if (modelStatus.value === "error") {
    return modelMessage.value;
  }

  if (poseModelStatus.value === "error") {
    return poseModelMessage.value;
  }

  if (cameraStatus.value !== "ready") {
    return cameraMessage.value;
  }

  if (modelStatus.value !== "ready") {
    return modelMessage.value;
  }

  if (poseModelStatus.value !== "ready") {
    return poseModelMessage.value;
  }

  if (waveRuntimeStatus.value === "recording") {
    return "正在记录挥手周期。";
  }

  if (waveRuntimeStatus.value === "ended") {
    return "记录已结束，可继续开始或重置。";
  }

  return handCount.value > 0 ? "手部已入镜，可以开始记录。" : "请举起手进入画面。";
});

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

function formatElapsedTime(ms: number) {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function resetWaveTurnTracker() {
  lastWaveDirection = null;
  lastWaveTurnAt = 0;
  waveLowSignalFrames = 0;
}

function updateWaveElapsed(now: number) {
  if (waveRuntimeStatus.value !== "recording") {
    return;
  }

  waveElapsedMs.value = waveAccumulatedMs + (now - waveStartedAt);
}

function startWaveCounter() {
  if (waveRuntimeStatus.value === "recording") {
    return;
  }

  waveRuntimeStatus.value = "recording";
  waveStartedAt = performance.now();
  waveCycleGestureFrames.value = 0;
  waveCycleTurnCount.value = 0;
  resetWaveTurnTracker();
  waveCounterReason.value = "正在记录：连续左右往返挥手会自动累计次数。";
}

function finishWaveCounter() {
  if (waveRuntimeStatus.value !== "recording") {
    return;
  }

  const now = performance.now();
  waveAccumulatedMs += now - waveStartedAt;
  waveElapsedMs.value = waveAccumulatedMs;
  waveRuntimeStatus.value = "ended";
  waveCounterReason.value = "已结束记录。";
  resetWaveTurnTracker();
}

function resetWaveCounter() {
  waveRuntimeStatus.value = "idle";
  waveCount.value = 0;
  waveElapsedMs.value = 0;
  waveAccumulatedMs = 0;
  waveStartedAt = 0;
  lastWaveCountedAtText.value = "无";
  waveCycleGestureFrames.value = 0;
  waveCycleTurnCount.value = 0;
  waveCounterReason.value = "点击开始后连续挥手，结束只停止记录。";
  resetWaveTurnTracker();
}

function getWaveDirection(sample: HandMotionSample) {
  const previous = handMotionHistory[handMotionHistory.length - 2];

  if (!previous) {
    return null;
  }

  const delta = sample.centerX - previous.centerX;

  if (Math.abs(delta) < WAVE_DIRECTION_DELTA) {
    return null;
  }

  return delta > 0 ? "right" : "left";
}

function updateWaveCounter(
  now: number,
  signal: WaveCounterSignal,
  sample: HandMotionSample,
) {
  if (waveRuntimeStatus.value !== "recording") {
    return;
  }

  if (!signal.active) {
    waveLowSignalFrames += 1;
    waveCounterReason.value =
      waveLowSignalFrames <= WAVE_SIGNAL_GRACE_FRAMES
        ? `本周期记录中：${signal.reason}，短暂保留挥手上下文。`
        : `本周期记录中：${signal.reason}，等待重新出现横向摆动。`;

    if (waveLowSignalFrames > WAVE_SIGNAL_GRACE_FRAMES) {
      resetWaveTurnTracker();
    }

    return;
  }

  waveLowSignalFrames = 0;
  waveCycleGestureFrames.value += 1;
  waveCounterSignalReason.value = `${signal.reason}，计数信号 ${signal.score}`;

  const direction = getWaveDirection(sample);

  if (!direction) {
    waveCounterReason.value = "本周期记录中：挥手幅度较小，等待明确横向运动。";
    return;
  }

  if (!lastWaveDirection) {
    lastWaveDirection = direction;
    waveCounterReason.value = "本周期记录中：已捕捉挥手方向，等待换向。";
    return;
  }

  if (direction === lastWaveDirection) {
    return;
  }

  lastWaveDirection = direction;

  if (now - lastWaveTurnAt < WAVE_TURN_COOLDOWN_MS) {
    waveCounterReason.value = "本周期记录中：换向过快，可能是残影或抖动。";
    return;
  }

  lastWaveTurnAt = now;
  waveCycleTurnCount.value += 1;
  waveCounterReason.value = `记录中：已捕捉 ${waveCycleTurnCount.value}/${WAVE_TURNS_PER_COUNT} 次换向。`;

  if (waveCycleTurnCount.value >= WAVE_TURNS_PER_COUNT) {
    waveCount.value += 1;
    waveCycleTurnCount.value = 0;
    lastWaveCountedAtText.value = formatElapsedTime(waveElapsedMs.value);
    waveCounterReason.value = "完成一次左右往返，挥手次数 +1。";
  }
}

function stopRenderLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function stopCamera() {
  stopRenderLoop();
  finishWaveCounter();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;

  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }

  handCount.value = 0;
  latestHands = [];
  latestPoseLandmarks = null;
  handMotionHistory.length = 0;
  upperBodyMotionHistory.length = 0;
  poseStatus.value = "idle";
  upperBodyJointState.value = "等待姿态关键点";
  upperBodyMotionFeatures.value = {
    movementDistance: 0,
    movementSpeed: 0,
    movementAmplitude: 0,
    activeUpperBodyMs: 0,
  };
  upperBodyActivityScore.value = 0;
  upperBodyIntensityLevel.value = "静止";
  lastActiveUpperBodyScore.value = 0;
  lastActiveUpperBodyLevel.value = "静止";
  lastActiveUpperBodyAt.value = 0;
  upperBodyBasicActions.value = [];
  currentGesture.value = "等待检测";
  motionDirection.value = "等待检测";
  instantMotion.value = "等待检测";
  handScaleTrend.value = "等待检测";
  gestureReason.value = "等待手部入镜";
}

function disposeModel() {
  handLandmarker?.close();
  handLandmarker = null;
}

function disposePoseModel() {
  poseLandmarker?.close();
  poseLandmarker = null;
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

async function loadPoseModel() {
  if (poseLandmarker || poseModelStatus.value === "loading") {
    return;
  }

  poseModelStatus.value = "loading";
  poseModelMessage.value = "正在加载 MediaPipe Pose Landmarker";

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
    poseModelStatus.value = "ready";
    poseModelMessage.value = "姿态模型已就绪。";
  } catch (error) {
    disposePoseModel();
    poseModelStatus.value = "error";
    poseModelMessage.value =
      error instanceof Error ? error.message : "姿态模型加载失败。";
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

function getHandMotionSample(hand: LandmarkPoint[], now: number): HandMotionSample {
  const xs = hand.map((point) => point.x);
  const ys = hand.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const fingerPairs = [
    [8, 6],
    [12, 10],
    [16, 14],
    [20, 18],
  ] as const;
  const fingerCurl =
    fingerPairs.reduce((sum, [tipIndex, jointIndex]) => {
      const tip = hand[tipIndex];
      const joint = hand[jointIndex];
      return sum + (tip && joint ? tip.y - joint.y : 0);
    }, 0) / fingerPairs.length;

  return {
    time: now,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    scale: Math.max(maxX - minX, maxY - minY),
    fingerCurl,
  };
}

function getVisiblePosePoint(landmarks: LandmarkPoint[] | null, index: number) {
  const point = landmarks?.[index];

  if (!point || (point.visibility ?? 1) < MIN_POSE_VISIBILITY) {
    return null;
  }

  return point;
}

function getHandCenterPoint(hand: LandmarkPoint[] | undefined): LandmarkPoint | null {
  if (!hand?.length) {
    return null;
  }

  const sample = getHandMotionSample(hand, performance.now());
  return { x: sample.centerX, y: sample.centerY, z: 0 };
}

function getDistance(pointA: LandmarkPoint | null, pointB: LandmarkPoint | null) {
  if (!pointA || !pointB) {
    return 0;
  }

  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
}

function getSideJointDistance(
  previous: UpperBodySideSample,
  current: UpperBodySideSample,
) {
  const pairs = [
    [previous.shoulder, current.shoulder],
    [previous.elbow, current.elbow],
    [previous.wrist, current.wrist],
    [previous.handCenter, current.handCenter],
  ] as const;

  const distances = pairs
    .map(([from, to]) => getDistance(from, to))
    .filter((distance) => distance > 0);

  return distances.length
    ? distances.reduce((sum, distance) => sum + distance, 0) / distances.length
    : 0;
}

function getSideAmplitude(samples: UpperBodySideSample[], key: keyof UpperBodySideSample) {
  const points = samples.map((sample) => sample[key]).filter((point): point is LandmarkPoint => Boolean(point));

  if (points.length < 2) {
    return 0;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);

  return Math.hypot(getRange(xs), getRange(ys));
}

function getRaisedArmScore(side: UpperBodySideSample) {
  const handOrWrist = side.wrist ?? side.handCenter;

  if (!side.shoulder || !handOrWrist) {
    return 0;
  }

  const verticalLift = side.shoulder.y - handOrWrist.y;

  return scoreFromRange(verticalLift, 0.02, 0.22);
}

function getUpperBodyIntensityLevel(score: number): UpperBodyIntensityLevel {
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

function updateUpperBodyActivityClassification(sample: UpperBodyMotionSample, now: number) {
  const features = upperBodyMotionFeatures.value;
  const distanceScore = scoreFromRange(features.movementDistance, 0.015, 0.12);
  const speedScore = scoreFromRange(features.movementSpeed, 0.12, 0.8);
  const amplitudeScore = scoreFromRange(features.movementAmplitude, 0.04, 0.28);
  const rawActivityScore = clampScore(
    distanceScore * 0.35 + speedScore * 0.35 + amplitudeScore * 0.3,
  );
  let activityScore = rawActivityScore;
  let intensityLevel = getUpperBodyIntensityLevel(rawActivityScore);

  if (intensityLevel !== "静止") {
    lastActiveUpperBodyScore.value = rawActivityScore;
    lastActiveUpperBodyLevel.value = intensityLevel;
    lastActiveUpperBodyAt.value = now;
  } else if (
    lastActiveUpperBodyLevel.value !== "静止" &&
    now - lastActiveUpperBodyAt.value <= UPPER_BODY_INTENSITY_HOLD_MS
  ) {
    activityScore = Math.max(rawActivityScore, Math.min(lastActiveUpperBodyScore.value, 18));
    intensityLevel = "低强度";
  }

  const leftRaisedScore = getRaisedArmScore(sample.left);
  const rightRaisedScore = getRaisedArmScore(sample.right);
  const candidates: BasicActionCandidate[] = [
    {
      label: "手向前移动",
      score: instantMotion.value === "手向前移动中" ? motionEvidence.value.depthMoveScore : 0,
      reason: "手部尺度变大，表示向镜头方向移动",
    },
    {
      label: "手向后移动",
      score: instantMotion.value === "手向后收回中" ? motionEvidence.value.depthMoveScore : 0,
      reason: "手部尺度变小，表示从镜头方向收回",
    },
    {
      label: "左抬胳膊",
      score: leftRaisedScore,
      reason: "左手/腕高于左肩",
    },
    {
      label: "右抬胳膊",
      score: rightRaisedScore,
      reason: "右手/腕高于右肩",
    },
    {
      label: "双臂活动",
      score: Math.min(100, Math.round((leftRaisedScore + rightRaisedScore) / 2)),
      reason: "左右上肢同时出现抬起证据",
    },
  ]
    .filter((candidate) => candidate.score >= 20)
    .sort((left, right) => right.score - left.score);

  upperBodyActivityScore.value = activityScore;
  upperBodyIntensityLevel.value = intensityLevel;
  upperBodyBasicActions.value = candidates;
}

function getUpperBodyMotionSample(now: number): UpperBodyMotionSample {
  const leftHand = latestHands[0];
  const rightHand = latestHands[1];

  return {
    time: now,
    left: {
      shoulder: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftShoulder),
      elbow: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftElbow),
      wrist: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftWrist),
      handCenter: getHandCenterPoint(leftHand),
    },
    right: {
      shoulder: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightShoulder),
      elbow: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightElbow),
      wrist: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightWrist),
      handCenter: getHandCenterPoint(rightHand),
    },
  };
}

function updateUpperBodyMotionFeatures(now: number) {
  const sample = getUpperBodyMotionSample(now);

  upperBodyMotionHistory.push(sample);

  while (
    upperBodyMotionHistory.length &&
    now - upperBodyMotionHistory[0].time > UPPER_BODY_HISTORY_MS
  ) {
    upperBodyMotionHistory.shift();
  }

  const leftJointCount = [
    sample.left.shoulder,
    sample.left.elbow,
    sample.left.wrist,
    sample.left.handCenter,
  ].filter(Boolean).length;
  const rightJointCount = [
    sample.right.shoulder,
    sample.right.elbow,
    sample.right.wrist,
    sample.right.handCenter,
  ].filter(Boolean).length;
  upperBodyJointState.value = `左 ${leftJointCount}/4，右 ${rightJointCount}/4`;

  if (upperBodyMotionHistory.length < 2) {
    upperBodyMotionFeatures.value = {
      ...upperBodyMotionFeatures.value,
      movementDistance: 0,
      movementSpeed: 0,
      movementAmplitude: 0,
    };
    return;
  }

  const previous = upperBodyMotionHistory[upperBodyMotionHistory.length - 2];
  const elapsedSeconds = Math.max((sample.time - previous.time) / 1000, 0.001);
  const leftDistance = getSideJointDistance(previous.left, sample.left);
  const rightDistance = getSideJointDistance(previous.right, sample.right);
  const movementDistance = leftDistance + rightDistance;
  const movementSpeed = movementDistance / elapsedSeconds;
  const leftSamples = upperBodyMotionHistory.map((item) => item.left);
  const rightSamples = upperBodyMotionHistory.map((item) => item.right);
  const movementAmplitude = Math.max(
    getSideAmplitude(leftSamples, "handCenter"),
    getSideAmplitude(leftSamples, "wrist"),
    getSideAmplitude(leftSamples, "elbow"),
    getSideAmplitude(rightSamples, "handCenter"),
    getSideAmplitude(rightSamples, "wrist"),
    getSideAmplitude(rightSamples, "elbow"),
  );
  const nextActiveMs =
    movementDistance >= UPPER_BODY_ACTIVE_DISTANCE_THRESHOLD
      ? upperBodyMotionFeatures.value.activeUpperBodyMs + elapsedSeconds * 1000
      : upperBodyMotionFeatures.value.activeUpperBodyMs;

  upperBodyMotionFeatures.value = {
    movementDistance,
    movementSpeed,
    movementAmplitude,
    activeUpperBodyMs: nextActiveMs,
  };
  updateUpperBodyActivityClassification(sample, now);
}

function countDirectionChanges(values: number[]) {
  let lastSign = 0;
  let changes = 0;

  for (let index = 1; index < values.length; index += 1) {
    const delta = values[index] - values[index - 1];
    const sign =
      Math.abs(delta) < MIN_MOTION_DELTA ? 0 : delta > 0 ? 1 : -1;

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

function getRange(values: number[]) {
  return Math.max(...values) - Math.min(...values);
}

function getInstantMotionLabel(history: HandMotionSample[]) {
  if (history.length < 3) {
    return "采样中";
  }

  const recent = history.slice(-4);
  const first = recent[0];
  const latest = recent[recent.length - 1];
  const elapsedSeconds = Math.max((latest.time - first.time) / 1000, 0.001);
  const xVelocity = (latest.centerX - first.centerX) / elapsedSeconds;
  const yVelocity = (latest.centerY - first.centerY) / elapsedSeconds;
  const scaleVelocity = first.scale
    ? ((latest.scale - first.scale) / first.scale) / elapsedSeconds
    : 0;
  const absX = Math.abs(xVelocity);
  const absY = Math.abs(yVelocity);
  const absScale = Math.abs(scaleVelocity);

  if (absScale > 0.9 && absScale > absX * 1.6) {
    return scaleVelocity > 0 ? "手向前移动中" : "手向后收回中";
  }

  if (absX > 0.18 && absX > absY * 1.25) {
    return xVelocity > 0 ? "向画面右侧移动中" : "向画面左侧移动中";
  }

  if (absY > 0.18) {
    return yVelocity > 0 ? "向下移动中" : "向上移动中";
  }

  return "短时静止";
}

function clampScore(value: number) {
  return Math.max(0, Math.min(100, Math.round(value)));
}

function scoreFromRange(value: number, low: number, high: number) {
  if (value <= low) {
    return 0;
  }

  if (value >= high) {
    return 100;
  }

  return clampScore(((value - low) / (high - low)) * 100);
}

function resetMotionEvidence(reason: string) {
  motionEvidence.value = { ...EMPTY_MOTION_EVIDENCE };
  gestureCandidates.value = [];
  waveCounterSignalReason.value = reason;
}

function getTopGestureCandidate(candidates: GestureCandidate[]) {
  return candidates[0] ?? { label: "静止" as GestureLabel, score: 0, reason: "暂无候选动作" };
}

function createGestureCandidate(
  label: GestureLabel,
  score: number,
  reason: string,
): GestureCandidate {
  return { label, score, reason };
}

function buildWaveCounterSignal(evidence: MotionEvidence): WaveCounterSignal {
  const score = Math.round(
    evidence.waveCycleScore * 0.55 +
      evidence.horizontalSwingScore * 0.3 +
      evidence.stabilityScore * 0.15,
  );
  const active = score >= WAVE_SIGNAL_MIN_SCORE;
  const reason = active
    ? `挥手证据稳定：周期 ${evidence.waveCycleScore}，横向 ${evidence.horizontalSwingScore}`
    : `挥手证据不足：周期 ${evidence.waveCycleScore}，横向 ${evidence.horizontalSwingScore}`;

  return { active, score, reason };
}

function updateGestureAnalysis(now: number, hands: LandmarkPoint[][]) {
  if (!hands.length) {
    handMotionHistory.length = 0;
    currentGesture.value = "等待检测";
    motionDirection.value = "等待检测";
    instantMotion.value = "等待检测";
    handScaleTrend.value = "等待检测";
    gestureReason.value = "未检测到手部";
    resetMotionEvidence("未检测到手部");
    resetWaveTurnTracker();
    return;
  }

  const primaryHand = [...hands].sort((left, right) => {
    const leftScale = getHandMotionSample(left, now).scale;
    const rightScale = getHandMotionSample(right, now).scale;
    return rightScale - leftScale;
  })[0];
  const sample = getHandMotionSample(primaryHand, now);

  handMotionHistory.push(sample);

  while (
    handMotionHistory.length &&
    now - handMotionHistory[0].time > MOTION_HISTORY_MS
  ) {
    handMotionHistory.shift();
  }

  if (handMotionHistory.length < 6) {
    currentGesture.value = "静止";
    motionDirection.value = "采样中";
    instantMotion.value = getInstantMotionLabel(handMotionHistory);
    handScaleTrend.value = "采样中";
    gestureReason.value = "等待更多手部运动帧";
    resetMotionEvidence("采样帧不足");
    updateWaveCounter(now, { active: false, score: 0, reason: "采样帧不足" }, sample);
    return;
  }

  const first = handMotionHistory[0];
  const xValues = handMotionHistory.map((item) => item.centerX);
  const yValues = handMotionHistory.map((item) => item.centerY);
  const scaleValues = handMotionHistory.map((item) => item.scale);
  const curlValues = handMotionHistory.map((item) => item.fingerCurl);
  const xRange = getRange(xValues);
  const yRange = getRange(yValues);
  const scaleRange = getRange(scaleValues);
  const curlRange = getRange(curlValues);
  const xDirectionChanges = countDirectionChanges(xValues);
  const scaleChange = first.scale
    ? (sample.scale - first.scale) / first.scale
    : 0;
  const xDelta = sample.centerX - first.centerX;
  const yDelta = sample.centerY - first.centerY;

  motionDirection.value =
    Math.abs(xDelta) > Math.abs(yDelta)
      ? xDelta > 0
        ? "向画面右侧"
        : "向画面左侧"
      : yDelta > 0
        ? "向下"
        : "向上";
  instantMotion.value = getInstantMotionLabel(handMotionHistory);
  handScaleTrend.value =
    scaleChange > 0.12
      ? "变大"
      : scaleChange < -0.12
        ? "变小"
        : "稳定";

  const horizontalSwingScore = clampScore(
    scoreFromRange(xRange, 0.06, 0.18) * 0.65 +
      Math.min(xDirectionChanges, 4) * 8,
  );
  const depthMoveScore = scoreFromRange(Math.abs(scaleChange), 0.08, 0.28);
  const fingerCurlScore = scoreFromRange(curlRange, 0.04, 0.12);
  const waveCycleScore = clampScore(
    horizontalSwingScore * 0.65 + scoreFromRange(xDirectionChanges, 1, 3) * 0.35,
  );
  const stabilityScore = clampScore(scoreFromRange(handMotionHistory.length, 6, 14));
  const nextEvidence: MotionEvidence = {
    horizontalSwingScore,
    depthMoveScore,
    fingerCurlScore,
    waveCycleScore,
    stabilityScore,
  };
  const forwardScore = scaleChange > 0 ? depthMoveScore : 0;
  const backwardScore = scaleChange < 0 ? depthMoveScore : 0;
  const punchScore =
    scaleChange > 0 && sample.scale > 0.12
      ? clampScore(depthMoveScore + scoreFromRange(sample.scale, 0.12, 0.2) * 0.25)
      : 0;
  const waveScore = waveCycleScore;
  const beckonScore =
    xRange < 0.1 && scaleRange < 0.12 ? fingerCurlScore : clampScore(fingerCurlScore * 0.45);
  const stillScore = clampScore(
    100 -
      Math.max(horizontalSwingScore, depthMoveScore, fingerCurlScore, waveCycleScore),
  );
  const forwardLabel: GestureLabel = punchScore >= forwardScore ? "出拳" : "手向前";
  const curlLabel: GestureLabel = beckonScore > 65 ? "招手" : "握拳";
  const candidates = [
    createGestureCandidate(
      "挥手",
      waveScore,
      `横向 ${xRange.toFixed(2)}，换向 ${xDirectionChanges} 次`,
    ),
    createGestureCandidate(
      forwardLabel,
      Math.max(punchScore, forwardScore),
      `手部尺度变大 ${(Math.max(scaleChange, 0) * 100).toFixed(0)}%`,
    ),
    createGestureCandidate(
      "手向后",
      backwardScore,
      `手部尺度变小 ${(Math.max(-scaleChange, 0) * 100).toFixed(0)}%`,
    ),
    createGestureCandidate(
      curlLabel,
      beckonScore,
      `手指弯曲变化 ${curlRange.toFixed(2)}`,
    ),
    createGestureCandidate("静止", stillScore, "主要运动证据较低"),
  ]
    .filter((candidate) => candidate.score > 0)
    .sort((left, right) => right.score - left.score);

  motionEvidence.value = nextEvidence;
  gestureCandidates.value = candidates;

  const topCandidate = getTopGestureCandidate(candidates);
  currentGesture.value = topCandidate.label;
  gestureReason.value = topCandidate.reason;

  const waveSignal = buildWaveCounterSignal(nextEvidence);
  waveCounterSignalReason.value = `${waveSignal.reason}，信号 ${waveSignal.score}`;
  updateWaveCounter(now, waveSignal, sample);
}

function drawLandmarkPoint(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  point: LandmarkPoint,
  radius: number,
  fillStyle: string,
) {
  context.beginPath();
  context.fillStyle = fillStyle;
  context.strokeStyle = "rgba(15, 23, 42, 0.82)";
  context.lineWidth = 2;
  context.arc(point.x * canvas.width, point.y * canvas.height, radius, 0, Math.PI * 2);
  context.fill();
  context.stroke();
}

function drawLandmarkLine(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  from: LandmarkPoint | null,
  to: LandmarkPoint | null,
  strokeStyle: string,
  lineWidth: number,
) {
  if (!from || !to) {
    return;
  }

  context.beginPath();
  context.strokeStyle = strokeStyle;
  context.lineWidth = lineWidth;
  context.moveTo(from.x * canvas.width, from.y * canvas.height);
  context.lineTo(to.x * canvas.width, to.y * canvas.height);
  context.stroke();
}

function findNearestHandToWrist(
  poseWrist: LandmarkPoint | null,
  usedHandIndexes: Set<number>,
) {
  if (!poseWrist) {
    return null;
  }

  let nearestIndex = -1;
  let nearestDistance = Number.POSITIVE_INFINITY;

  latestHands.forEach((hand, index) => {
    const handWrist = hand[0];

    if (!handWrist || usedHandIndexes.has(index)) {
      return;
    }

    const distance = getDistance(poseWrist, handWrist);

    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });

  if (nearestIndex < 0 || nearestDistance > 0.24) {
    return null;
  }

  usedHandIndexes.add(nearestIndex);
  return latestHands[nearestIndex][0] ?? null;
}

function drawUpperBody(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  if (!latestPoseLandmarks?.length) {
    return;
  }

  const left = {
    shoulder: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftShoulder),
    elbow: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftElbow),
    wrist: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.leftWrist),
  };
  const right = {
    shoulder: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightShoulder),
    elbow: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightElbow),
    wrist: getVisiblePosePoint(latestPoseLandmarks, POSE_LANDMARK_INDEX.rightWrist),
  };
  const usedHandIndexes = new Set<number>();
  const leftHandWrist = findNearestHandToWrist(left.wrist, usedHandIndexes);
  const rightHandWrist = findNearestHandToWrist(right.wrist, usedHandIndexes);
  const jointRadius = Math.max(5, canvas.width / 220);
  const lineWidth = Math.max(5, canvas.width / 180);

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";

  drawLandmarkLine(
    context,
    canvas,
    left.shoulder,
    left.elbow,
    "rgba(20, 184, 166, 0.96)",
    lineWidth,
  );
  drawLandmarkLine(
    context,
    canvas,
    left.elbow,
    left.wrist,
    "rgba(20, 184, 166, 0.96)",
    lineWidth,
  );
  drawLandmarkLine(
    context,
    canvas,
    left.wrist,
    leftHandWrist,
    "rgba(20, 184, 166, 0.78)",
    Math.max(3, lineWidth - 1),
  );

  drawLandmarkLine(
    context,
    canvas,
    right.shoulder,
    right.elbow,
    "rgba(59, 130, 246, 0.96)",
    lineWidth,
  );
  drawLandmarkLine(
    context,
    canvas,
    right.elbow,
    right.wrist,
    "rgba(59, 130, 246, 0.96)",
    lineWidth,
  );
  drawLandmarkLine(
    context,
    canvas,
    right.wrist,
    rightHandWrist,
    "rgba(59, 130, 246, 0.78)",
    Math.max(3, lineWidth - 1),
  );

  [left.shoulder, left.elbow, left.wrist].forEach((point) => {
    if (point) {
      drawLandmarkPoint(context, canvas, point, jointRadius, "#2dd4bf");
    }
  });
  [right.shoulder, right.elbow, right.wrist].forEach((point) => {
    if (point) {
      drawLandmarkPoint(context, canvas, point, jointRadius, "#60a5fa");
    }
  });

  context.restore();
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
    (!handLandmarker && !poseLandmarker) ||
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
      lastHandedness.value =
        handResult.handednesses
          ?.flat()
          .map((item) => item.categoryName)
          .join(" / ") || "无";
      updateGestureAnalysis(now, latestHands);
    }

    if (poseLandmarker) {
      const poseResult = poseLandmarker.detectForVideo(video, now);
      latestPoseLandmarks = (poseResult.landmarks[0] as LandmarkPoint[] | undefined) ?? null;
      poseStatus.value = latestPoseLandmarks?.length ? "detected" : "missing";
      updateUpperBodyMotionFeatures(now);
    }

    updateInferenceFps(now);
  } catch (error) {
    modelStatus.value = "error";
    poseModelStatus.value = "error";
    modelMessage.value =
      error instanceof Error ? `视觉推理失败：${error.message}` : "视觉推理失败。";
    poseModelMessage.value = modelMessage.value;
    stopRenderLoop();
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
  const now = performance.now();
  updateWaveElapsed(now);
  runInference(now);
  drawUpperBody(context, canvas);
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
    cameraMessage.value = getCameraErrorMessage(error);
  }
}

onMounted(() => {
  void startCamera();
  void loadModel();
  void loadPoseModel();
});

onBeforeUnmount(() => {
  stopCamera();
  disposeModel();
  disposePoseModel();
});
</script>

<template>
  <main class="single-tool-page">
    <section class="stage-panel" aria-label="手势识别区域">
      <div class="stage-toolbar">
        <div>
          <p class="eyebrow">Gesture</p>
          <h1>上肢活动估算</h1>
        </div>
        <span class="status-pill" :class="`status-${appRuntimeStatus}`">
          {{ appRuntimeLabel }}
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
            {{ appRuntimeReason }}
          </div>
          <div
            v-if="cameraStatus === 'ready' && modelStatus === 'ready' && handCount === 0"
            class="pose-hint"
          >
            未检测到手部，请举起手并保持在画面内。
          </div>
          <div v-if="cameraStatus === 'ready'" class="motion-state-badge">
            <span>手部动作</span>
            <strong>{{ currentGesture }}</strong>
            <small>挥手 {{ waveCount }} 次 / {{ waveRuntimeLabel }}</small>
          </div>
        </div>
      </div>

      <div class="controls-row">
        <button type="button" @click="startCamera">启动摄像头</button>
        <button type="button" class="secondary" @click="stopCamera">关闭摄像头</button>
      </div>
    </section>

    <aside class="side-panel" aria-label="上肢活动估算指标">
      <section class="metric-section">
        <div class="metric-section-header">
          <span>记录设置</span>
          <small>体重和记录周期</small>
        </div>

        <label class="weight-field">
          <span>体重 kg</span>
          <input
            type="number"
            min="20"
            max="200"
            :value="weightKg"
            @input="updateWeight"
          />
        </label>

        <div class="controls-row compact-controls">
          <button type="button" @click="startWaveCounter">开始</button>
          <button type="button" class="secondary" @click="finishWaveCounter">结束</button>
          <button type="button" class="secondary" @click="resetWaveCounter">重置</button>
        </div>
      </section>

      <section class="metric-section">
        <div class="metric-section-header">
          <span>上肢活动估算</span>
          <small>通用活动强度和消耗</small>
        </div>

        <div class="metrics-grid">
          <section class="metric-card">
            <span>有效时长</span>
            <strong>{{ formatElapsedTime(upperBodyMotionFeatures.activeUpperBodyMs) }}</strong>
            <small>{{ upperBodyJointState }}</small>
          </section>
          <section class="metric-card">
            <span>活动分数</span>
            <strong>{{ upperBodyActivityScore }}</strong>
            <small>{{ upperBodyIntensityLevel }}</small>
          </section>
          <section class="metric-card">
            <span>上肢 MET</span>
            <strong>{{ currentUpperBodyMet.toFixed(1) }}</strong>
            <small>按强度档位估算</small>
          </section>
          <section class="metric-card">
            <span>上肢 cal</span>
            <strong>{{ upperBodyCalories.toFixed(2) }}</strong>
            <small>通用上肢活动粗略估算</small>
          </section>
          <section class="metric-card">
            <span>估算置信度</span>
            <strong>{{ upperBodyConfidence }}</strong>
            <small>{{ upperBodyConfidenceSummary }}</small>
          </section>
        </div>
      </section>

      <section class="metric-section">
        <div class="metric-section-header">
          <span>基本动作和运动证据</span>
          <small>动作名不是热量唯一入口</small>
        </div>

        <div class="metrics-grid">
          <section class="metric-card">
            <span>基本动作候选</span>
            <strong>{{ topUpperBodyBasicActionsText }}</strong>
            <small>基本动作不等于完整运动名</small>
          </section>
          <section class="metric-card">
            <span>当前动作</span>
            <strong>{{ currentGesture }}</strong>
            <small>{{ gestureReason }}</small>
          </section>
          <section class="metric-card">
            <span>即时运动</span>
            <strong>{{ instantMotion }}</strong>
            <small>短窗口反馈，动作名以候选状态为准</small>
          </section>
          <section class="metric-card">
            <span>候选状态</span>
            <strong>{{ topGestureCandidatesText }}</strong>
            <small>各状态独立评分，不共享 100</small>
          </section>
          <section class="metric-card">
            <span>运动证据</span>
            <strong>{{ motionEvidence.horizontalSwingScore }} / {{ motionEvidence.depthMoveScore }}</strong>
            <small>横向摆动 / 前后移动</small>
          </section>
          <section class="metric-card">
            <span>稳定证据</span>
            <strong>{{ motionEvidence.stabilityScore }}</strong>
            <small>手指弯曲 {{ motionEvidence.fingerCurlScore }}</small>
          </section>
        </div>
      </section>

      <details class="metric-section">
        <summary class="metric-section-header">
          <span>挥手示例指标</span>
          <small>Task1 保留示例</small>
        </summary>

        <div class="metrics-grid">
          <section class="metric-card">
            <span>挥手次数</span>
            <strong>{{ waveCount }}</strong>
            <small>{{ waveCounterReason }}</small>
          </section>
          <section class="metric-card">
            <span>挥手时长</span>
            <strong>{{ formatElapsedTime(waveElapsedMs) }}</strong>
            <small>{{ waveRuntimeLabel }}</small>
          </section>
          <section class="metric-card">
            <span>挥手频率</span>
            <strong>{{ waveFrequencyPerMin.toFixed(1) }}</strong>
            <small>次/分钟</small>
          </section>
          <section class="metric-card">
            <span>挥手 MET</span>
            <strong>{{ currentWaveMet.toFixed(1) }}</strong>
            <small>粗略档位</small>
          </section>
          <section class="metric-card">
            <span>挥手 cal</span>
            <strong>{{ waveCalories.toFixed(2) }}</strong>
            <small>cal，仅挥手示例粗略估算</small>
          </section>
          <section class="metric-card">
            <span>挥手置信度</span>
            <strong>{{ waveConfidence }}</strong>
            <small>{{ waveConfidenceSummary }}</small>
          </section>
          <section class="metric-card">
            <span>当前周期换向</span>
            <strong>{{ waveCycleTurnCount }}</strong>
            <small>本段挥手帧 {{ waveCycleGestureFrames }}</small>
          </section>
          <section class="metric-card">
            <span>挥手证据</span>
            <strong>{{ motionEvidence.waveCycleScore }}</strong>
            <small>{{ waveCounterSignalReason }}</small>
          </section>
        </div>
      </details>

      <details class="metric-section">
        <summary class="metric-section-header">
          <span>运动学调试</span>
          <small>位移、速度和推理状态</small>
        </summary>

        <div class="metrics-grid">
          <section class="metric-card">
            <span>上肢位移</span>
            <strong>{{ upperBodyMotionFeatures.movementDistance.toFixed(3) }}</strong>
            <small>肩肘腕手综合位移</small>
          </section>
          <section class="metric-card">
            <span>上肢速度</span>
            <strong>{{ upperBodyMotionFeatures.movementSpeed.toFixed(3) }}</strong>
            <small>最近帧归一化速度</small>
          </section>
          <section class="metric-card">
            <span>上肢幅度</span>
            <strong>{{ upperBodyMotionFeatures.movementAmplitude.toFixed(3) }}</strong>
            <small>最近窗口最大活动幅度</small>
          </section>
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
          <p>姿态模型: {{ poseModelStatus }} / {{ poseModelMessage }}</p>
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
