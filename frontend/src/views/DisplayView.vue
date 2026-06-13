<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, shallowRef } from "vue";
import "../session-ui.css";
import { useMediaPipe } from "../composables/useMediaPipe";
import { useActionWs }  from "../composables/useActionWs";
import {
  FilesetResolver,
  PoseLandmarker,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

// ---- MediaPipe constants ---------------------------------------------------

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;

// ---- Types -----------------------------------------------------------------

type ActionTab = {
  key:   string;
  label: string;
  mode:  "count" | "duration";
  icon:  string;
};

type CountState = {
  count:          number;
  countLeft?:     number;
  countRight?:    number;
  stage:          string;
  score:          number | null;
  qualityLabel?:  string;
  standardCount:  number;
  shallowCount:   number;
  lastAdvice:     string;
};

type DurationState = {
  stage:          string;
  score:          number | null;
  holdSecLeft:    number;
  holdSecRight:   number;
};

type ActionState = CountState | DurationState;

type DebugMessageType = "single_frame" | "dual_frame" | "none";

type SentFrameInfo = {
  messageType: DebugMessageType;
  payloadEstimatedBytes: number | null;
};

type FrontendDebugFrame = {
  sample_index: number;
  timestamp_ms: number;
  loop_dt_ms: number | null;
  front_infer_ms: number | null;
  side_infer_ms: number | null;
  draw_ms: number;
  send_ms: number;
  total_loop_ms: number;
  total_pose_infer_ms: number;
  message_type: DebugMessageType;
  side_enabled: boolean;
  show_side_video: boolean;
  front_ready: boolean;
  side_ready: boolean;
  front_has_pose: boolean;
  side_has_pose: boolean;
  target_interval_ms: number;
  front_video_width: number | null;
  front_video_height: number | null;
  side_video_width: number | null;
  side_video_height: number | null;
  payload_estimated_bytes: number | null;
};

type NumberStats = {
  avg: number | null;
  p50: number | null;
  p95: number | null;
  min: number | null;
  max: number | null;
};

// ---- Constants -------------------------------------------------------------

const TABS: ActionTab[] = [
  { key: "squat",           label: "深蹲",     mode: "count",    icon: "🏋️" },
  { key: "jumping_jack",    label: "开合跳",   mode: "count",    icon: "🤸" },
  { key: "lunge",           label: "弓步压腿", mode: "duration", icon: "🦵" },
  { key: "clap_under_knee", label: "胯下击掌", mode: "count",    icon: "👏" },
  { key: "high_knee",       label: "高抬腿",   mode: "count",    icon: "🏃" },
];

const STAGE_ZH: Record<string, string> = {
  stand: "站立", down: "下蹲", downing: "下蹲中", rising: "起身",
  closed: "合拢", open: "打开", opening: "打开中", closing: "回收",
  left_hold: "左腿前压", right_hold: "右腿前压",
  transition: "换腿中", idle: "待机", unknown: "--",
};

const QUALITY_ZH: Record<number, string> = {
  1.0:  "优秀",
  0.75: "良好",
  0.50: "一般",
  0.0:  "未达标",
};


// ---- State -----------------------------------------------------------------

const activeTab    = ref<string>(TABS[0].key);
const videoRef     = ref<HTMLVideoElement | null>(null);
const videoRefSide = ref<HTMLVideoElement | null>(null);
const canvasRef    = ref<HTMLCanvasElement | null>(null);
const canvasRefSide = ref<HTMLCanvasElement | null>(null);
const weightKg     = ref(70);

// Dual camera state
const sideEnabled = ref(false);          // "侧边摄像头"开关
const showSideVideo = ref(false);        // "侧边画面"开关
const frontDeviceId = ref<string | null>(null);
const sideDeviceId = ref<string | null>(null);
const availableDevices = ref<MediaDeviceInfo[]>([]);

// Dual camera MediaPipe instances
let streamFront: MediaStream | null = null;
let streamSide: MediaStream | null = null;
let poseFront = shallowRef<PoseLandmarker | null>(null);
let poseSide = shallowRef<PoseLandmarker | null>(null);
let animFrameId: number | null = null;
let lastInferenceAt = 0;
const dualCameraStatus = ref<"idle" | "loading" | "ready" | "error">("idle");

// Per-action runtime state (reset on tab switch)
const actionStates = ref<Record<string, ActionState>>({});
const sessionStart = ref<number>(Date.now());
const elapsedMs    = ref(0);

// Calories from backend
const backendMotionKcal = ref<number | null>(null);
const backendEventKcal  = ref<number | null>(null);
const backendByAction   = ref<Record<string, number>>({});
const instantKcalPerMin = ref<number | null>(null); // Direct from backend, no windowing needed

// Debug recording
const debugRecording = ref(false);
const debugData = ref<any>(null);
let frontendDebugFrames: FrontendDebugFrame[] = [];
let frontendDebugStartedAt = 0;
let frontendDebugLastLoopAt: number | null = null;

async function startDebugRecording() {
  if (debugRecording.value) return;
  debugRecording.value = true;
  debugData.value = null;
  beginFrontendDebugDiagnostics();

  try {
    const result = await ws.startDebugRecording(true);
    const frontendDiagnostics = buildFrontendDebugDiagnostics(performance.now());
    const backendDiagnostics = buildBackendDebugDiagnostics(result);
    const combined = {
      ...result,
      diagnostics: {
        frontend: frontendDiagnostics,
        backend: backendDiagnostics,
        bottleneck_hints: buildBottleneckHints(
          frontendDiagnostics.summary,
          backendDiagnostics.summary,
        ),
      },
    };
    debugData.value = combined;
    console.log("Debug recording completed:", combined);

    // Download JSON file
    const blob = new Blob([JSON.stringify(combined, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `debug_log_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    alert(`调试数据已记录，后端 ${result.count} 帧，前端 ${frontendDiagnostics.summary.sample_count} 帧数据已下载`);
  } catch (err) {
    console.error("Debug recording failed:", err);
    alert("调试记录失败：" + err);
  } finally {
    debugRecording.value = false;
  }
}

// ---- Composables -----------------------------------------------------------

const mp = useMediaPipe();
const ws = useActionWs();

// ---- Helpers ---------------------------------------------------------------

function fmt(sec: number): string {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = Math.floor(sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function qualityLabel(score: number | null): string {
  if (score === null) return "--";
  const rounded = Math.round(score * 4) / 4;
  return QUALITY_ZH[rounded] ?? "--";
}

function roundMetric(value: number | null | undefined, digits = 2): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function numberStats(values: Array<number | null | undefined>, digits = 2): NumberStats {
  const xs = values
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (!xs.length) {
    return { avg: null, p50: null, p95: null, min: null, max: null };
  }
  const pick = (q: number) => xs[Math.min(xs.length - 1, Math.floor((xs.length - 1) * q))];
  const avg = xs.reduce((sum, v) => sum + v, 0) / xs.length;
  return {
    avg: roundMetric(avg, digits),
    p50: roundMetric(pick(0.50), digits),
    p95: roundMetric(pick(0.95), digits),
    min: roundMetric(xs[0], digits),
    max: roundMetric(xs[xs.length - 1], digits),
  };
}

function ratio(count: number, total: number): number {
  return total > 0 ? roundMetric(count / total, 4)! : 0;
}

function beginFrontendDebugDiagnostics() {
  frontendDebugFrames = [];
  frontendDebugStartedAt = performance.now();
  frontendDebugLastLoopAt = null;
}

function currentVideoSummary(video: HTMLVideoElement | null, stream: MediaStream | null) {
  const settings = stream?.getVideoTracks()[0]?.getSettings();
  return {
    video_width: video?.videoWidth || null,
    video_height: video?.videoHeight || null,
    track_width: settings?.width ?? null,
    track_height: settings?.height ?? null,
    track_frame_rate: settings?.frameRate ?? null,
    device_id: settings?.deviceId ?? null,
  };
}

function estimateJsonBytes(payload: unknown): number | null {
  try {
    const text = JSON.stringify(payload);
    if (typeof TextEncoder !== "undefined") {
      return new TextEncoder().encode(text).length;
    }
    return text.length;
  } catch {
    return null;
  }
}

function recordFrontendDebugFrame(frame: Omit<FrontendDebugFrame, "sample_index">) {
  if (!debugRecording.value || frontendDebugStartedAt <= 0) return;
  frontendDebugFrames.push({
    sample_index: frontendDebugFrames.length,
    ...frame,
  });
}

function summarizeFrontendBottleneck(frames: FrontendDebugFrame[]): string {
  if (!frames.length) return "no_frontend_samples";
  const totalInferAvg = numberStats(frames.map(f => f.total_pose_infer_ms)).avg ?? 0;
  const drawAvg = numberStats(frames.map(f => f.draw_ms)).avg ?? 0;
  const sendAvg = numberStats(frames.map(f => f.send_ms)).avg ?? 0;
  const noSendRatio = ratio(frames.filter(f => f.message_type === "none").length, frames.length);

  if (totalInferAvg >= MIN_INFERENCE_INTERVAL_MS * 0.85) return "frontend_pose_inference_near_budget";
  if (drawAvg >= 10) return "frontend_canvas_draw_cost_high";
  if (sendAvg >= 8) return "frontend_websocket_send_cost_high";
  if (noSendRatio >= 0.30) return "pose_missing_often";
  return "frontend_within_budget";
}

function buildFrontendDebugDiagnostics(endAt: number) {
  const frames = frontendDebugFrames.slice();
  const durationMs = Math.max(0, endAt - frontendDebugStartedAt);
  const sentSingle = frames.filter(f => f.message_type === "single_frame").length;
  const sentDual = frames.filter(f => f.message_type === "dual_frame").length;
  const sentNone = frames.filter(f => f.message_type === "none").length;

  return {
    summary: {
      sample_count: frames.length,
      duration_ms: roundMetric(durationMs),
      loop_fps: durationMs > 0 ? roundMetric(frames.length * 1000 / durationMs) : null,
      target_inference_fps: TARGET_INFERENCE_FPS,
      target_interval_ms: roundMetric(MIN_INFERENCE_INTERVAL_MS),
      loop_dt_ms: numberStats(frames.map(f => f.loop_dt_ms)),
      front_infer_ms: numberStats(frames.map(f => f.front_infer_ms)),
      side_infer_ms: numberStats(frames.map(f => f.side_infer_ms)),
      total_pose_infer_ms: numberStats(frames.map(f => f.total_pose_infer_ms)),
      draw_ms: numberStats(frames.map(f => f.draw_ms)),
      send_ms: numberStats(frames.map(f => f.send_ms)),
      total_loop_ms: numberStats(frames.map(f => f.total_loop_ms)),
      payload_estimated_bytes: numberStats(frames.map(f => f.payload_estimated_bytes), 0),
      front_pose_ratio: ratio(frames.filter(f => f.front_has_pose).length, frames.length),
      side_pose_ratio: ratio(frames.filter(f => f.side_has_pose).length, frames.length),
      side_enabled_ratio: ratio(frames.filter(f => f.side_enabled).length, frames.length),
      sent_single_count: sentSingle,
      sent_dual_count: sentDual,
      sent_none_count: sentNone,
      frontend_bottleneck_guess: summarizeFrontendBottleneck(frames),
      front_camera: currentVideoSummary(videoRef.value, streamFront),
      side_camera: currentVideoSummary(videoRefSide.value, streamSide),
      user_agent: navigator.userAgent,
    },
    frames,
  };
}

function buildBackendDebugDiagnostics(result: any) {
  const rows = Array.isArray(result?.data) ? result.data : [];
  return {
    summary: {
      sample_count: rows.length,
      fps: numberStats(rows.map((r: any) => r.fps)),
      frame_dt_ms: numberStats(rows.map((r: any) => r.frame_dt_ms)),
      server_receive_dt_ms: numberStats(rows.map((r: any) => r.server_receive_dt_ms)),
      backend_process_ms: numberStats(rows.map((r: any) => r.backend_process_ms)),
      core_vis: numberStats(rows.map((r: any) => r.core_vis), 3),
      low_visibility_count: rows.filter((r: any) => r.skipped_reason === "low_core_visibility").length,
    },
  };
}

function buildBottleneckHints(frontend: any, backend: any): string[] {
  const hints: string[] = [];
  const loopFps = frontend.loop_fps ?? 0;
  const inferAvg = frontend.total_pose_infer_ms?.avg ?? 0;
  const sideAvg = frontend.side_infer_ms?.avg ?? 0;
  const backendAvg = backend.backend_process_ms?.avg ?? 0;
  const receiveAvg = backend.server_receive_dt_ms?.avg ?? 0;

  if (loopFps > 0 && loopFps < TARGET_INFERENCE_FPS * 0.75) {
    hints.push("前端实际推理 FPS 明显低于目标值，优先看 frontend.total_pose_infer_ms。");
  }
  if (inferAvg >= MIN_INFERENCE_INTERVAL_MS * 0.85) {
    hints.push("浏览器姿态推理耗时接近单帧预算，双摄顺序 detectForVideo 很可能是主要瓶颈。");
  }
  if (sideAvg >= 15) {
    hints.push("侧摄推理耗时不低，可以考虑侧摄降频、错峰推理，或动作识别只用正面。");
  }
  if (backendAvg >= 15) {
    hints.push("后端单帧处理耗时偏高，服务器 CPU 或 Python 处理链路需要进一步排查。");
  }
  if (receiveAvg >= 80 && inferAvg < MIN_INFERENCE_INTERVAL_MS * 0.5) {
    hints.push("前端推理不慢但后端收到帧间隔大，需检查 WebSocket 发送、浏览器主线程或网络。");
  }
  if (!hints.length) {
    hints.push("本次摘要未看到单个明显瓶颈，建议对比单摄/双摄两份日志的 frontend 与 backend 摘要。");
  }
  return hints;
}

// ---- Active tab data -------------------------------------------------------

const activeTabDef = computed(() => TABS.find(t => t.key === activeTab.value)!);

const activeState = computed<ActionState | null>(
  () => actionStates.value[activeTab.value] ?? null,
);

const currentCalories = computed(() =>
  backendMotionKcal.value !== null
    ? backendMotionKcal.value.toFixed(2)
    : "--"
);

const activeElapsedSec = computed(() => elapsedMs.value / 1000);

// ---- Backend result handling -----------------------------------------------

watch(ws.actions, (newActions) => {
  for (const a of newActions) {
    if (a.name === "lunge") {
      actionStates.value["lunge"] = {
        stage:        a.stage,
        score:        a.score,
        holdSecLeft:  (a as any).hold_sec_left  ?? 0,
        holdSecRight: (a as any).hold_sec_right ?? 0,
      };
    } else if (a.name === "clap_under_knee") {
      const prev = actionStates.value["clap_under_knee"] as CountState | undefined;
      actionStates.value["clap_under_knee"] = {
        count:         a.count,
        countLeft:     (a as any).count_left  ?? 0,
        countRight:    (a as any).count_right ?? 0,
        stage:         a.stage,
        score:         a.score,
        standardCount: prev?.standardCount ?? 0,
        shallowCount:  prev?.shallowCount  ?? 0,
        lastAdvice:    prev?.lastAdvice    ?? "",
      };
    } else if (a.name === "high_knee") {
      const prev = actionStates.value["high_knee"] as CountState | undefined;
      actionStates.value["high_knee"] = {
        count:         a.count,
        countLeft:     (a as any).count_left  ?? 0,
        countRight:    (a as any).count_right ?? 0,
        stage:         a.stage,
        score:         a.score,
        standardCount: prev?.standardCount ?? 0,
        shallowCount:  prev?.shallowCount  ?? 0,
        lastAdvice:    prev?.lastAdvice    ?? "",
      };
    } else {
      const prev = actionStates.value[a.name] as CountState | undefined;
      actionStates.value[a.name] = {
        count:         a.count,
        stage:         a.stage,
        score:         a.score,
        standardCount: prev?.standardCount ?? 0,
        shallowCount:  prev?.shallowCount  ?? 0,
        lastAdvice:    prev?.lastAdvice    ?? "",
      };
    }
  }
});

// Watch count events — update quality counts and advice per action
watch(ws.countEvents, (events) => {
  for (const ev of events) {
    const state = actionStates.value[ev.action] as CountState | undefined;
    if (!state) continue;
    if (ev.quality === "standard") {
      state.standardCount = (state.standardCount ?? 0) + 1;
    } else if (ev.quality === "shallow") {
      state.shallowCount = (state.shallowCount ?? 0) + 1;
      if (ev.advice) state.lastAdvice = ev.advice;
    }
  }
});

// Watch calorie summary from backend — update instant window
watch(ws.calorieSummary, (summary) => {
  if (!summary) return;
  const motionKcal = summary.motion_kcal ?? summary.total_kcal ?? null;
  backendMotionKcal.value = motionKcal;
  backendEventKcal.value  = summary.event_kcal  ?? null;
  backendByAction.value   = summary.by_action   ?? {};
  instantKcalPerMin.value = summary.instant_kcal_per_min ?? null;
});

// Send weight config whenever it changes (debounced via watch)
watch(weightKg, (w) => {
  ws.sendConfig(w);
}, { immediate: false });

// Send config when WS connects
watch(ws.status, (s) => {
  if (s === "connected") ws.sendConfig(weightKg.value);
});

// ---- Tab switching — reset right-side state --------------------------------

function switchTab(key: string) {
  activeTab.value      = key;
  sessionStart.value   = Date.now();
  elapsedMs.value      = 0;
  actionStates.value[key] = key === "lunge"
    ? { stage: "idle", score: null, holdSecLeft: 0, holdSecRight: 0 }
    : { count: 0, stage: "unknown", score: null, standardCount: 0, shallowCount: 0, lastAdvice: "" };
  ws.disconnect();
  setTimeout(() => ws.connect(), 200);
}

// ---- Dual camera controls --------------------------------------------------

function swapCameras() {
  const tmp = frontDeviceId.value;
  frontDeviceId.value = sideDeviceId.value;
  sideDeviceId.value = tmp;
  // Restart cameras with swapped device IDs
  stopDualCamera();
  setTimeout(() => startDualCamera(), 100);
}

// Enumerate camera devices and populate dropdown
async function enumerateDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableDevices.value = devices.filter(d => d.kind === "videoinput");
  } catch (err) {
    console.error("Failed to enumerate devices:", err);
  }
}

// Load MediaPipe Pose models
async function loadPoseLandmarkers() {
  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

    poseFront.value = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    if (sideEnabled.value) {
      poseSide.value = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }
  } catch (err) {
    console.error("Failed to load PoseLandmarker:", err);
    dualCameraStatus.value = "error";
  }
}

// Start dual camera streams
async function startDualCamera() {
  if (!videoRef.value) return;

  dualCameraStatus.value = "loading";

  try {
    // Start front camera
    streamFront = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: frontDeviceId.value ? { exact: frontDeviceId.value } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });
    videoRef.value.srcObject = streamFront;
    await videoRef.value.play();

    // Start side camera (if enabled)
    if (sideEnabled.value && videoRefSide.value && sideDeviceId.value) {
      streamSide = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: sideDeviceId.value },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      videoRefSide.value.srcObject = streamSide;
      await videoRefSide.value.play();
    }

    // Load Pose models
    await loadPoseLandmarkers();

    dualCameraStatus.value = "ready";

    // Start inference loop
    if (animFrameId !== null) cancelAnimationFrame(animFrameId);
    animFrameId = requestAnimationFrame(dualInferenceLoop);

  } catch (err) {
    console.error("Failed to start dual camera:", err);
    dualCameraStatus.value = "error";
  }
}

// Stop dual camera streams
function stopDualCamera() {
  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  streamFront?.getTracks().forEach(t => t.stop());
  streamFront = null;

  streamSide?.getTracks().forEach(t => t.stop());
  streamSide = null;

  poseFront.value?.close();
  poseFront.value = null;

  poseSide.value?.close();
  poseSide.value = null;

  dualCameraStatus.value = "idle";
}

// Dual camera inference loop
function dualInferenceLoop() {
  const now = performance.now();
  const loopStart = now;

  // Throttle inference to target FPS
  if (now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS) {
    animFrameId = requestAnimationFrame(dualInferenceLoop);
    return;
  }
  const loopDtMs = frontendDebugLastLoopAt == null ? null : now - frontendDebugLastLoopAt;
  frontendDebugLastLoopAt = now;
  lastInferenceAt = now;

  // Run inference on both cameras
  let resultFront: any = null;
  let resultSide: any = null;
  let frontInferMs: number | null = null;
  let sideInferMs: number | null = null;
  const frontReady = !!(
    poseFront.value &&
    videoRef.value &&
    videoRef.value.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  );
  const sideReady = !!(
    sideEnabled.value &&
    poseSide.value &&
    videoRefSide.value &&
    videoRefSide.value.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  );

  if (frontReady && poseFront.value && videoRef.value) {
    const t = performance.now();
    try {
      resultFront = poseFront.value.detectForVideo(videoRef.value, now);
    } catch (err) {
      console.error("Front pose inference error:", err);
    } finally {
      frontInferMs = performance.now() - t;
    }
  }

  if (sideReady && poseSide.value && videoRefSide.value) {
    const t = performance.now();
    try {
      resultSide = poseSide.value.detectForVideo(videoRefSide.value, now);
    } catch (err) {
      console.error("Side pose inference error:", err);
    } finally {
      sideInferMs = performance.now() - t;
    }
  }

  // Extract landmarks
  const landmarksFront = resultFront?.landmarks?.[0] || null;
  const landmarksSide = resultSide?.landmarks?.[0] || null;

  // Send to backend
  const sendStart = performance.now();
  const sentFrame = sendDualFrameToBackend(landmarksFront, landmarksSide, now);
  const sendMs = performance.now() - sendStart;

  // Draw on canvas (optional, for visualization)
  const drawStart = performance.now();
  drawDualCanvas(landmarksFront, landmarksSide);
  const drawMs = performance.now() - drawStart;

  recordFrontendDebugFrame({
    timestamp_ms: roundMetric(now, 2)!,
    loop_dt_ms: roundMetric(loopDtMs),
    front_infer_ms: roundMetric(frontInferMs),
    side_infer_ms: roundMetric(sideInferMs),
    draw_ms: roundMetric(drawMs)!,
    send_ms: roundMetric(sendMs)!,
    total_loop_ms: roundMetric(performance.now() - loopStart)!,
    total_pose_infer_ms: roundMetric((frontInferMs ?? 0) + (sideInferMs ?? 0))!,
    message_type: sentFrame.messageType,
    side_enabled: sideEnabled.value,
    show_side_video: showSideVideo.value,
    front_ready: frontReady,
    side_ready: sideReady,
    front_has_pose: !!landmarksFront,
    side_has_pose: !!landmarksSide,
    target_interval_ms: roundMetric(MIN_INFERENCE_INTERVAL_MS)!,
    front_video_width: videoRef.value?.videoWidth || null,
    front_video_height: videoRef.value?.videoHeight || null,
    side_video_width: videoRefSide.value?.videoWidth || null,
    side_video_height: videoRefSide.value?.videoHeight || null,
    payload_estimated_bytes: sentFrame.payloadEstimatedBytes,
  });

  animFrameId = requestAnimationFrame(dualInferenceLoop);
}

// Send dual frame to backend
function sendDualFrameToBackend(
  landmarksFront: NormalizedLandmark[] | null,
  landmarksSide: NormalizedLandmark[] | null,
  timestamp: number
): SentFrameInfo {
  const namedFront = landmarksFront ? extractNamedLandmarks(landmarksFront) : null;
  const namedSide = landmarksSide ? extractNamedLandmarks(landmarksSide) : null;

  if (sideEnabled.value && (namedFront || namedSide)) {
    ws.sendDualFrame(namedFront, namedSide, timestamp);
    return {
      messageType: "dual_frame",
      payloadEstimatedBytes: debugRecording.value
        ? estimateJsonBytes({
            type: "dual_frame",
            front: namedFront ? { timestamp_ms: timestamp, landmarks: namedFront } : null,
            side: namedSide ? { timestamp_ms: timestamp, landmarks: namedSide } : null,
          })
        : null,
    };
  } else if (namedFront) {
    ws.sendLandmarks(namedFront, timestamp);
    return {
      messageType: "single_frame",
      payloadEstimatedBytes: debugRecording.value
        ? estimateJsonBytes({ timestamp_ms: timestamp, landmarks: namedFront })
        : null,
    };
  }

  return { messageType: "none", payloadEstimatedBytes: null };
}

// Extract named landmarks for backend
function extractNamedLandmarks(landmarks: NormalizedLandmark[]) {
  const LANDMARK_INDICES: Record<string, number> = {
    LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,    RIGHT_WRIST: 16,
    LEFT_HIP: 23,      RIGHT_HIP: 24,
    LEFT_KNEE: 25,     RIGHT_KNEE: 26,
    LEFT_ANKLE: 27,    RIGHT_ANKLE: 28,
  };

  const named: Record<string, { x: number; y: number; z: number; visibility: number }> = {};
  for (const [name, idx] of Object.entries(LANDMARK_INDICES)) {
    const lm = landmarks[idx];
    if (lm) {
      named[name] = {
        x: lm.x,
        y: lm.y,
        z: lm.z ?? 0,
        visibility: lm.visibility ?? 0,
      };
    }
  }
  return named;
}

// Draw landmarks on canvas (simple visualization)
function drawDualCanvas(
  landmarksFront: NormalizedLandmark[] | null,
  landmarksSide: NormalizedLandmark[] | null
) {
  // Draw front camera
  if (canvasRef.value && videoRef.value && landmarksFront) {
    const ctx = canvasRef.value.getContext("2d");
    if (ctx) {
      const w = canvasRef.value.width = videoRef.value.videoWidth;
      const h = canvasRef.value.height = videoRef.value.videoHeight;
      ctx.clearRect(0, 0, w, h);
      drawPoseLandmarks(ctx, landmarksFront, w, h, "#22c55e");
    }
  }

  // Draw side camera
  if (canvasRefSide.value && videoRefSide.value && landmarksSide && showSideVideo.value) {
    const ctx = canvasRefSide.value.getContext("2d");
    if (ctx) {
      const w = canvasRefSide.value.width = videoRefSide.value.videoWidth;
      const h = canvasRefSide.value.height = videoRefSide.value.videoHeight;
      ctx.clearRect(0, 0, w, h);
      drawPoseLandmarks(ctx, landmarksSide, w, h, "#22c55e");
    }
  }
}

// Simple pose landmark drawing
function drawPoseLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarks: NormalizedLandmark[],
  w: number,
  h: number,
  color: string
) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.fillStyle = color;

  // Draw connections (simple skeleton)
  const connections = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // arms
    [11, 23], [12, 24], [23, 24], // torso
    [23, 25], [25, 27], [24, 26], [26, 28], // legs
  ];

  for (const [a, b] of connections) {
    const lmA = landmarks[a];
    const lmB = landmarks[b];
    if (lmA && lmB && (lmA.visibility ?? 0) > 0.5 && (lmB.visibility ?? 0) > 0.5) {
      ctx.beginPath();
      ctx.moveTo(lmA.x * w, lmA.y * h);
      ctx.lineTo(lmB.x * w, lmB.y * h);
      ctx.stroke();
    }
  }

  // Draw joints
  for (const lm of landmarks) {
    if ((lm.visibility ?? 0) > 0.5) {
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ---- Elapsed timer (wall-clock, for display only) --------------------------

let elapsedTimer: ReturnType<typeof setInterval> | null = null;

// ---- Lifecycle -------------------------------------------------------------

onMounted(async () => {
  for (const t of TABS) switchTab(t.key);
  activeTab.value = TABS[0].key;
  sessionStart.value = Date.now();

  ws.connect();
  elapsedTimer = setInterval(() => {
    elapsedMs.value = Date.now() - sessionStart.value;
  }, 500);

  // Enumerate camera devices for dual camera mode
  await enumerateDevices();

  // Load localStorage settings for dual camera
  const savedSideEnabled = localStorage.getItem("sideEnabled");
  const savedShowSideVideo = localStorage.getItem("showSideVideo");
  const savedFrontDeviceId = localStorage.getItem("frontDeviceId");
  const savedSideDeviceId = localStorage.getItem("sideDeviceId");

  if (savedSideEnabled !== null) sideEnabled.value = savedSideEnabled === "true";
  if (savedShowSideVideo !== null) showSideVideo.value = savedShowSideVideo === "true";
  if (savedFrontDeviceId) frontDeviceId.value = savedFrontDeviceId;
  if (savedSideDeviceId) sideDeviceId.value = savedSideDeviceId;

  // Start dual camera mode (new system, parallel to useMediaPipe)
  if (videoRef.value && canvasRef.value) {
    await startDualCamera();
  }

  // Load MediaPipe models for status display only (no camera inference)
  await mp.loadPoseLandmarker();
});

onUnmounted(() => {
  stopDualCamera();
  mp.dispose();
  ws.disconnect();
  if (elapsedTimer) clearInterval(elapsedTimer);
});

// Watch dual camera settings and save to localStorage
watch(sideEnabled, (val) => {
  localStorage.setItem("sideEnabled", String(val));
  if (val && !poseSide.value) {
    // Restart to load side camera
    stopDualCamera();
    setTimeout(() => startDualCamera(), 100);
  } else if (!val && poseSide.value) {
    // Stop side camera only
    streamSide?.getTracks().forEach(t => t.stop());
    streamSide = null;
    poseSide.value?.close();
    poseSide.value = null;
  }
});

watch(showSideVideo, (val) => {
  localStorage.setItem("showSideVideo", String(val));
});

watch(frontDeviceId, (val) => {
  if (val) localStorage.setItem("frontDeviceId", val);
});

watch(sideDeviceId, (val) => {
  if (val) localStorage.setItem("sideDeviceId", val);
});
</script>

<template>
  <div class="display-shell">

    <!-- LEFT: action tabs -->
    <nav class="display-left" aria-label="动作选择">
      <div class="display-logo">CalorieCal</div>

      <div class="tab-list" role="tablist">
        <button
          v-for="tab in TABS"
          :key="tab.key"
          class="tab-btn"
          :class="{ 'tab-btn--active': activeTab === tab.key }"
          role="tab"
          :aria-selected="activeTab === tab.key"
          type="button"
          @click="switchTab(tab.key)"
        >
          <span class="tab-icon">{{ tab.icon }}</span>
          <span class="tab-label">{{ tab.label }}</span>
          <span
            v-if="activeTab === tab.key"
            class="tab-indicator"
            aria-hidden="true"
          />
        </button>
      </div>

      <!-- Status dots at bottom of left panel -->
      <div class="left-status">
        <div class="left-status-row">
          <span class="cc-dot" :class="mp.cameraStatus.value === 'ready' ? '' : 'cc-dot--yellow'" />
          <span>摄像头</span>
        </div>
        <div class="left-status-row">
          <span class="cc-dot" :class="ws.status.value === 'connected' ? '' : 'cc-dot--yellow'" />
          <span>后端</span>
        </div>
        <div class="left-status-row">
          <span class="cc-dot" :class="mp.modelStatus.value === 'ready' ? '' : 'cc-dot--yellow'" />
          <span>姿态模型</span>
        </div>
      </div>
    </nav>

    <!-- CENTER: camera -->
    <main class="display-center">
      <div class="display-camera-wrap">
        <!-- Front camera video and canvas -->
        <video ref="videoRef" class="display-video" playsinline muted aria-hidden="true" />
        <canvas ref="canvasRef" class="display-canvas" aria-hidden="true" />

        <!-- Side camera video and canvas (hidden by default) -->
        <video
          ref="videoRefSide"
          class="display-video display-video--side"
          :style="{ display: showSideVideo ? 'block' : 'none' }"
          playsinline
          muted
          aria-hidden="true"
        />
        <canvas
          ref="canvasRefSide"
          class="display-canvas display-canvas--side"
          :style="{ display: showSideVideo ? 'block' : 'none' }"
          aria-hidden="true"
        />

        <!-- Current action badge -->
        <div class="cam-overlay cam-overlay--tl">
          <span class="action-badge">{{ activeTabDef.label }}</span>
        </div>

        <!-- Stage hint -->
        <div class="cam-overlay cam-overlay--bl" aria-live="polite">
          <span class="stage-badge cc-mono">
            {{ STAGE_ZH[activeState && 'stage' in activeState ? activeState.stage : 'idle'] ?? '--' }}
          </span>
        </div>

        <!-- No pose hint -->
        <div v-if="mp.poseStatus.value === 'missing'" class="cam-pose-hint">
          请站入画面，确保全身可见
        </div>
      </div>

      <!-- Debug bar -->
      <div class="display-debug">
        <span>{{ mp.inferenceFps.value }} FPS</span>
        <span>·</span>
        <span>姿态 {{ mp.completeness.value }}%</span>
        <span>·</span>
        <span>手 {{ mp.handCount.value }}</span>
        <span>·</span>
        <span>{{ ws.latencyMs.value != null ? `后端 ${ws.latencyMs.value.toFixed(1)}ms` : '等待后端' }}</span>
      </div>
    </main>

    <!-- RIGHT: data panel -->
    <aside class="display-right" aria-label="训练数据">

      <!-- Weight input (top of right panel) -->
      <div class="data-section data-section--weight">
        <p class="data-section__label">体重</p>
        <div class="weight-input-row">
          <input
            v-model.number="weightKg"
            type="number"
            min="30"
            max="200"
            class="cc-input weight-input"
            aria-label="体重（千克）"
          />
          <span class="weight-unit cc-muted">kg</span>
        </div>
      </div>

      <div class="data-divider" />

      <!-- Dual camera settings -->
      <div class="data-section data-section--dual-camera">
        <p class="data-section__label">双摄像头设置</p>

        <!-- Side camera enabled toggle -->
        <div class="camera-toggle-row">
          <label class="camera-toggle-label">
            <input
              v-model="sideEnabled"
              type="checkbox"
              class="camera-toggle-input"
            />
            <span>侧边摄像头</span>
          </label>
          <span class="camera-toggle-hint cc-muted">开启后融合正面和侧面视角</span>
        </div>

        <!-- Show side video toggle (only when side enabled) -->
        <div v-if="sideEnabled" class="camera-toggle-row">
          <label class="camera-toggle-label">
            <input
              v-model="showSideVideo"
              type="checkbox"
              class="camera-toggle-input"
            />
            <span>侧边画面</span>
          </label>
          <span class="camera-toggle-hint cc-muted">显示侧面摄像头画面</span>
        </div>

        <!-- Camera device selectors (when side enabled) -->
        <div v-if="sideEnabled" class="camera-selectors">
          <div class="camera-select-row">
            <label class="camera-select-label">正面</label>
            <select v-model="frontDeviceId" class="cc-select">
              <option :value="null">默认摄像头</option>
              <option
                v-for="device in availableDevices"
                :key="device.deviceId"
                :value="device.deviceId"
              >
                {{ device.label || `摄像头 ${device.deviceId.slice(0, 8)}` }}
              </option>
            </select>
          </div>

          <div class="camera-select-row">
            <label class="camera-select-label">侧面</label>
            <select v-model="sideDeviceId" class="cc-select">
              <option :value="null">默认摄像头</option>
              <option
                v-for="device in availableDevices"
                :key="device.deviceId"
                :value="device.deviceId"
              >
                {{ device.label || `摄像头 ${device.deviceId.slice(0, 8)}` }}
              </option>
            </select>
          </div>

          <!-- Swap button -->
          <button class="cc-btn cc-btn--secondary camera-swap-btn" @click="swapCameras">
            交换正侧
          </button>
        </div>
      </div>

      <div class="data-divider" />

      <!-- Debug recording button -->
      <div class="data-section">
        <p class="data-section__label">调试工具</p>
        <button
          class="cc-btn cc-btn--secondary"
          :disabled="debugRecording"
          @click="startDebugRecording"
          style="width: 100%; margin-top: 8px;"
        >
          {{ debugRecording ? "记录中..." : "记录10秒数据" }}
        </button>
        <p class="data-section__sub cc-muted" style="margin-top: 8px;">
          点击后记录10秒内的所有识别数据，自动下载为JSON文件
        </p>
      </div>

      <div class="data-divider" />

      <!-- Timer -->
      <div class="data-section">
        <p class="data-section__label">时间</p>
        <div class="data-hero cc-mono">{{ fmt(activeElapsedSec) }}</div>
        <p class="data-section__sub cc-muted">本轮计时</p>
      </div>

      <div class="data-divider" />

      <!-- Count or duration -->
      <template v-if="activeTabDef.mode === 'count'">
        <div class="data-section">
          <p class="data-section__label">次数</p>
          <div class="data-hero cc-mono">
            {{ activeState && 'count' in activeState ? activeState.count : 0 }}
          </div>
          <!-- Left/right breakdown for clap_under_knee and high_knee -->
          <template v-if="(activeTabDef.key === 'clap_under_knee' || activeTabDef.key === 'high_knee') && activeState && 'countLeft' in activeState">
            <div class="lunge-times" style="margin-top:8px;">
              <div class="lunge-side">
                <span class="lunge-side__label cc-muted">左腿</span>
                <span class="lunge-side__value cc-mono">{{ activeState.countLeft ?? 0 }}</span>
              </div>
              <div class="lunge-side">
                <span class="lunge-side__label cc-muted">右腿</span>
                <span class="lunge-side__value cc-mono">{{ activeState.countRight ?? 0 }}</span>
              </div>
            </div>
          </template>
          <p class="data-section__sub cc-muted">完成次数</p>
        </div>
      </template>

      <template v-else>
        <!-- Lunge: left + right hold durations -->
        <div class="data-section">
          <p class="data-section__label">有效时长</p>
          <div class="lunge-times">
            <div class="lunge-side">
              <span class="lunge-side__label cc-muted">左腿</span>
              <span class="lunge-side__value cc-mono">
                {{ activeState && 'holdSecLeft' in activeState ? fmt(activeState.holdSecLeft) : '00:00' }}
              </span>
            </div>
            <div class="lunge-side">
              <span class="lunge-side__label cc-muted">右腿</span>
              <span class="lunge-side__value cc-mono">
                {{ activeState && 'holdSecRight' in activeState ? fmt(activeState.holdSecRight) : '00:00' }}
              </span>
            </div>
          </div>
          <p class="data-section__sub cc-muted">
            合计
            <span class="cc-mono">
              {{
                activeState && 'holdSecLeft' in activeState
                  ? fmt(activeState.holdSecLeft + activeState.holdSecRight)
                  : '00:00'
              }}
            </span>
          </p>
        </div>
      </template>

      <div class="data-divider" />

      <!-- Calories: total -->
      <div class="data-section">
        <p class="data-section__label">总热量</p>
        <div class="data-hero-sm cc-mono cc-green">{{ currentCalories }} <span class="cc-muted" style="font-size:0.7em">kcal</span></div>
        <p class="data-section__sub cc-muted">体重 {{ weightKg }}kg</p>
        <!-- event_kcal as auxiliary -->
        <div v-if="backendEventKcal !== null && backendEventKcal > 0" class="calorie-breakdown">
          <div class="calorie-breakdown__row">
            <span class="cc-muted">识别动作</span>
            <span class="cc-mono">{{ backendEventKcal.toFixed(1) }}</span>
          </div>
        </div>
      </div>

      <div class="data-divider" />

      <!-- Instant calorie rate (motion-based, any action) -->
      <div class="data-section">
        <p class="data-section__label">瞬时消耗</p>
        <template v-if="instantKcalPerMin !== null">
          <div class="data-hero data-hero--green cc-mono cc-green">
            {{ (instantKcalPerMin * 1000 / 60).toFixed(2) }}
            <span class="cc-muted" style="font-size:0.5em;margin-left:2px;">cal/s</span>
          </div>
        </template>
        <p v-else class="data-section__sub cc-muted">等待运动数据…</p>
      </div>

      <!-- View fusion info (dual camera mode) -->
      <template v-if="sideEnabled && ws.viewInfo.value">
        <div class="data-divider" />
        <div class="data-section data-section--view-fusion">
          <p class="data-section__label">视角融合</p>

          <!-- Active view indicator -->
          <div class="view-active-row">
            <span class="view-active-label">主导视角</span>
            <span
              class="view-active-badge"
              :class="{
                'view-active-badge--front': ws.viewInfo.value.active_view === 'front',
                'view-active-badge--side': ws.viewInfo.value.active_view === 'side',
                'view-active-badge--none': ws.viewInfo.value.active_view === 'none',
              }"
            >
              {{ ws.viewInfo.value.active_view === 'front' ? '正面' : ws.viewInfo.value.active_view === 'side' ? '侧面' : '无' }}
            </span>
          </div>

          <!-- Visibility percentages -->
          <div class="view-info-grid">
            <div class="view-info-item">
              <span class="view-info-label">正面可见度</span>
              <span class="view-info-value">{{ (ws.viewInfo.value.front_core_vis * 100).toFixed(0) }}%</span>
            </div>
            <div class="view-info-item">
              <span class="view-info-label">侧面可见度</span>
              <span class="view-info-value">{{ (ws.viewInfo.value.side_core_vis * 100).toFixed(0) }}%</span>
            </div>
          </div>

          <!-- Motion signals (debug info, collapsible) -->
          <details class="view-debug-details">
            <summary class="view-debug-summary">调试信息</summary>
            <div class="view-debug-content">
              <div class="view-debug-row">
                <span class="view-debug-label">正面运动</span>
                <span class="view-debug-value">{{ ws.viewInfo.value.front_motion_e.toFixed(4) }}</span>
              </div>
              <div class="view-debug-row">
                <span class="view-debug-label">侧面运动</span>
                <span class="view-debug-value">{{ ws.viewInfo.value.side_motion_e.toFixed(4) }}</span>
              </div>
              <div class="view-debug-row">
                <span class="view-debug-label">融合运动</span>
                <span class="view-debug-value">{{ ws.viewInfo.value.merged_motion_e.toFixed(4) }}</span>
              </div>
            </div>
          </details>
        </div>
      </template>

      <div class="data-divider" />

      <!-- Quality score -->
      <div class="data-section">
        <p class="data-section__label">动作打分</p>
        <div class="score-display">
          <div
            class="score-bar"
            :style="{ width: `${((activeState?.score ?? 0) * 100).toFixed(0)}%` }"
            :class="{
              'score-bar--good':    (activeState?.score ?? 0) >= 0.75,
              'score-bar--fair':    (activeState?.score ?? 0) >= 0.5 && (activeState?.score ?? 0) < 0.75,
              'score-bar--poor':    (activeState?.score ?? 0) > 0 && (activeState?.score ?? 0) < 0.5,
            }"
          />
        </div>
        <p class="data-section__sub">
          <span
            class="quality-label"
            :class="{
              'quality-label--good': (activeState?.score ?? 0) >= 0.75,
              'quality-label--fair': (activeState?.score ?? 0) >= 0.5 && (activeState?.score ?? 0) < 0.75,
            }"
          >
            {{ qualityLabel(activeState?.score ?? null) }}
          </span>
          <span class="cc-muted" style="margin-left:4px;">
            {{ activeState?.score != null ? (activeState.score * 100).toFixed(0) + ' / 100' : '' }}
          </span>
        </p>
      </div>

      <!-- Quality breakdown + advice (count-mode actions) -->
      <template v-if="activeTabDef.mode === 'count' && activeState && 'standardCount' in activeState">
        <div class="data-divider" />
        <div class="data-section">
          <p class="data-section__label">质量分布</p>
          <div class="quality-pills">
            <span class="cc-pill cc-pill--good">标准 {{ activeState.standardCount }} 次</span>
            <span class="cc-pill cc-pill--neutral">较浅 {{ activeState.shallowCount }} 次</span>
          </div>
          <p
            v-if="activeState.lastAdvice"
            class="advice-text"
            aria-live="polite"
          >
            {{ activeState.lastAdvice }}
          </p>
        </div>
      </template>

    </aside>
  </div>
</template>

<style scoped>
/* ---- Shell: 3-column layout ---- */
.display-shell {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr) 300px;
  grid-template-rows: 100vh;
  background: #0f1117;
  color: #e5e7eb;
  font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

/* ---- LEFT ---- */
.display-left {
  background: #13151f;
  border-right: 1px solid #1e2030;
  display: flex;
  flex-direction: column;
  padding: 24px 0 20px;
  overflow: hidden;
}

.display-logo {
  font-size: 13px;
  font-weight: 800;
  color: #22c55e;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0 16px 20px;
}

.tab-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  flex: 1;
}

.tab-btn {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  background: none;
  border: none;
  border-left: 3px solid transparent;
  color: #6b7280;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  text-align: left;
  font-family: inherit;
  position: relative;
  transition: none;
}

.tab-btn:hover { background: rgba(255,255,255,0.04); color: #e5e7eb; }

.tab-btn--active {
  border-left-color: #22c55e;
  background: rgba(34, 197, 94, 0.08);
  color: #ffffff;
  font-weight: 700;
}

.tab-icon  { font-size: 18px; line-height: 1; }
.tab-label { flex: 1; }

.left-status {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  border-top: 1px solid #1e2030;
}

.left-status-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #6b7280;
}

/* ---- CENTER ---- */
.display-center {
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 8px;
  overflow: hidden;
}

.display-camera-wrap {
  flex: 1;
  position: relative;
  background: #18191f;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #2a2d3a;
}

.display-video,
.display-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

/* Side camera: picture-in-picture style */
.display-video--side,
.display-canvas--side {
  position: absolute;
  top: 16px;
  right: 16px;
  width: clamp(320px, 34vw, 460px);
  height: clamp(240px, 25.5vw, 345px);
  max-width: calc(100% - 32px);
  max-height: calc(100% - 96px);
  inset: auto;
  border: 2px solid #22c55e;
  border-radius: 8px;
  z-index: 10;
}

/* ---- Camera overlays ---- */
.cam-overlay { position: absolute; pointer-events: none; }
.cam-overlay--tl { top: 14px; left: 16px; }
.cam-overlay--bl { bottom: 14px; left: 16px; }

.action-badge {
  background: rgba(34, 197, 94, 0.85);
  color: #0f1117;
  font-size: 13px;
  font-weight: 800;
  padding: 4px 12px;
  border-radius: 20px;
  letter-spacing: 0.04em;
}

.stage-badge {
  font-size: 32px;
  font-weight: 700;
  color: rgba(255,255,255,0.85);
  text-shadow: 0 2px 12px rgba(0,0,0,0.7);
  line-height: 1;
}

.cam-pose-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(0,0,0,0.65);
  color: #f3f4f6;
  font-size: 14px;
  padding: 8px 18px;
  border-radius: 20px;
  white-space: nowrap;
  pointer-events: none;
}

.display-debug {
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: 11px;
  color: #4b5563;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  flex-shrink: 0;
}

/* ---- RIGHT ---- */
.display-right {
  background: #13151f;
  border-left: 1px solid #1e2030;
  display: flex;
  flex-direction: column;
  padding: 20px 16px;
  overflow-y: auto;
  gap: 0;
}

.data-section {
  padding: 16px 0;
}

.data-section__label {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6b7280;
  margin: 0 0 8px;
}

.data-section__sub {
  font-size: 12px;
  margin: 6px 0 0;
}

.data-hero {
  font-size: 56px;
  font-weight: 700;
  color: #ffffff;
  line-height: 1;
  letter-spacing: -0.02em;
}

.data-hero--green { color: #22c55e; }

.data-divider {
  height: 1px;
  background: #1e2030;
  flex-shrink: 0;
}

/* ---- Lunge times ---- */
.lunge-times {
  display: flex;
  gap: 12px;
}

.lunge-side {
  flex: 1;
  background: #1a1d27;
  border: 1px solid #2a2d3a;
  border-radius: 10px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lunge-side__label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.lunge-side__value {
  font-size: 28px;
  font-weight: 700;
  color: #ffffff;
  line-height: 1;
}

/* ---- Score bar ---- */
.score-display {
  height: 8px;
  background: #2a2d3a;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 6px;
}

.score-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
  background: #374151;
}

.score-bar--good { background: #22c55e; }
.score-bar--fair { background: #f59e0b; }
.score-bar--poor { background: #ef4444; }

.quality-label { font-size: 13px; font-weight: 700; color: #9ca3af; }
.quality-label--good { color: #22c55e; }
.quality-label--fair { color: #f59e0b; }

/* ---- Quality pills ---- */
.quality-pills { display: flex; gap: 6px; flex-wrap: wrap; }

/* ---- Calorie breakdown ---- */
.calorie-breakdown {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.calorie-breakdown__row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
}

/* ---- Instant calorie rate ---- */
.instant-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  padding: 2px 0;
}
.instant-value {
  font-size: 13px;
  font-weight: 600;
  color: #e5e7eb;
}

/* ---- Dual camera settings ---- */
.data-section--dual-camera {
  overflow: visible;
}

.camera-toggle-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 12px;
}

.camera-toggle-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
}

.camera-toggle-input {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

.camera-toggle-hint {
  font-size: 11px;
  padding-left: 24px;
}

.camera-selectors {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 12px;
}

.camera-select-row {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 6px;
}

.camera-select-label {
  font-size: 12px;
  font-weight: 500;
  min-width: 0;
}

.cc-select {
  width: 100%;
  min-width: 0;
  background: #1a1d27;
  border: 1px solid #2a2d3a;
  border-radius: 6px;
  color: #e5e7eb;
  font-size: 12px;
  padding: 6px 8px;
  cursor: pointer;
}

.cc-select:hover {
  border-color: #22c55e;
}

.camera-swap-btn {
  margin-top: 4px;
  font-size: 12px;
  padding: 6px 12px;
}

.cc-btn {
  background: #22c55e;
  border: none;
  border-radius: 6px;
  color: #0f1117;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

.cc-btn:hover {
  opacity: 0.9;
}

.cc-btn--secondary {
  background: #2a2d3a;
  color: #e5e7eb;
}

.cc-btn--secondary:hover {
  background: #353841;
}

.cc-input {
  background: #1a1d27;
  border: 1px solid #2a2d3a;
  border-radius: 6px;
  color: #e5e7eb;
  font-size: 14px;
  padding: 8px 10px;
}

.cc-input:focus {
  outline: none;
  border-color: #22c55e;
}

.weight-input-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.weight-input {
  flex: 1;
  min-width: 0;
  text-align: center;
}

.weight-unit {
  font-size: 13px;
  font-weight: 500;
}

/* ---- View fusion info ---- */
.data-section--view-fusion {
  font-size: 13px;
}

.view-active-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.view-active-label {
  font-size: 12px;
  color: #9ca3af;
}

.view-active-badge {
  font-size: 13px;
  font-weight: 600;
  padding: 4px 12px;
  border-radius: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.view-active-badge--front {
  background: rgba(34, 197, 94, 0.2);
  color: #22c55e;
}

.view-active-badge--side {
  background: rgba(59, 130, 246, 0.2);
  color: #3b82f6;
}

.view-active-badge--none {
  background: rgba(156, 163, 175, 0.2);
  color: #9ca3af;
}

.view-info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}

.view-info-item {
  background: #1a1d27;
  border: 1px solid #2a2d3a;
  border-radius: 8px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.view-info-label {
  font-size: 11px;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.view-info-value {
  font-size: 18px;
  font-weight: 700;
  color: #e5e7eb;
  font-family: 'Courier New', monospace;
}

.view-debug-details {
  margin-top: 8px;
}

.view-debug-summary {
  font-size: 11px;
  color: #6b7280;
  cursor: pointer;
  user-select: none;
  padding: 4px 0;
}

.view-debug-summary:hover {
  color: #9ca3af;
}

.view-debug-content {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #2a2d3a;
}

.view-debug-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
  font-size: 11px;
}

.view-debug-label {
  color: #9ca3af;
}

.view-debug-value {
  font-family: 'Courier New', monospace;
  color: #e5e7eb;
}

@media (max-width: 900px) {
  .display-video--side,
  .display-canvas--side {
    width: min(46vw, 360px);
    height: min(34.5vw, 270px);
    top: 12px;
    right: 12px;
  }
}

</style>
