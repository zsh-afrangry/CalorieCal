/**
 * useMediaPipe.ts
 *
 * 完整提取自 DemoView.vue 的摄像头 + Pose + Hand 推理 + 绘制逻辑。
 * SessionView 直接使用，与 DemoView 保持完全一致的识别效果（包含手部和面部点）。
 */

import { ref, shallowRef, onUnmounted } from "vue";
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

// ---- 常量（与 DemoView.vue 完全一致）------------------------------------

const MEDIAPIPE_TASKS_VERSION = "0.10.22-rc.20250304";
const MEDIAPIPE_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_TASKS_VERSION}/wasm`;
const HAND_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const TARGET_INFERENCE_FPS = 20;
const MIN_INFERENCE_INTERVAL_MS = 1000 / TARGET_INFERENCE_FPS;
const SMOOTHING_WINDOW_SIZE = 5;
const MIN_LANDMARK_VISIBILITY = 0.5;
const REQUIRED_LANDMARK_INDICES = [11, 12, 15, 16, 23, 24, 27, 28];
const DISPLAY_FACE_LANDMARK_INDICES = [0, 3, 6, 9, 10];
const EMPHASIZED_JOINT_INDICES = new Set([11, 12, 23, 24, 25, 26, 27, 28]);

const LEFT_BODY_SEGMENTS = [[11, 13], [13, 15], [23, 25], [25, 27]] as const;
const RIGHT_BODY_SEGMENTS = [[12, 14], [14, 16], [24, 26], [26, 28]] as const;
const CENTER_BODY_SEGMENTS = [[11, 12], [11, 23], [12, 24], [23, 24]] as const;
const FOOT_SEGMENTS = [[27, 29], [29, 31], [28, 30], [30, 32]] as const;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
] as const;

export const LANDMARK_INDEX = {
  nose: 0, leftEar: 7, rightEar: 8,
  leftShoulder: 11, rightShoulder: 12,
  leftElbow: 13,    rightElbow: 14,
  leftWrist: 15,    rightWrist: 16,
  leftHip: 23,      rightHip: 24,
  leftKnee: 25,     rightKnee: 26,
  leftAnkle: 27,    rightAnkle: 28,
} as const;

export type LandmarkPoint = { x: number; y: number; z?: number; visibility?: number; };

// ---- 绘制工具函数（与 DemoView 完全一致）---------------------------------

function toCanvasPoint(lm: LandmarkPoint, w: number, h: number) {
  return { x: lm.x * w, y: lm.y * h };
}

function isVisibleLandmark(lm: LandmarkPoint | undefined): lm is LandmarkPoint {
  if (!lm) return false;
  return (lm.x > 0 || lm.y > 0) && (lm.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY;
}

function shouldDrawPosePoint(index: number) {
  return index >= 11 || DISPLAY_FACE_LANDMARK_INDICES.includes(index);
}

function drawPoseJoint(
  ctx: CanvasRenderingContext2D,
  point: { x: number; y: number },
  radius: number,
  fill: string,
) {
  ctx.beginPath();
  ctx.fillStyle = fill;
  ctx.strokeStyle = "rgba(15, 23, 42, 0.76)";
  ctx.lineWidth = 2;
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawBodySegmentGroup(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[],
  segments: readonly (readonly [number, number])[],
  color: string,
  width: number,
) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  segments.forEach(([a, b]) => {
    const fa = landmarks[a], fb = landmarks[b];
    if (!isVisibleLandmark(fa) || !isVisibleLandmark(fb)) return;
    const pa = toCanvasPoint(fa, canvas.width, canvas.height);
    const pb = toCanvasPoint(fb, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawPoseOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  landmarks: LandmarkPoint[],
) {
  drawBodySegmentGroup(ctx, canvas, landmarks, CENTER_BODY_SEGMENTS, "rgba(45,212,191,0.92)", Math.max(5, canvas.width / 220));
  drawBodySegmentGroup(ctx, canvas, landmarks, LEFT_BODY_SEGMENTS,   "rgba(96,165,250,0.96)", Math.max(8, canvas.width / 150));
  drawBodySegmentGroup(ctx, canvas, landmarks, RIGHT_BODY_SEGMENTS,  "rgba(250,204,21,0.96)", Math.max(8, canvas.width / 150));
  drawBodySegmentGroup(ctx, canvas, landmarks, FOOT_SEGMENTS,        "rgba(226,232,240,0.86)", Math.max(4, canvas.width / 260));

  landmarks.forEach((lm, i) => {
    if (!isVisibleLandmark(lm) || !shouldDrawPosePoint(i)) return;
    const pt = toCanvasPoint(lm, canvas.width, canvas.height);
    const isRequired   = REQUIRED_LANDMARK_INDICES.includes(i);
    const isFace       = DISPLAY_FACE_LANDMARK_INDICES.includes(i);
    const isEmphasized = EMPHASIZED_JOINT_INDICES.has(i);
    drawPoseJoint(
      ctx, pt,
      isEmphasized ? Math.max(8, canvas.width / 170) : isRequired ? 7 : 5,
      isEmphasized ? "#22d3ee" : isRequired ? "#facc15" : isFace ? "#fb7185" : "#ffffff",
    );
  });
}

function drawHandsOverlay(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  hands: LandmarkPoint[][],
) {
  if (!hands.length) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(250,204,21,0.95)";
  ctx.lineWidth = Math.max(3, canvas.width / 280);
  hands.forEach((hand) => {
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const fa = hand[a], fb = hand[b];
      if (!fa || !fb) return;
      ctx.beginPath();
      ctx.moveTo(fa.x * canvas.width, fa.y * canvas.height);
      ctx.lineTo(fb.x * canvas.width, fb.y * canvas.height);
      ctx.stroke();
    });
    hand.forEach((pt) => {
      ctx.beginPath();
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "rgba(15,23,42,0.72)";
      ctx.lineWidth = 1.5;
      ctx.arc(pt.x * canvas.width, pt.y * canvas.height, Math.max(2.5, canvas.width / 520), 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  });
  ctx.restore();
}

// ---- 平滑（与 DemoView 完全一致）-----------------------------------------

function toLandmarkPoints(landmarks: LandmarkPoint[]): LandmarkPoint[] {
  return landmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));
}

function computeSmoothed(raw: LandmarkPoint[], history: LandmarkPoint[][]): LandmarkPoint[] {
  history.push(toLandmarkPoints(raw));
  while (history.length > SMOOTHING_WINDOW_SIZE) history.shift();
  return raw.map((_, i) => {
    const pts = history.map(f => f[i]).filter(Boolean);
    const n = pts.length || 1;
    return {
      x:          pts.reduce((s, p) => s + p.x, 0) / n,
      y:          pts.reduce((s, p) => s + p.y, 0) / n,
      z:          pts.reduce((s, p) => s + (p.z ?? 0), 0) / n,
      visibility: pts.reduce((s, p) => s + (p.visibility ?? 1), 0) / n,
    };
  });
}

// ---- Composable ----------------------------------------------------------

export function useMediaPipe() {
  const cameraStatus    = ref<"idle" | "loading" | "ready" | "error">("idle");
  const modelStatus     = ref<"idle" | "loading" | "ready" | "error">("idle");
  const handModelStatus = ref<"idle" | "loading" | "ready" | "error">("idle");
  const poseStatus      = ref<"idle" | "detected" | "missing">("idle");
  const inferenceFps    = ref(0);
  const handCount       = ref(0);
  const completeness    = ref(0); // % of required landmarks visible

  // Exposed to consumers for sending to backend
  const poseLandmarks = shallowRef<LandmarkPoint[] | null>(null);
  const handLandmarks = shallowRef<LandmarkPoint[][]>([]);

  let stream: MediaStream | null = null;
  let animFrameId: number | null = null;
  let poseLandmarker: PoseLandmarker | null = null;
  let handLandmarker: HandLandmarker | null = null;
  let isInferencing = false;
  let lastInferenceAt = 0;
  let lastFpsSampleAt = 0;
  let fpsFrameCount = 0;
  const landmarkHistory: LandmarkPoint[][] = [];

  // Callback invoked after each successful inference (use for sending to backend)
  let onInferenceCallback: ((pose: LandmarkPoint[] | null, hands: LandmarkPoint[][]) => void) | null = null;

  function setOnInference(cb: typeof onInferenceCallback) {
    onInferenceCallback = cb;
  }

  // ---- Model loading ----

  async function loadPoseLandmarker() {
    if (poseLandmarker || modelStatus.value === "loading") return;
    modelStatus.value = "loading";
    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      modelStatus.value = "ready";
    } catch {
      poseLandmarker = null;
      modelStatus.value = "error";
    }
  }

  async function loadHandLandmarker() {
    if (handLandmarker || handModelStatus.value === "loading") return;
    handModelStatus.value = "loading";
    try {
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_URL },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45,
        minTrackingConfidence: 0.45,
      });
      handModelStatus.value = "ready";
    } catch {
      handLandmarker = null;
      handModelStatus.value = "error";
    }
  }

  // ---- Inference ----

  function updateFps(now: number) {
    fpsFrameCount++;
    if (now - lastFpsSampleAt >= 1000) {
      inferenceFps.value = Math.round(fpsFrameCount * 1000 / (now - lastFpsSampleAt));
      fpsFrameCount = 0;
      lastFpsSampleAt = now;
    }
  }

  function runInference(video: HTMLVideoElement, now: number) {
    if (
      (!poseLandmarker && !handLandmarker) ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      isInferencing ||
      now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS
    ) return;

    isInferencing = true;
    lastInferenceAt = now;

    try {
      let hands: LandmarkPoint[][] = [];
      if (handLandmarker) {
        const hr = handLandmarker.detectForVideo(video, now);
        hands = (hr.landmarks as LandmarkPoint[][]).map(toLandmarkPoints);
        handCount.value = hands.length;
      } else {
        handCount.value = 0;
      }
      handLandmarks.value = hands;

      updateFps(now);

      if (!poseLandmarker) return;

      const result = poseLandmarker.detectForVideo(video, now);
      const raw = result.landmarks[0] as LandmarkPoint[] | undefined;

      if (!raw?.length) {
        poseStatus.value = "missing";
        completeness.value = 0;
        landmarkHistory.length = 0;
        poseLandmarks.value = null;
        onInferenceCallback?.(null, hands);
        return;
      }

      poseStatus.value = "detected";
      const smoothed = computeSmoothed(raw, landmarkHistory);
      poseLandmarks.value = smoothed;

      const visCount = REQUIRED_LANDMARK_INDICES.filter(
        i => (smoothed[i]?.visibility ?? 1) >= MIN_LANDMARK_VISIBILITY
      ).length;
      completeness.value = Math.round((visCount / REQUIRED_LANDMARK_INDICES.length) * 100);

      onInferenceCallback?.(smoothed, hands);
    } finally {
      isInferencing = false;
    }
  }

  // ---- Render loop ----

  function renderFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (video.videoWidth && canvas.width !== video.videoWidth) {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const now = performance.now();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    runInference(video, now);
    if (poseLandmarks.value) drawPoseOverlay(ctx, canvas, poseLandmarks.value);
    drawHandsOverlay(ctx, canvas, handLandmarks.value);

    animFrameId = requestAnimationFrame(() => renderFrame(video, canvas));
  }

  // ---- Camera ----

  async function startCamera(
    videoEl: HTMLVideoElement,
    canvasEl: HTMLCanvasElement,
  ) {
    if (!navigator.mediaDevices?.getUserMedia) {
      cameraStatus.value = "error";
      return;
    }
    cameraStatus.value = "loading";
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      cameraStatus.value = "ready";
      if (animFrameId !== null) cancelAnimationFrame(animFrameId);
      animFrameId = requestAnimationFrame(() => renderFrame(videoEl, canvasEl));
    } catch {
      cameraStatus.value = "error";
    }
  }

  function stopCamera() {
    if (animFrameId !== null) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
  }

  function dispose() {
    stopCamera();
    poseLandmarker?.close(); poseLandmarker = null;
    handLandmarker?.close(); handLandmarker = null;
    poseLandmarks.value = null;
    handLandmarks.value = [];
    landmarkHistory.length = 0;
    cameraStatus.value = "idle";
    modelStatus.value = "idle";
    handModelStatus.value = "idle";
    poseStatus.value = "idle";
  }

  onUnmounted(dispose);

  return {
    cameraStatus,
    modelStatus,
    handModelStatus,
    poseStatus,
    inferenceFps,
    handCount,
    completeness,
    poseLandmarks,
    handLandmarks,
    setOnInference,
    loadPoseLandmarker,
    loadHandLandmarker,
    startCamera,
    stopCamera,
    dispose,
    // expose for external use
    LANDMARK_INDEX,
  };
}
