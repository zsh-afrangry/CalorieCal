import { ref, onUnmounted, type Ref } from "vue";

const ACTION_SERVER_URL = `ws://${window.location.hostname}:8766/ws`;

// Maps camelCase frontend key → uppercase backend landmark name
const LANDMARK_KEY_MAP: Record<string, string> = {
  leftShoulder: "LEFT_SHOULDER", rightShoulder: "RIGHT_SHOULDER",
  leftElbow:    "LEFT_ELBOW",    rightElbow:    "RIGHT_ELBOW",
  leftWrist:    "LEFT_WRIST",    rightWrist:    "RIGHT_WRIST",
  leftHip:      "LEFT_HIP",      rightHip:      "RIGHT_HIP",
  leftKnee:     "LEFT_KNEE",     rightKnee:     "RIGHT_KNEE",
  leftAnkle:    "LEFT_ANKLE",    rightAnkle:    "RIGHT_ANKLE",
};

const LANDMARK_INDEX: Record<string, number> = {
  leftShoulder: 11, rightShoulder: 12,
  leftElbow:    13, rightElbow:    14,
  leftWrist:    15, rightWrist:    16,
  leftHip:      23, rightHip:      24,
  leftKnee:     25, rightKnee:     26,
  leftAnkle:    27, rightAnkle:    28,
};

export type ActionResult = {
  name: string;
  count: number;
  stage: string;
  score: number | null;
};

export type CalorieSummary = {
  motion_kcal: number;
  event_kcal: number;
  total_kcal: number;
  by_action: Record<string, number>;
  instant_kcal_per_min: number;
};

export type CountEvent = {
  action: string;
  side: string | null;
  rep_duration_sec: number;
  amplitude: number | null;
  quality: string;
  advice: string;
  timestamp_ms: number;
  kcal: number;
};

export type ViewInfo = {
  front_core_vis: number;
  side_core_vis: number;
  active_view: "front" | "side" | "none";
  front_motion_e: number;
  side_motion_e: number;
  merged_motion_e: number;
};

export type BackendFrame = {
  frame_index?: number;
  actions: ActionResult[];
  latency_ms?: number;
  features?: Record<string, number | null>;
  count_events?: CountEvent[];
  calorie_summary?: CalorieSummary;
  view_info?: ViewInfo;
};

export type LandmarkPoint = {
  x: number; y: number; z?: number; visibility?: number;
};

export function useActionWs() {
  const status = ref<"disconnected" | "connecting" | "connected" | "error">("disconnected");
  const actions = ref<ActionResult[]>([]);
  const latencyMs = ref<number | null>(null);
  const calorieSummary = ref<CalorieSummary | null>(null);
  const countEvents = ref<CountEvent[]>([]);
  const viewInfo = ref<ViewInfo | null>(null);

  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let frameIndex = 0;
  let _pendingConfig: { weight_kg: number; height_cm?: number } | null = null;

  function connect() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;
    status.value = "connecting";
    ws = new WebSocket(ACTION_SERVER_URL);

    ws.onopen = () => {
      status.value = "connected";
      frameIndex = 0;
      if (_pendingConfig) {
        ws!.send(JSON.stringify({ type: "config", ..._pendingConfig }));
      }
    };

    ws.onmessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data as string) as BackendFrame & { error?: string };
        if (data.error) return;
        if (data.actions) actions.value = data.actions;
        if (data.latency_ms != null) latencyMs.value = data.latency_ms;
        if (data.calorie_summary) calorieSummary.value = data.calorie_summary;
        if (data.count_events?.length) countEvents.value = data.count_events;
        if (data.view_info) viewInfo.value = data.view_info;
      } catch { /* ignore */ }
    };

    ws.onerror = () => { status.value = "error"; };

    ws.onclose = () => {
      ws = null;
      if (status.value !== "disconnected") {
        status.value = "disconnected";
        reconnectTimer = setTimeout(connect, 2000);
      }
    };
  }

  function disconnect() {
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
    status.value = "disconnected";
    ws?.close();
    ws = null;
    actions.value = [];
    latencyMs.value = null;
    calorieSummary.value = null;
    countEvents.value = [];
    viewInfo.value = null;
  }

  function sendConfig(weightKg: number, heightCm?: number) {
    _pendingConfig = { weight_kg: weightKg, height_cm: heightCm };
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "config", weight_kg: weightKg, height_cm: heightCm }));
    }
  }

  function sendLandmarks(
    named: Record<string, { x: number; y: number; z: number; visibility: number }>,
    timestampMs: number,
  ) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ frame_index: frameIndex++, timestamp_ms: timestampMs, landmarks: named }));
  }

  function sendDualFrame(
    frontNamed: Record<string, { x: number; y: number; z: number; visibility: number }> | null,
    sideNamed: Record<string, { x: number; y: number; z: number; visibility: number }> | null,
    timestampMs: number,
  ) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
      type: "dual_frame",
      front: frontNamed ? { timestamp_ms: timestampMs, landmarks: frontNamed } : null,
      side: sideNamed ? { timestamp_ms: timestampMs, landmarks: sideNamed } : null,
    }));
  }

  function startDebugRecording(): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }

      const handler = (ev: MessageEvent) => {
        const data = JSON.parse(ev.data);
        if (data.type === "debug_result") {
          ws!.removeEventListener("message", handler);
          resolve(data);
        }
      };

      ws.addEventListener("message", handler);
      ws.send(JSON.stringify({ type: "debug_start" }));

      // Auto-stop after 10 seconds
      setTimeout(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "debug_stop" }));
        }
      }, 10000);
    });
  }

  onUnmounted(disconnect);

  return {
    status, actions, latencyMs, calorieSummary, countEvents, viewInfo,
    connect, disconnect, sendConfig, sendLandmarks, sendDualFrame, startDebugRecording
  };
}
