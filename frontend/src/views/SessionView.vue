<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useRouter } from "vue-router";
import "../session-ui.css";
import { useMediaPipe } from "../composables/useMediaPipe";
import { useActionWs } from "../composables/useActionWs";
import { useSession } from "../composables/useSession";

const router  = useRouter();
const mp      = useMediaPipe();
const ws      = useActionWs();
const session = useSession();

const videoRef  = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);

// ---- Send pose to backend after each inference --------------------------

const BACKEND_KEY_MAP: Record<string, string> = {
  leftShoulder: "LEFT_SHOULDER", rightShoulder: "RIGHT_SHOULDER",
  leftElbow:    "LEFT_ELBOW",    rightElbow:    "RIGHT_ELBOW",
  leftWrist:    "LEFT_WRIST",    rightWrist:    "RIGHT_WRIST",
  leftHip:      "LEFT_HIP",      rightHip:      "RIGHT_HIP",
  leftKnee:     "LEFT_KNEE",     rightKnee:     "RIGHT_KNEE",
  leftAnkle:    "LEFT_ANKLE",    rightAnkle:    "RIGHT_ANKLE",
};

const BACKEND_IDX = mp.LANDMARK_INDEX;

mp.setOnInference((pose) => {
  if (!pose || session.state.value !== "running") return;
  const named: Record<string, { x: number; y: number; z: number; visibility: number }> = {};
  for (const [camel, backendName] of Object.entries(BACKEND_KEY_MAP)) {
    const idx = BACKEND_IDX[camel as keyof typeof BACKEND_IDX];
    const lm  = pose[idx];
    if (lm) named[backendName] = { x: lm.x, y: lm.y, z: lm.z ?? 0, visibility: lm.visibility ?? 0 };
  }
  ws.sendLandmarks(named, performance.now());
});

// ---- Watch backend results → update session ----------------------------

watch(ws.actions, (newActions) => {
  session.updateFromBackend(newActions);
});

// ---- Display helpers ----------------------------------------------------

const calories = computed(() => session.estimatedCalories().toFixed(1));
const elapsed  = computed(() => session.formatElapsed(session.elapsedMs.value));

const ACTION_LABELS: Record<string, string> = {
  squat:        "深蹲",
  jumping_jack: "开合跳",
};

const STAGE_LABELS: Record<string, string> = {
  stand: "站立", down: "下蹲", downing: "下蹲中", rising: "起身",
  closed: "合拢", open: "打开", opening: "打开中", closing: "回收",
  unknown: "--",
};

const stageHint = computed(() => {
  const all = ws.actions.value;
  if (!all.length) return "--";
  const best = all.reduce((a, b) => ((b.score ?? 0) > (a.score ?? 0) ? b : a));
  return STAGE_LABELS[best.stage] ?? best.stage;
});

// ---- Controls -----------------------------------------------------------

function handlePause() {
  if (session.state.value === "running") session.pause();
  else if (session.state.value === "paused") session.resume();
}

function handleEnd() {
  session.end();
  mp.dispose();
  ws.disconnect();
  router.push("/report");
}

// ---- Lifecycle ----------------------------------------------------------

onMounted(async () => {
  if (session.state.value === "idle") session.start();
  ws.connect();
  await Promise.all([mp.loadPoseLandmarker(), mp.loadHandLandmarker()]);
  if (videoRef.value && canvasRef.value) {
    await mp.startCamera(videoRef.value, canvasRef.value);
  }
});

onUnmounted(() => {
  mp.dispose();
  ws.disconnect();
});
</script>

<template>
  <div class="cc-page session-view">

    <!-- Camera area -->
    <div class="session-camera-wrap">
      <div class="session-camera" role="img" aria-label="摄像头画面">
        <video ref="videoRef" class="session-video" playsinline muted aria-hidden="true" />
        <canvas ref="canvasRef" class="session-canvas" aria-hidden="true" />

        <!-- Stage hint -->
        <div class="cam-overlay cam-overlay--tl" aria-live="polite">
          <span class="stage-hint cc-mono">{{ stageHint }}</span>
        </div>

        <!-- Timer -->
        <div class="cam-overlay cam-overlay--tr">
          <span class="session-timer cc-mono">{{ elapsed }}</span>
        </div>

        <!-- Status -->
        <div class="cam-overlay cam-overlay--br">
          <div class="cam-status-item">
            <span class="cc-dot" :class="ws.status.value === 'connected' ? '' : 'cc-dot--yellow'" aria-hidden="true" />
            <span class="cam-status-text">
              {{ ws.status.value === "connected" ? "后端已连接" : ws.status.value }}
            </span>
          </div>
        </div>

        <!-- Pose missing hint -->
        <div v-if="mp.poseStatus.value === 'missing'" class="cam-pose-hint">
          请站入画面，确保全身可见
        </div>
      </div>
    </div>

    <!-- Data cards -->
    <div class="session-cards" role="region" aria-label="训练数据">

      <div
        v-for="action in ws.actions.value"
        :key="action.name"
        class="cc-card data-card"
      >
        <span class="data-card__label cc-muted">{{ ACTION_LABELS[action.name] ?? action.name }}</span>
        <div class="data-card__count cc-mono">
          {{ session.actionMap.value[action.name]?.count ?? 0 }}
        </div>
        <div class="data-card__meta">
          <span class="cc-pill">{{ STAGE_LABELS[action.stage] ?? action.stage }}</span>
        </div>
        <span class="data-card__quality cc-muted">
          <template v-if="action.name === 'squat' && session.actionMap.value['squat']">
            标准 {{ session.actionMap.value['squat'].standardCount }} ·
            浅蹲 {{ session.actionMap.value['squat'].shallowCount }}
          </template>
          <template v-else>&nbsp;</template>
        </span>
      </div>

      <!-- Calorie card -->
      <div class="cc-card data-card">
        <span class="data-card__label cc-muted">消耗热量</span>
        <div class="data-card__count cc-mono cc-green">{{ calories }}</div>
        <div class="data-card__meta">
          <span class="data-card__unit cc-muted">kcal</span>
        </div>
        <span class="data-card__quality cc-muted">&nbsp;</span>
      </div>

    </div>

    <!-- Controls -->
    <div class="session-controls" role="group" aria-label="训练控制">
      <button type="button" class="cc-btn-secondary ctrl-btn" @click="handlePause">
        {{ session.state.value === "paused" ? "继续" : "暂停" }}
      </button>
      <button type="button" class="cc-btn-danger ctrl-btn" @click="handleEnd">
        结束训练
      </button>
    </div>

    <!-- Debug row -->
    <div class="session-debug" aria-label="调试信息">
      <span class="debug-item">{{ ws.latencyMs.value != null ? `后端 ${ws.latencyMs.value.toFixed(1)}ms` : "等待后端" }}</span>
      <span class="debug-sep">·</span>
      <span class="debug-item">姿态 {{ mp.completeness.value }}%</span>
      <span class="debug-sep">·</span>
      <span class="debug-item">{{ mp.inferenceFps.value }} FPS</span>
      <span class="debug-sep">·</span>
      <span class="debug-item">手部 {{ mp.handCount.value }}</span>
    </div>

  </div>
</template>

<style scoped>
.session-view {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  padding: 0;
}

.session-camera-wrap {
  width: 100%;
  background: #0f1117;
  padding: 12px 12px 0;
}

.session-camera {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #18191f;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #2a2d3a;
}

.session-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

.session-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  transform: scaleX(-1);
}

.cam-overlay { position: absolute; pointer-events: none; }
.cam-overlay--tl { top: 16px; left: 20px; }
.cam-overlay--tr { top: 16px; right: 20px; }
.cam-overlay--br { bottom: 14px; right: 16px; }

.stage-hint {
  font-size: 48px;
  font-weight: 700;
  color: rgba(255,255,255,0.85);
  text-shadow: 0 2px 12px rgba(0,0,0,0.7);
  line-height: 1;
}

.session-timer {
  font-size: 28px;
  font-weight: 600;
  color: #fff;
  text-shadow: 0 2px 8px rgba(0,0,0,0.7);
  letter-spacing: 0.06em;
}

.cam-status-item {
  display: flex;
  align-items: center;
  gap: 5px;
  background: rgba(0,0,0,0.5);
  border-radius: 6px;
  padding: 4px 8px;
}

.cam-status-text { font-size: 11px; color: #d1fae5; font-weight: 600; }

.cam-pose-hint {
  position: absolute;
  bottom: 50px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0,0,0,0.6);
  color: #f3f4f6;
  font-size: 13px;
  padding: 6px 14px;
  border-radius: 20px;
  white-space: nowrap;
  pointer-events: none;
}

.session-cards {
  display: flex;
  gap: 10px;
  padding: 12px;
  flex: 1;
}

.data-card {
  flex: 1;
  padding: 14px 12px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.data-card__label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.data-card__count {
  font-size: 80px;
  font-weight: 700;
  color: #fff;
  line-height: 1;
  letter-spacing: -0.02em;
}

.data-card__meta { display: flex; align-items: center; gap: 6px; min-height: 24px; }
.data-card__unit { font-size: 18px; font-weight: 500; }
.data-card__quality { font-size: 12px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.session-controls { display: flex; gap: 10px; padding: 0 12px 10px; }
.ctrl-btn { flex: 1; }

.session-debug {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px 12px;
  justify-content: flex-end;
}

.debug-item { font-size: 11px; color: #4b5563; font-family: 'JetBrains Mono', 'Courier New', monospace; }
.debug-sep  { color: #374151; font-size: 11px; }

@media (min-width: 768px) {
  .session-camera-wrap { padding: 16px 16px 0; }
  .session-cards { padding: 16px; gap: 14px; }
  .session-controls { padding: 0 16px 12px; }
  .data-card__count { font-size: 96px; }
}
</style>
