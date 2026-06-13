<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import "../session-ui.css";
import { useMediaPipe } from "../composables/useMediaPipe";
import { useActionWs }  from "../composables/useActionWs";

// ---- Types -----------------------------------------------------------------

type ActionTab = {
  key:   string;
  label: string;
  mode:  "count" | "duration";
  icon:  string;
};

type CountState = {
  count:         number;
  countLeft?:    number;
  countRight?:   number;
  stage:         string;
  score:         number | null;
  qualityLabel?: string;
};

type DurationState = {
  stage:          string;
  score:          number | null;
  holdSecLeft:    number;
  holdSecRight:   number;
};

type ActionState = CountState | DurationState;

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

const BACKEND_KEY_MAP: Record<string, string> = {
  leftShoulder: "LEFT_SHOULDER", rightShoulder: "RIGHT_SHOULDER",
  leftElbow:    "LEFT_ELBOW",    rightElbow:    "RIGHT_ELBOW",
  leftWrist:    "LEFT_WRIST",    rightWrist:    "RIGHT_WRIST",
  leftHip:      "LEFT_HIP",      rightHip:      "RIGHT_HIP",
  leftKnee:     "LEFT_KNEE",     rightKnee:     "RIGHT_KNEE",
  leftAnkle:    "LEFT_ANKLE",    rightAnkle:    "RIGHT_ANKLE",
};

// ---- State -----------------------------------------------------------------

const activeTab    = ref<string>(TABS[0].key);
const videoRef     = ref<HTMLVideoElement | null>(null);
const canvasRef    = ref<HTMLCanvasElement | null>(null);
const weightKg     = ref(70);

// Per-action runtime state (reset on tab switch)
const actionStates = ref<Record<string, ActionState>>({});
const sessionStart = ref<number>(Date.now());
const elapsedMs    = ref(0);

// Calories from backend
const backendMotionKcal = ref<number | null>(null);
const backendEventKcal  = ref<number | null>(null);
const backendByAction   = ref<Record<string, number>>({});

// Instant rate: sliding window over motion_kcal (0.5s)
const INSTANT_WINDOW_MS = 500;
type KcalSample = { ts: number; kcal: number };
const kcalWindow: KcalSample[] = [];
const instantKcalPerMin = ref<number | null>(null);

function pushKcalSample(motionKcal: number) {
  const now = performance.now();
  kcalWindow.push({ ts: now, kcal: motionKcal });
  // trim entries older than window
  const cutoff = now - INSTANT_WINDOW_MS;
  while (kcalWindow.length > 0 && kcalWindow[0].ts < cutoff) kcalWindow.shift();
}

function computeInstantRate() {
  if (kcalWindow.length < 2) { instantKcalPerMin.value = null; return; }
  const oldest = kcalWindow[0];
  const newest = kcalWindow[kcalWindow.length - 1];
  const dtMin = (newest.ts - oldest.ts) / 60000;
  if (dtMin < 0.0005) { instantKcalPerMin.value = null; return; }
  instantKcalPerMin.value = Math.max(0, (newest.kcal - oldest.kcal) / dtMin);
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

mp.setOnInference((pose) => {
  if (!pose) return;
  const named: Record<string, { x: number; y: number; z: number; visibility: number }> = {};
  const idx = mp.LANDMARK_INDEX;
  for (const [camel, backendName] of Object.entries(BACKEND_KEY_MAP)) {
    const lm = pose[idx[camel as keyof typeof idx]];
    if (lm) named[backendName] = { x: lm.x, y: lm.y, z: lm.z ?? 0, visibility: lm.visibility ?? 0 };
  }
  ws.sendLandmarks(named, performance.now());
});

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
      actionStates.value["clap_under_knee"] = {
        count:      a.count,
        countLeft:  (a as any).count_left  ?? 0,
        countRight: (a as any).count_right ?? 0,
        stage:      a.stage,
        score:      a.score,
      };
    } else if (a.name === "high_knee") {
      actionStates.value["high_knee"] = {
        count:      a.count,
        countLeft:  (a as any).count_left  ?? 0,
        countRight: (a as any).count_right ?? 0,
        stage:      a.stage,
        score:      a.score,
      };
    } else {
      actionStates.value[a.name] = {
        count: a.count,
        stage: a.stage,
        score: a.score,
      };
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
  if (motionKcal !== null) pushKcalSample(motionKcal);
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
    : { count: 0, stage: "unknown", score: null };
  ws.disconnect();
  setTimeout(() => ws.connect(), 200);
}

// ---- Elapsed timer (wall-clock, for display only) --------------------------

let elapsedTimer: ReturnType<typeof setInterval> | null = null;
let rateTimer:    ReturnType<typeof setInterval> | null = null;

// ---- Lifecycle -------------------------------------------------------------

onMounted(async () => {
  for (const t of TABS) switchTab(t.key);
  activeTab.value = TABS[0].key;
  sessionStart.value = Date.now();

  ws.connect();
  elapsedTimer = setInterval(() => {
    elapsedMs.value = Date.now() - sessionStart.value;
  }, 500);
  rateTimer = setInterval(computeInstantRate, 200);
  await Promise.all([mp.loadPoseLandmarker(), mp.loadHandLandmarker()]);
  if (videoRef.value && canvasRef.value) {
    await mp.startCamera(videoRef.value, canvasRef.value);
  }
});

onUnmounted(() => {
  mp.dispose();
  ws.disconnect();
  if (elapsedTimer) clearInterval(elapsedTimer);
  if (rateTimer)    clearInterval(rateTimer);
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
        <video ref="videoRef" class="display-video" playsinline muted aria-hidden="true" />
        <canvas ref="canvasRef" class="display-canvas" aria-hidden="true" />

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

      <!-- Stage detail (for squat: quality breakdown placeholder) -->
      <template v-if="activeTabDef.key === 'squat'">
        <div class="data-divider" />
        <div class="data-section">
          <p class="data-section__label">质量分布</p>
          <div class="quality-pills">
            <span class="cc-pill">标准 -- 次</span>
            <span class="cc-pill cc-pill--neutral">浅蹲 -- 次</span>
          </div>
          <p class="data-section__sub cc-muted">计算逻辑待定</p>
        </div>
      </template>

    </aside>
  </div>
</template>

<style scoped>
/* ---- Shell: 3-column layout ---- */
.display-shell {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr) 260px;
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
</style>
