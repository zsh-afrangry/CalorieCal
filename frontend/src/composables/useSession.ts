import { ref, readonly } from "vue";

export type TrainingMode = "free" | "target" | "timed";

export type ActionEvent = {
  action: string;
  timestamp: number;
  quality?: string;
};

export type SessionProfile = {
  weightKg: number;
  heightCm: number;
  mode: TrainingMode;
  targetReps?: number;
  targetSec?: number;
  selectedActions: string[];
};

export type ActionSummary = {
  count: number;
  events: ActionEvent[];
  standardCount: number;
  shallowCount: number;
};

export type SessionState = "idle" | "running" | "paused" | "done";

// Singleton session store (shared across views via module scope)
const state      = ref<SessionState>("idle");
const profile    = ref<SessionProfile>({
  weightKg: 70, heightCm: 170, mode: "free", selectedActions: ["squat", "jumping_jack"],
});
const actionMap  = ref<Record<string, ActionSummary>>({});
const startedAt  = ref<number | null>(null);
const endedAt    = ref<number | null>(null);
const elapsedMs  = ref(0);

let timerInterval: ReturnType<typeof setInterval> | null = null;

function _initActions() {
  const map: Record<string, ActionSummary> = {};
  for (const name of profile.value.selectedActions) {
    map[name] = { count: 0, events: [], standardCount: 0, shallowCount: 0 };
  }
  actionMap.value = map;
}

function setProfile(p: Partial<SessionProfile>) {
  profile.value = { ...profile.value, ...p };
}

function start() {
  _initActions();
  startedAt.value = Date.now();
  endedAt.value = null;
  elapsedMs.value = 0;
  state.value = "running";
  timerInterval = setInterval(() => {
    if (startedAt.value) elapsedMs.value = Date.now() - startedAt.value;
  }, 500);
}

function pause() {
  if (state.value !== "running") return;
  state.value = "paused";
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function resume() {
  if (state.value !== "paused") return;
  // Adjust startedAt so elapsedMs continues correctly
  if (startedAt.value) startedAt.value = Date.now() - elapsedMs.value;
  state.value = "running";
  timerInterval = setInterval(() => {
    if (startedAt.value) elapsedMs.value = Date.now() - startedAt.value;
  }, 500);
}

function end() {
  endedAt.value = Date.now();
  if (startedAt.value) elapsedMs.value = endedAt.value - startedAt.value;
  state.value = "done";
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

function reset() {
  state.value = "idle";
  startedAt.value = null;
  endedAt.value = null;
  elapsedMs.value = 0;
  actionMap.value = {};
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

/**
 * Called every time the backend sends a new frame result.
 * Detects count increments and records events.
 */
function updateFromBackend(
  backendActions: Array<{ name: string; count: number; stage: string; score: number | null }>,
) {
  if (state.value !== "running") return;
  for (const ba of backendActions) {
    if (!actionMap.value[ba.name]) continue;
    const local = actionMap.value[ba.name];
    if (ba.count > local.count) {
      const added = ba.count - local.count;
      for (let i = 0; i < added; i++) {
        local.events.push({ action: ba.name, timestamp: Date.now() });
      }
      local.count = ba.count;
    }
  }
}

/** Simple calorie estimate: MET × 3.5 × weightKg × minutes / 200 */
const MET: Record<string, number> = { squat: 5.0, jumping_jack: 8.0 };

function estimatedCalories(): number {
  const minutes = elapsedMs.value / 60_000;
  const weight = profile.value.weightKg;
  let total = 0;
  for (const [name, summary] of Object.entries(actionMap.value)) {
    if (!summary.count) continue;
    const met = MET[name] ?? 4.0;
    total += (met * 3.5 * weight * minutes) / 200;
  }
  return Math.max(0, total);
}

function formatElapsed(ms: number): string {
  const total = Math.floor(ms / 1000);
  const m = Math.floor(total / 60).toString().padStart(2, "0");
  const s = (total % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

export function useSession() {
  return {
    state: readonly(state),
    profile: readonly(profile),
    actionMap: readonly(actionMap),
    elapsedMs: readonly(elapsedMs),
    setProfile,
    start,
    pause,
    resume,
    end,
    reset,
    updateFromBackend,
    estimatedCalories,
    formatElapsed,
  };
}
