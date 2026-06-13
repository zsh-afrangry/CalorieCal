<script setup lang="ts">
import { ref, reactive } from "vue";
import { useRouter } from "vue-router";
import "../session-ui.css";
import { useSession, type TrainingMode } from "../composables/useSession";

const router  = useRouter();
const session = useSession();

const weightKg = ref(session.profile.value.weightKg);
const heightCm = ref(session.profile.value.heightCm);
const mode     = ref<TrainingMode>(session.profile.value.mode);

const ACTION_OPTIONS = [
  { key: "squat",        label: "深蹲",  sub: "Squat",         icon: "🏋️" },
  { key: "jumping_jack", label: "开合跳", sub: "Jumping Jack",  icon: "🤸" },
];

const selected = reactive<Record<string, boolean>>({
  squat:        true,
  jumping_jack: true,
});

const MODE_OPTIONS: { key: TrainingMode; label: string }[] = [
  { key: "free",   label: "自由模式" },
  { key: "target", label: "目标次数" },
  { key: "timed",  label: "计时模式" },
];

function toggleAction(key: string) {
  selected[key] = !selected[key];
}

function startTraining() {
  session.reset();
  session.setProfile({
    weightKg:        Number(weightKg.value) || 70,
    heightCm:        Number(heightCm.value) || 170,
    mode:            mode.value,
    selectedActions: Object.entries(selected).filter(([, v]) => v).map(([k]) => k),
  });
  session.start();
  router.push("/session");
}
</script>

<template>
  <div class="cc-page home-view">
    <div class="home-inner">

      <!-- Header -->
      <header class="home-header">
        <h1 class="home-title">CalorieCal</h1>
        <p class="home-tagline">实时动作识别 · 热量追踪</p>
      </header>

      <!-- Action selection -->
      <section class="home-section">
        <p class="cc-section-label">选择训练动作</p>
        <div class="action-cards">
          <div
            v-for="opt in ACTION_OPTIONS"
            :key="opt.key"
            class="action-card"
            :class="{ 'action-card--selected': selected[opt.key] }"
            role="checkbox"
            :aria-checked="selected[opt.key]"
            tabindex="0"
            @click="toggleAction(opt.key)"
            @keydown.enter.space.prevent="toggleAction(opt.key)"
          >
            <div class="action-card__icon">{{ opt.icon }}</div>
            <div class="action-card__body">
              <span class="action-card__name">{{ opt.label }}</span>
              <span class="action-card__en">{{ opt.sub }}</span>
            </div>
            <div v-if="selected[opt.key]" class="action-card__check" aria-hidden="true">✓</div>
          </div>
        </div>
      </section>

      <!-- Training mode -->
      <section class="home-section">
        <p class="cc-section-label">训练模式</p>
        <div class="cc-segmented" role="radiogroup" aria-label="训练模式">
          <button
            v-for="m in MODE_OPTIONS"
            :key="m.key"
            class="cc-segmented__option"
            :class="{ 'cc-segmented__option--active': mode === m.key }"
            role="radio"
            :aria-checked="mode === m.key"
            type="button"
            @click="mode = m.key"
          >{{ m.label }}</button>
        </div>
      </section>

      <!-- User profile -->
      <section class="home-section">
        <p class="cc-section-label">用户信息</p>
        <div class="profile-inputs">
          <div class="profile-field">
            <label class="profile-label" for="weight">体重 (kg)</label>
            <input id="weight" v-model.number="weightKg" class="cc-input" type="number" min="20" max="300" placeholder="70" aria-label="体重（千克）" />
          </div>
          <div class="profile-field">
            <label class="profile-label" for="height">身高 (cm)</label>
            <input id="height" v-model.number="heightCm" class="cc-input" type="number" min="100" max="250" placeholder="170" aria-label="身高（厘米）" />
          </div>
        </div>
      </section>

      <!-- Status -->
      <section class="home-section">
        <div class="cc-status-row">
          <div class="cc-status-item">
            <span class="cc-dot" aria-hidden="true"></span>
            摄像头就绪
          </div>
          <div class="cc-status-item">
            <span class="cc-dot" aria-hidden="true"></span>
            后端服务可用
          </div>
        </div>
      </section>

      <!-- CTA -->
      <section class="home-section home-section--cta">
        <button
          type="button"
          class="cc-btn-primary"
          :disabled="!Object.values(selected).some(Boolean)"
          @click="startTraining"
        >
          开始训练
        </button>
      </section>

      <!-- Debug link -->
      <div class="home-debug-link">
        <RouterLink to="/pose" class="debug-link">调试视图 /pose</RouterLink>
      </div>

    </div>
  </div>
</template>

<style scoped>
.home-view {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 40px 16px 60px;
}

.home-inner {
  width: 100%;
  max-width: 480px;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.home-header { text-align: center; padding: 16px 0 8px; }

.home-title {
  font-size: 32px;
  font-weight: 800;
  color: #fff;
  margin: 0;
  letter-spacing: -0.02em;
  font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.home-tagline { margin: 8px 0 0; color: #6b7280; font-size: 14px; letter-spacing: 0.04em; }

.home-section { display: flex; flex-direction: column; }
.home-section--cta { margin-top: 4px; }

.action-cards { display: flex; gap: 12px; }

.action-card {
  flex: 1;
  background: #1a1d27;
  border: 2px solid #2a2d3a;
  border-radius: 12px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  position: relative;
  text-align: center;
}

.action-card--selected {
  border-color: #22c55e;
  background: rgba(34, 197, 94, 0.06);
}

.action-card__icon { font-size: 28px; line-height: 1; }
.action-card__body { display: flex; flex-direction: column; gap: 2px; }
.action-card__name { font-size: 16px; font-weight: 700; color: #e5e7eb; }
.action-card__en   { font-size: 11px; color: #6b7280; }

.action-card__check {
  position: absolute;
  top: 10px; right: 12px;
  width: 20px; height: 20px;
  background: #22c55e;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700;
  color: #0f1117; line-height: 1;
}

.profile-inputs { display: flex; gap: 12px; }
.profile-field  { flex: 1; display: flex; flex-direction: column; gap: 6px; }
.profile-label  { font-size: 12px; font-weight: 600; color: #9ca3af; }

.cc-btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }

.home-debug-link { text-align: center; }
.debug-link      { font-size: 12px; color: #4b5563; text-decoration: none; }
.debug-link:hover { color: #6b7280; }
</style>
