<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import "../session-ui.css";
import { useSession } from "../composables/useSession";

const router  = useRouter();
const session = useSession();

const ACTION_LABELS: Record<string, string> = {
  squat:        "深蹲",
  jumping_jack: "开合跳",
};

const MET: Record<string, number> = { squat: 5.0, jumping_jack: 8.0 };

function actionCalories(name: string): number {
  const minutes = session.elapsedMs.value / 60_000;
  const weight  = session.profile.value.weightKg;
  const met     = MET[name] ?? 4.0;
  const count   = session.actionMap.value[name]?.count ?? 0;
  if (!count) return 0;
  return Math.max(0, (met * 3.5 * weight * minutes) / 200);
}

const totalCalories = computed(() => session.estimatedCalories().toFixed(0));
const duration      = computed(() => session.formatElapsed(session.elapsedMs.value));

const actionEntries = computed(() =>
  Object.entries(session.actionMap.value).map(([name, summary]) => ({
    name,
    label: ACTION_LABELS[name] ?? name,
    count:         summary.count,
    standardCount: summary.standardCount,
    shallowCount:  summary.shallowCount,
    calories:      actionCalories(name).toFixed(0),
  }))
);

function newSession() {
  session.reset();
  router.push("/session");
}

function goHome() {
  session.reset();
  router.push("/home");
}
</script>

<template>
  <div class="cc-page report-view">
    <div class="report-inner">

      <!-- Header -->
      <header class="report-header">
        <div class="report-check" aria-hidden="true">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="16" fill="rgba(34,197,94,0.18)" />
            <path d="M9 16.5L13.5 21L23 12" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <h1 class="report-title">训练完成</h1>
        <div class="report-duration cc-mono" aria-label="`训练时长 ${duration}`">{{ duration }}</div>
        <span class="report-duration-label cc-muted">训练时长</span>
      </header>

      <!-- Per-action breakdown -->
      <section v-if="actionEntries.length" class="report-section" aria-labelledby="breakdown-heading">
        <p class="cc-section-label" id="breakdown-heading">动作详情</p>
        <div class="breakdown-cards">
          <div
            v-for="entry in actionEntries"
            :key="entry.name"
            class="cc-card breakdown-card"
          >
            <div class="breakdown-card__top">
              <span class="breakdown-card__name">{{ entry.label }}</span>
              <span class="breakdown-card__kcal cc-green cc-mono">约 {{ entry.calories }} kcal</span>
            </div>
            <div class="breakdown-card__count cc-mono" :aria-label="`完成 ${entry.count} 次`">{{ entry.count }}</div>
            <div class="breakdown-card__bottom">
              <span class="cc-pill">完成 {{ entry.count }} 次</span>
              <template v-if="entry.name === 'squat' && entry.count > 0">
                <span class="cc-pill">标准 {{ entry.standardCount }} 次</span>
                <span class="cc-pill cc-pill--neutral">浅蹲 {{ entry.shallowCount }} 次</span>
              </template>
            </div>
          </div>
        </div>
      </section>

      <!-- Total calories -->
      <section class="report-section">
        <div class="cc-card total-card">
          <span class="total-card__label cc-muted">总消耗热量</span>
          <div class="total-card__value cc-mono cc-green" :aria-label="`总消耗热量 ${totalCalories} 千卡`">
            {{ totalCalories }}
          </div>
          <span class="total-card__unit cc-muted">kcal</span>
        </div>
      </section>

      <!-- Key metrics -->
      <section class="report-section" aria-labelledby="metrics-heading">
        <p class="cc-section-label" id="metrics-heading">关键指标</p>
        <div class="metrics-row">
          <div class="cc-card metric-chip">
            <span class="metric-chip__label cc-muted">活跃时长</span>
            <span class="metric-chip__value cc-mono">{{ duration }}</span>
          </div>
          <div class="cc-card metric-chip">
            <span class="metric-chip__label cc-muted">体重</span>
            <span class="metric-chip__value cc-mono">{{ session.profile.value.weightKg }}</span>
            <span class="metric-chip__unit cc-muted">kg</span>
          </div>
          <div class="cc-card metric-chip">
            <span class="metric-chip__label cc-muted">模式</span>
            <span class="metric-chip__value" style="font-size:14px;color:#e5e7eb;">
              {{ session.profile.value.mode === "free" ? "自由" : session.profile.value.mode }}
            </span>
          </div>
        </div>
      </section>

      <!-- CTA -->
      <section class="report-section report-section--cta">
        <button type="button" class="cc-btn-primary" @click="newSession">再来一组</button>
        <button type="button" class="home-link-btn" @click="goHome">返回首页</button>
      </section>

    </div>
  </div>
</template>

<style scoped>
.report-view {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 40px 16px 60px;
}

.report-inner {
  width: 100%;
  max-width: 480px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.report-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 16px 0 8px;
  text-align: center;
}

.report-check { margin-bottom: 4px; }

.report-title {
  font-size: 24px;
  font-weight: 800;
  color: #fff;
  margin: 0;
  font-family: system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.report-duration {
  font-size: 52px;
  font-weight: 700;
  color: #fff;
  line-height: 1.1;
  letter-spacing: 0.04em;
}

.report-duration-label { font-size: 13px; margin-top: -2px; }

.report-section { display: flex; flex-direction: column; }
.report-section--cta { gap: 14px; align-items: center; margin-top: 4px; }

.breakdown-cards { display: flex; flex-direction: column; gap: 10px; }

.breakdown-card { padding: 16px; display: flex; flex-direction: column; gap: 8px; }

.breakdown-card__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.breakdown-card__name  { font-size: 15px; font-weight: 700; color: #e5e7eb; }
.breakdown-card__kcal  { font-size: 14px; font-weight: 600; }

.breakdown-card__count {
  font-size: 56px;
  font-weight: 700;
  color: #fff;
  line-height: 1;
  letter-spacing: -0.02em;
}

.breakdown-card__bottom { display: flex; gap: 6px; flex-wrap: wrap; }

.total-card {
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
}

.total-card__label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.total-card__value {
  font-size: 96px;
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}

.total-card__unit { font-size: 20px; font-weight: 500; }

.metrics-row { display: flex; gap: 10px; }

.metric-chip {
  flex: 1;
  padding: 12px 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
}

.metric-chip__label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  white-space: nowrap;
}

.metric-chip__value { font-size: 22px; font-weight: 700; color: #fff; line-height: 1; }
.metric-chip__unit  { font-size: 11px; }

.home-link-btn {
  background: none;
  border: none;
  font-size: 13px;
  color: #4b5563;
  cursor: pointer;
  font-family: inherit;
  padding: 4px 8px;
}

.home-link-btn:hover { color: #6b7280; }
</style>
