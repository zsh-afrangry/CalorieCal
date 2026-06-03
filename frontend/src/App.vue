<script setup lang="ts">
const navItems = [
  { to: "/pose", label: "姿态检测", caption: "开合跳 MVP" },
  { to: "/gesture", label: "手势识别", caption: "手部关键点" },
  { to: "/face", label: "人脸检测", caption: "人脸框检测" },
];
</script>

<template>
  <div class="root-layout">
    <aside class="app-sidebar" aria-label="功能导航">
      <div class="brand-block">
        <p class="eyebrow">CalorieCal</p>
        <h1>视觉识别 MVP</h1>
      </div>

      <nav class="route-nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="route-link"
        >
          <span>{{ item.label }}</span>
          <small>{{ item.caption }}</small>
        </RouterLink>
      </nav>
    </aside>

    <RouterView />
  </div>
</template>
