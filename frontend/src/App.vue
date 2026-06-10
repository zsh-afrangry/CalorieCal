<script setup lang="ts">
import { ref } from "vue";

const navItems = [
  { to: "/pose", label: "姿态检测", caption: "实时调试" },
  { to: "/segmentation", label: "分割测试", caption: "Task8 实验" },
  { to: "/demo", label: "演示版", caption: "外场展示" },
];

const isSidebarCollapsed = ref(false);
</script>

<template>
  <div
    class="root-layout"
    :class="{ 'sidebar-collapsed': isSidebarCollapsed }"
  >
    <aside
      class="app-sidebar"
      :class="{ collapsed: isSidebarCollapsed }"
      aria-label="功能导航"
    >
      <div class="brand-block">
        <div>
          <p class="eyebrow">CalorieCal</p>
          <h1>视觉识别</h1>
        </div>
        <button
          type="button"
          class="sidebar-toggle"
          :aria-label="isSidebarCollapsed ? '展开侧边栏' : '收起侧边栏'"
          :aria-expanded="!isSidebarCollapsed"
          @click="isSidebarCollapsed = !isSidebarCollapsed"
        >
          {{ isSidebarCollapsed ? ">" : "<" }}
        </button>
      </div>

      <nav class="route-nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.to"
          :to="item.to"
          class="route-link"
          :title="item.label"
        >
          <span class="route-icon" aria-hidden="true">{{ item.label.slice(0, 1) }}</span>
          <span class="route-text">{{ item.label }}</span>
          <small>{{ item.caption }}</small>
        </RouterLink>
      </nav>
    </aside>

    <RouterView />
  </div>
</template>
