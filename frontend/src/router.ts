import { createRouter, createWebHashHistory } from "vue-router";
import PoseView from "./views/PoseView.vue";
import DemoView from "./views/DemoView.vue";
import SegmentationView from "./views/SegmentationView.vue";
import HomeView from "./views/HomeView.vue";
import SessionView from "./views/SessionView.vue";
import ReportView from "./views/ReportView.vue";
import DisplayView from "./views/DisplayView.vue";

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    // Redirect root to the display view (展示模式)
    { path: "/", redirect: "/display" },

    // Fitness UI routes (fullscreen, no sidebar)
    { path: "/display", name: "display", component: DisplayView, meta: { fullscreen: true } },
    { path: "/home",    name: "home",    component: HomeView,    meta: { fullscreen: true } },
    { path: "/session", name: "session", component: SessionView, meta: { fullscreen: true } },
    { path: "/report",  name: "report",  component: ReportView,  meta: { fullscreen: true } },

    // Existing debug/dev routes (keep sidebar layout)
    { path: "/pose",          name: "pose",          component: PoseView },
    { path: "/segmentation",  name: "segmentation",  component: SegmentationView },
    { path: "/demo",          name: "demo",          component: DemoView },

    // Catch-all fallback
    { path: "/:pathMatch(.*)*", redirect: "/display" },
  ],
});

export default router;
