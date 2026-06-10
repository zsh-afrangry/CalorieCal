import { createRouter, createWebHashHistory } from "vue-router";
import PoseView from "./views/PoseView.vue";
import DemoView from "./views/DemoView.vue";
import SegmentationView from "./views/SegmentationView.vue";

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/pose" },
    { path: "/pose", name: "pose", component: PoseView },
    { path: "/segmentation", name: "segmentation", component: SegmentationView },
    { path: "/demo", name: "demo", component: DemoView },
    { path: "/:pathMatch(.*)*", redirect: "/pose" },
  ],
});

export default router;
