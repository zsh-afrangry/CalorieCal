import { createRouter, createWebHashHistory } from "vue-router";
import PoseView from "./views/PoseView.vue";
import GestureView from "./views/GestureView.vue";
import FaceView from "./views/FaceView.vue";

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", redirect: "/pose" },
    { path: "/pose", name: "pose", component: PoseView },
    { path: "/gesture", name: "gesture", component: GestureView },
    { path: "/face", name: "face", component: FaceView },
  ],
});

export default router;
