# CalorieCal

## Task1 MVP Runtime Versions

Task1 固定使用以下姿态识别资源：

```text
@mediapipe/tasks-vision: 0.10.22-rc.20250304
MediaPipe wasm: https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm
Pose Landmarker model: https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
Hand Landmarker model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Face Detector model: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```

版本边界：

```text
1. package.json 中 @mediapipe/tasks-vision 不使用 ^ 或 ~ 浮动范围。
2. 模型和 wasm 使用明确版本路径。
3. 如果后续更换模型或 tasks-vision 版本，需要同步更新本文档和 Phase 3 验证记录。
```
