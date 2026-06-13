# CalorieCal

基于视觉动作捕捉的实时运动热量估算系统。前端负责摄像头采集和姿态识别，后端负责动作计数和热量计算。

---

## 项目结构

```
CalorieCal/
├── frontend/          Vue 3 前端
├── backend/           Python FastAPI 后端
│   ├── main.py                    启动入口（同时拉起 depth server）
│   ├── action_server.py           FastAPI + WebSocket 动作识别服务
│   └── action_engine/
│       ├── realtime_engine.py     实时状态机引擎
│       └── action_specs/
│           ├── squat.json         深蹲动作参数
│           └── jumping_jack.json  开合跳动作参数
└── task5-SAM 3D Body尝试/
    ├── openni_depth_server.py     OpenNI depth HTTP 服务
    └── openni_depth_probe.py      OpenNI 驱动封装（支持 Linux/Windows）
```

---

## 启动方式

```bash
# 终端 1：后端（自动尝试拉起 depth server，摄像头不在时降级 2D）
conda activate CalorieCalPose
cd /path/to/CalorieCal
python backend/main.py

# 终端 2：前端
cd frontend
npm run dev
```

打开浏览器访问 `http://localhost:5173`，从首页开始训练流程。

---

## 后端

### 服务端口

| 服务 | 端口 | 说明 |
|---|---|---|
| 动作识别 FastAPI | 8766 | WebSocket `/ws`，HTTP `/health`，Swagger `/docs` |
| OpenNI depth server | 8765 | HTTP `/depth`，查询 RGB-D 摄像头深度值 |

### 动作识别服务（port 8766）

- **Swagger UI**: `http://127.0.0.1:8766/docs`
- **WebSocket**: `ws://127.0.0.1:8766/ws`

每个 WebSocket 连接持有独立的 rolling buffer（3秒）和状态机实例。

**前端 → 后端（每帧）**
```json
{
  "frame_index": 123,
  "timestamp_ms": 4567.8,
  "landmarks": {
    "LEFT_SHOULDER": {"x": 0.42, "y": 0.31, "z": -0.05, "visibility": 0.98},
    "LEFT_HIP": {...},
    ...
  }
}
```

**后端 → 前端（每帧）**
```json
{
  "frame_index": 123,
  "actions": [
    {"name": "squat",        "count": 3, "stage": "down",   "score": 0.84},
    {"name": "jumping_jack", "count": 0, "stage": "closed", "score": 0.09}
  ],
  "latency_ms": 3.2,
  "buffer_size": 72
}
```

### 状态机逻辑

`realtime_engine.py` 复用 `task10` 离线状态机的特征计算和评分逻辑：

- 特征：膝角（`mean_knee_angle`）、髋高（`hip_height_above_ankles_torso`）、脚距比（`foot_spread_shoulder_ratio`）、腕距比（`wrist_spread_shoulder_ratio`）、手高比（`hands_above_shoulders_ratio`）
- 归一化：以肩宽/躯干长为尺度，对体型和距离鲁棒
- 评分：深蹲 `0.65 × 膝屈 + 0.35 × 髋降`；开合跳 `0.45 × 脚距 + 0.35 × 腕距 + 0.20 × 手高`
- 计数：状态机检测 `stand→down→stand`（深蹲）或 `closed→open→closed`（开合跳）完整周期

### 动作参数（action spec）

每个动作的阈值存在 `backend/action_engine/action_specs/*.json` 中，不硬编码在代码里。新增动作只需新建 JSON 文件。

当前支持：`squat`（正面/侧面）、`jumping_jack`（正面）。

### depth server（port 8765）

读取 Orbbec Astra Pro（OpenNI 协议）深度帧，为前端提供关键点的深度值（mm）。

- `GET /health` — 健康检查
- `POST /depth` — 批量查询归一化坐标的深度值

如果摄像头不在位，`main.py` 启动时自动降级为 2D 模式，动作识别正常工作。

---

## 前端

### 技术栈

- Vue 3 + TypeScript + Vite
- MediaPipe Tasks Vision（Pose Landmarker + Hand Landmarker）
- Vue Router（hash history）
- 无 UI 库，自定义 CSS

### 路由

| 路径 | 说明 |
|---|---|
| `/#/home` | 训练首页：选动作、训练模式、体重身高 |
| `/#/session` | 训练主界面：摄像头 + 实时计数 + 热量 |
| `/#/report` | 训练报告：汇总次数、热量、时长 |
| `/#/demo` | 调试用 Demo 界面（保留） |
| `/#/pose` | 调试用 Pose 界面（保留） |

### Composables

| 文件 | 职责 |
|---|---|
| `useMediaPipe.ts` | 摄像头 + Pose + Hand 推理 + 骨架绘制，完全复刻 DemoView 效果 |
| `useActionWs.ts` | 后端 WebSocket 连接管理、自动重连、发送 landmarks |
| `useSession.ts` | Session 状态机（idle/running/paused/done）、计时、热量估算、动作事件记录 |

### MediaPipe 识别

`useMediaPipe.ts` 与 `DemoView.vue` 使用完全相同的推理参数和绘制逻辑：

- Pose：33 点，含头部点（鼻、眼、耳显示为粉色）；躯干中轴青色，左侧蓝色，右侧黄色，脚部灰色
- Hand：双手，黄色骨架
- 帧率：20 FPS，5 帧滑动平均平滑
- 摄像头画面镜像显示（符合直觉），关键点坐标保持 MediaPipe 原始方向发送给后端

### 热量计算（当前版本）

基于 Ainsworth MET 公式：`kcal = MET × 3.5 × 体重(kg) × 时长(min) / 200`

MET 取值：深蹲 5.0，开合跳 8.0。后续计划根据幅度和速度动态修正。

---

## 环境要求

- Python 3.10+，conda 环境 `CalorieCalPose`
- 依赖：`fastapi uvicorn websockets numpy`（已在 conda 环境中）
- Node.js 18+
- 可选：Orbbec Astra Pro + OpenNI 驱动（`libopenni2-0` + Orbbec 官方 `liborbbec.so`）
