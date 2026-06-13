# Task12：双摄像头融合架构

## 本 task 要完成的内容

在现有单摄像头（正面视角）基础上，引入第二个摄像头（侧面视角），实现双视角融合的运动识别和热量计算。解决单摄像头在用户侧身、扭腰、前后平移等场景下因遮挡导致的识别准确度下降和热量计算中断问题。

核心设计理念：
- **可见度门槛 + 最大运动信号融合**：两路摄像头独立计算 motion_energy，先按 core_visible_ratio 判断视角是否可用，再采用更强的运动信号，避免前后方向运动被正面视角投影低估
- **简化架构**：不做复杂的 3D 坐标重建和摄像头标定，只做视角级融合
- **开发友好**：保留调试开关，生产环境可隐藏侧面画面
- **向后兼容**：支持关闭侧摄像头退化为单摄像头模式

## 预期结果

1. **热量计算连续性提升**：用户侧身、扭腰、背对正面摄像头时，热量计算不中断，motion_kcal 持续累积
2. **识别覆盖范围扩大**：支持非正面动作（扭腰拉伸、侧身弯腰）的热量估算
3. **调试开关可用**：
   - "侧边摄像头"开关：控制是否启用双摄像头融合（关闭=单摄像头模式）
   - "侧边画面"开关：控制是否显示侧面视频流（关闭=界面简洁，仅显示正面）
4. **动作识别保持稳定**：第一版动作识别仍使用正面视角，侧面视角只参与 motion_kcal；双路动作识别留作后续优化

## 可能的实现方案和技术栈

### 前端
- 继续使用 Vite + Vue 3 + TypeScript
- MediaPipe Pose：启动两个 PoseLandmarker 实例（front + side）
- 两个 `<video>` 元素，两路独立的关键点检测
- WebSocket 消息格式扩展：从单路 landmarks 改为 `{ type: "dual_frame", front: {...}, side: {...} }`

### 后端
- FastAPI + WebSocket：接收双路 landmarks
- RealtimeActionEngine 新增 `push_dual_frame()` 方法
- 双路独立历史状态 → 双路独立特征提取 → 可见度门槛 + 最大运动信号融合 → 热量计算
- 动作识别：第一版只使用 front 视角，保持原有计数稳定性

### 融合算法
- **motion_energy 融合（第一版）**：
  ```python
  VIS_THRESHOLD = 0.5

  if core_vis_front >= VIS_THRESHOLD and core_vis_side >= VIS_THRESHOLD:
      motion_e_merged = max(motion_e_front, motion_e_side)
      active_view = "front" if motion_e_front >= motion_e_side else "side"
  elif core_vis_front >= VIS_THRESHOLD:
      motion_e_merged = motion_e_front
      active_view = "front"
  elif core_vis_side >= VIS_THRESHOLD:
      motion_e_merged = motion_e_side
      active_view = "side"
  else:
      motion_e_merged = 0.0
      active_view = "none"
  ```
- **动作识别融合**（可选）：
  - 第一版不做：动作识别只用 front 视角
  - 正面动作（深蹲、开合跳）：继续沿用 front 视角
  - 侧面动作（后续扩展）：可优先用 side 视角
  - 可以两路都跑识别，取 score 更高的结果

## 边界

1. **不做完整 3D 重建**：不做摄像头标定、坐标系转换、关键点三角测量
2. **不要求摄像头严格 90 度**：只需"一个正面、一个侧面"，角度容差 ±30 度
3. **不做硬件时间同步**：前端尽力对齐两路 timestamp，容忍 50ms 内的时间差
4. **不追求学术级精度**：以"符合实际、可解释"为准，不追求毫米级精度
5. **第一版只优化热量计算**：动作识别融合不做，避免影响现有计数稳定性

## 技术挑战和应对

### 1. 前端性能：两路 MediaPipe 推理
- **挑战**：两个 PoseLandmarker 实例同时运行，CPU/GPU 占用翻倍
- **应对**：
  - 第一版先两路同时推理，实测性能
  - 如果卡顿，再降低推理频率或分辨率
  - 错峰推理作为后续优化：front 和 side 交错调度（front 在偶数帧，side 在奇数帧）
  - 使用 lite 模型：已经在用 `pose_landmarker_lite`

### 2. 时间同步：两路摄像头帧时间戳对齐
- **挑战**：USB 摄像头时间戳漂移，两路帧可能不在同一时刻
- **应对**：
  - 前端记录两路各自的 `performance.now()` 作为统一时间戳
  - 后端容忍 ±50ms 的时间差
  - 如果某一路 landmarks 为 null（未检测到 pose），另一路继续工作

### 3. 摄像头选择：如何让前端区分正面/侧面
- **挑战**：浏览器 `navigator.mediaDevices.enumerateDevices()` 返回的设备 ID 不稳定
- **应对**：
  - 用户手动选择"正面摄像头"和"侧面摄像头"（下拉菜单）
  - 保存到 localStorage，下次自动恢复
  - 提供"交换正侧"按钮，方便调试

### 4. 后端兼容性：单/双摄像头模式切换
- **挑战**：后端需要同时支持旧的单路消息和新的双路消息
- **应对**：
  - 保留原有 `push_frame(landmarks)` 方法（单摄像头模式）
  - 新增 `push_dual_frame(landmarks_front, landmarks_side)` 方法
  - 根据消息 `type` 字段路由到不同方法

## 实施风险

### 低风险
- 前端增加一个 video 元素和一个 MediaPipe 实例：成熟方案
- 后端增加视角级融合逻辑：简单算术运算
- 调试开关实现：Vue ref + CSS 显隐

### 中风险
- **前端性能**：两路推理可能导致帧率下降，需要实测和优化
- **摄像头选择 UX**：用户可能不理解"正面"和"侧面"的区别，需要提示文案
- **状态隔离**：双视角必须分离 prev_landmarks / prev_hip_height，否则 motion_energy 会跨视角串扰

### 高风险（第一版不涉及）
- 完整 3D 重建：需要标定、坐标转换、关键点融合，复杂度高

## 后续优化方向（不在第一版范围）

1. **深度信息利用**：如果两个摄像头都是深度摄像头，用深度图校正 landmark 物理坐标
2. **自动视角判断**：根据肩膀连线方向自动判断哪个是正面、哪个是侧面
3. **动作识别融合**：两路都跑识别，取 confidence 加权平均，提升识别准确率
4. **三摄像头扩展**：增加俯视摄像头，用于跳跃高度估算
5. **点云配准**：用深度图做自动标定，不需要手动测量摄像头角度
6. **关键点 3D 融合**：将两个视角的 2D landmarks 融合为 3D 坐标

## Task12 预期产物

1. **plan.md**：本文档
2. **todolist.txt**：详细实施清单
3. **前端改造**：
   - DisplayView.vue：增加侧面 video + MediaPipe 实例 + 两个调试开关
   - MediaPipe 逻辑：抽出可创建多实例的工厂函数或改造 useMediaPipe
   - useActionWs.ts：新增 sendDualFrame 和 viewInfo
4. **后端改造**：
   - realtime_engine.py：新增 push_dual_frame() 方法
   - realtime_engine.py：front/side 历史状态隔离
   - action_server.py：新增 dual_frame 消息处理
5. **测试验证**：
   - 用户正面深蹲：front 主导，结果与单摄像头一致
   - 用户侧身弯腰：side 主导，motion_kcal 不为 0
   - 用户扭腰拉伸：active_view 动态变化，motion_kcal 合理
   - 关闭侧边摄像头：退化为单摄像头模式
6. **文档**：
   - 摄像头部署指南（物理位置、角度、距离）
   - 调试开关使用说明
   - 性能优化记录（帧率、CPU 占用）
