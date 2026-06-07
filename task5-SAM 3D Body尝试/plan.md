# Task5：RGB-D 深度摄像头接入评估

## 本 task 要完成的内容

优先探索当前 Orbbec Astra Pro HD / ORBBEC Depth Sensor 是否能为项目提供真实纵深信息。用户已通过 OpenNI NiViewer 验证设备可用，并确认开启 device registration 的 depth -> image 后，可以看到 Image registration is on，且拖拽点位时能显示 2D distance 和 3D distance。

本 task 的重点从 SAM 3D Body 调整为 RGB-D 深度摄像头接入评估。SAM 3D Body 保留为备选研究路线，但当前更优先验证真实 depth 数据是否能改善前后方向运动、动作幅度、遮挡、身体比例和热量估算置信度。

当前主链路仍然是 MediaPipe Pose / Hand Landmarker。RGB-D 摄像头不替代 MediaPipe，而是给 MediaPipe 的 2D 关键点补充 z/depth 信息。

## 预期结果

1. 跑通 Python / OpenNI 最小 depth 读取脚本，确认能读取 ORBBEC Depth Sensor 的 depth 帧。
2. 确认 depth 单位、分辨率、帧率、中心点距离和静止抖动范围。
3. 明确 depth -> image registration 对本项目的意义：MediaPipe 在 RGB 图上给出 x/y，depth 图在同一坐标系下补充 z。
4. 评估 RGB-D 对前后方向动作的增益，例如手向前、手向后、出拳、身体前倾、身体后退。
5. 给出后续接入判断：只做后端探针、接入 /pose 调试、接入 /demo 展示，或暂缓。
6. SAM 3D Body 作为备选路线保留，后续只在真实 depth 摄像头收益不足时再评估。

## 当前结论

1. RGB-D depth 已经证明在正对摄像头时基本可用，可以补足手向前/手向后这类前后方向动作证据。
2. 侧身时容易丢失稳定躯干基准，主要问题是 2D 姿态关键点可见性和身体参考面估计，不是单纯 depth 数值问题。
3. SAM 3D Body / 3D 骨架恢复暂不接入实时 MVP 主链路，但保留为后续解决侧身、遮挡、躯干不可见问题的研究方向。

## 可能的实现方案和技术栈

1. 技术验证：Python + OpenNI2 / primesense binding，优先读取 depth 帧。
2. 输入：Orbbec Astra Pro HD 的 RGB 画面和 ORBBEC Depth Sensor 的 depth 帧。
3. 输出：中心点 depth、指定点 depth、depth 分辨率、有效深度比例。
4. 对比：浏览器只能看到 Astra Pro HD Camera 的 RGB 视频，depth 需要 Python 后端或本地服务读取。
5. 集成：前端继续用 MediaPipe 识别关键点；后端根据关键点 x/y 查询 depth，返回 z/depth 和置信度。
6. 后续可选：HTTP / WebSocket depth 服务、/pose 调试面板、/#/demo 演示指标、SAM 3D Body 对比研究。

## 边界

1. 不作为 Task1-Task4 的必要依赖。
2. 不急于商业化或产品化。
3. 不用 RGB-D 替代当前 MediaPipe 主链路，而是补充 z/depth。
4. 不急于接入前端；先验证 Python 是否能稳定读取 depth。
5. 如果 depth 抖动大、registration 不稳定或后端接入成本过高，则暂缓接入。
6. 不把 Task5 做成“为了使用 3D 而使用 3D”；它必须服务运动时间、频率、幅度、强度、估算消耗和置信度。

## SAM 3D Body 备选路线

SAM 3D Body / 3DB 仍保留为后续研究方向。它更适合离线人体网格恢复或论文展示，不是当前实时演示链路的优先选择。若 RGB-D 摄像头方案无法满足需求，再回到 SAM 3D Body 做对比评估。
