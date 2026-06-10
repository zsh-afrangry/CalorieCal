# Task8：更详细人体识别方案研究

## 本 task 要完成的内容

Task8 负责研究“识别更详细”的方案，用于补足 MediaPipe Pose 只有 33 个关键点、躯干和四肢只能粗略表示的问题。它与 Task7 分工不同：Task7 只优化 /#/demo 的显示效果，Task8 才评估人体分割、RGB-D 区域采样、Whole-Body Pose Estimation、DensePose、人体 mesh 等更细粒度人体识别或建模方案。

当前优先级先收敛到轻量、可实时、能接入现有链路的方案：MediaPipe segmentation mask、RGB-D depth 区域采样、现有 Pose/Hand 阈值边界梳理、Whole-Body Pose Estimation 可行性评估。SAM 3D Body / Human Mesh Recovery 暂不作为 Task8 的实现重点，只作为未来离线建模或个体化人体参数估计方向记录。

## 预期结果

1. 明确当前 MediaPipe Pose 的上限：关键点足够做运动量估算，但不提供大腿、小腿、躯干表面的细粒度结构。
2. 明确现有误判中哪些属于阈值 / 时序可调，哪些需要人体分割、depth 区域采样或更重模型。
3. 评估 MediaPipe segmentation mask 是否能用于人体轮廓、背景过滤和 depth 采样辅助。
4. 评估 RGB-D depth 是否能结合人体分割或骨架区域采样形成更可靠的身体表面/空间幅度估算。
5. 评估 Whole-Body Pose Estimation 是否值得作为 MediaPipe Pose 的增强替代方案；COCO-WholeBody 常见 133 点包括身体、脚、脸和手，可补足脚部、手部和面部关键点密度。
6. 评估 DensePose 是否值得作为后续细粒度人体区域方案，例如躯干、大腿、小腿、手臂区域。
7. 记录 SAM 3D Body / Human Mesh Recovery 的潜在价值，但 Task8 暂缓实现。
8. 给出是否值得接入后端、离线实验或实时演示链路的判断。

## 可能的实现方案和技术栈

1. 继续保留 MediaPipe Pose / Hand 作为实时主链路。
2. 优先评估 MediaPipe Pose segmentation mask，尽量保持浏览器实时链路。
3. RGB-D 方案继续使用 OpenNI depth server，优先增加人体区域过滤、骨架周围局部采样和多点 depth 稳定性实验。
4. Whole-Body Pose Estimation 优先评估 OpenMMLab MMPose / RTMPose / RTMW 一类方案；先判断是否可用 ONNXRuntime、TensorRT、ncnn 或 Python 后端跑实时 / 准实时推理。
5. Python 后端暂作为 Whole-Body Pose、DensePose、普通人体分割或更重模型的研究入口，不直接影响现有 /#/demo。
6. SAM 3D Body 先暂缓，不直接进入实时 MVP；若后续需要用户正面照 / 侧面照 / 背面照建模，再作为离线方向重新评估。
7. 若模型过重，则只作为论文展示、离线分析或后续产品化方向。

## 边界

1. Task8 不负责 /#/demo 的视觉布局和大屏 UI，这些属于 Task7。
2. Task8 不急于接入实时页面，先做方案研究和小实验。
3. Task8 不以“模型更高级”为目标，必须服务运动时间、频率、幅度、强度、估算消耗和置信度。
4. 若新模型不能明显改善侧身、遮挡、身体参考面或空间幅度估算，则暂缓接入。
5. Task8 当前不优先做 SAM 3D Body、重型 HMR 或完整人体 mesh 实时接入。

## Phase 2 初步结论

1. 优先尝试 Pose Landmarker 自带 `outputSegmentationMasks`，因为它能复用当前 MediaPipe Pose 链路，接入成本最低。
2. 独立 MediaPipe Image Segmenter 或普通人体分割模型可作为对比实验，但实时视频同步分割可能阻塞 UI，若使用应放到独立测试 tab 或 Web Worker。
3. segmentation mask 的第一目标不是替代关键点，而是辅助 RGB-D depth 背景过滤、人体区域判断和关键点质量反馈。

## Phase 4 初步结论

1. Whole-Body Pose Estimation 值得继续评估，COCO-WholeBody / MMPose 常见 133 点可补足脚、手、脸等关键点密度。
2. 它的定位是增强关键点密度，不替代 segmentation mask 或 RGB-D；mask + depth 仍负责人体区域和前后距离可信度。
3. 推荐下一步先做 Python + MMPose / RTMW 单帧或视频验证，输出 133 点 JSON，再评估 FPS、稳定性和对动作识别的实际收益。
4. 暂不建议直接接入 /#/demo；若 Python/ONNX 链路稳定，再新增 /#/wholebody-test 独立测试页。

## Phase 5 初步结论

1. DensePose 的价值是人体表面区域和 dense correspondence，不是更多关节点；它更适合区域级人体分析，而不是直接替代当前 Pose/Hand。
2. 它可用于躯干、大腿、小腿、手臂等区域参与度、人体表面展示和更细的 depth 区域过滤，但工程成本明显高于 segmentation mask。
3. 当前不实现 DensePose，不新增前端 tab；后续若人体区域细分成为瓶颈，再做 Python/Detectron2 离线图片或视频验证。
4. 当前优先级仍是 segmentation mask + RGB-D 质量门控，以及 Whole-Body Pose 的 133 点 JSON 验证。

## Phase 7 收尾结论

1. Whole-Body Pose 的 Python / MMPose / JSON 链路已经验证可用；RTX 3060 + Ubuntu 22.04 + torch 2.1.2+cu121 路线可以正常使用 CUDA 输出 133 点 JSON 和可视化结果。
2. GPU 短视频验证显示第一帧有模型预热成本，后续帧约 40ms / 帧，证明该路线具备准实时研究价值。
3. 133 个关键点的新增价值主要集中在面部和双手：body 17、foot 6、face 68、hands 42。它不是更密的躯干、大腿、小腿表面识别方案。
4. 对当前 ToC 商业化前期目标而言，继续追求 Whole-Body 后端服务、/#/wholebody-test 或更重模型，短期收益不如优化现有点位的后端算法层。
5. Task8 不继续推进 133 点实时接入；Whole-Body 保留为研究记录和后续参考。
6. 后续重心转入 Task10：后端算法层快速验证，重点研究如何用现有 MediaPipe Pose / Hand、segmentation mask 和 RGB-D depth 点位序列获得更稳定、更快、更可解释的动作识别结果。
