# Task9 Phase 5: 动作标准度评分和建议反馈系统

> 创建时间: 2026-06-13  
> 状态: 规划中  
> 目标: 为每个动作提供标准度评分（0-100），当动作不标准时给出具体改进建议

---

## 一、需求分析

### 1.1 用户需求
- **标准度评分**: 用户希望知道"我这个动作做得标不标准"（量化评分 0-100）
- **改进建议**: 当分数低于标准时，告诉用户"哪里不对，应该怎么改"

### 1.2 当前系统能力
- ✅ 后端已有动作评分机制（score 0-1）
- ✅ 每个动作有特征提取（膝角、髋高、脚距、腕距等）
- ✅ 有动作规格文件定义质量门槛
- ❌ score 是内部评分，未转换为用户可理解的标准度
- ❌ 缺少基于特征的诊断性建议

---

## 二、设计方案

### 2.1 标准度评分体系

#### 评分等级定义
```
优秀（Excellent）：   90-100  动作标准，姿态规范
良好（Good）：        75-89   动作基本标准，有小幅改进空间
一般（Fair）：        60-74   动作幅度不足或姿态偏差
需改进（Poor）：       40-59   动作不标准，需要调整
未达标（Insufficient）: 0-39   动作严重不标准或识别不完整
```

#### Score 到标准度的映射逻辑
```python
def compute_quality_score(
    raw_score: float,      # 0-1 的动作评分
    feature_checks: dict,  # 特征检查结果
    visibility: float,     # 关键点可见度
) -> tuple[int, str]:
    """
    返回: (标准度分数 0-100, 等级标签)
    
    规则:
    1. 基础分 = raw_score * 100
    2. 可见度惩罚: visibility < 0.7 时 -10分
    3. 特征惩罚: 每个不达标特征 -5到-15分
    4. 下限保护: 最低0分
    """
```

### 2.2 建议反馈机制

#### 建议生成策略
针对每个动作，基于特征检查结果生成具体建议：

**深蹲 (squat)**
```python
suggestions = []
if knee_angle_min > 140:
    suggestions.append("膝盖弯曲不够，尝试蹲得更低")
if knee_angle_range < 45:
    suggestions.append("动作幅度不足，站立和下蹲的差异要更明显")
if hip_height_range < 0.3:
    suggestions.append("髋部下降幅度不够，重心要降低")
if core_visible_ratio < 0.7:
    suggestions.append("请保持全身入镜，特别是髋膝踝关键点")
if left_right_knee_diff > 20:
    suggestions.append("左右膝盖不对称，注意双腿同步下蹲")
```

**开合跳 (jumping_jack)**
```python
suggestions = []
if foot_spread_range < 0.8:
    suggestions.append("双脚打开幅度不够，跳得更开一些")
if wrist_spread_range < 2.0:
    suggestions.append("手臂打开幅度不够，双臂要完全伸展到头顶")
if hands_above_shoulders_ratio < 0.25:
    suggestions.append("手臂高度不够，打开时双手要超过肩膀")
if motion_energy < 0.03:
    suggestions.append("动作强度偏低，跳得更有力一些")
if left_right_arm_diff > 0.3:
    suggestions.append("左右手不对称，注意双臂同步抬起")
```

**高抬腿 (high_knee)**
```python
suggestions = []
if hip_angle_peak < 150:
    suggestions.append("膝盖抬得不够高，大腿要尽量靠近胸部")
if left_right_count_diff > 2:
    suggestions.append(f"左右腿次数不均衡（左{left_count}次，右{right_count}次）")
if motion_frequency < 30:  # 次/分钟
    suggestions.append("速度偏慢，提高抬腿频率")
if standing_leg_knee_bent:
    suggestions.append("支撑腿要保持伸直，不要弯曲膝盖")
```

---

## 三、实施步骤

### Phase 5.1: 后端标准度评分和建议生成

#### 3.1.1 新增质量评估模块
文件: `backend/action_engine/quality_evaluator.py`

```python
"""Action quality evaluation and feedback generation."""

from typing import Any

# 标准度等级定义
QUALITY_LEVELS = {
    (90, 100): ("excellent", "优秀"),
    (75, 89):  ("good", "良好"),
    (60, 74):  ("fair", "一般"),
    (40, 59):  ("poor", "需改进"),
    (0, 39):   ("insufficient", "未达标"),
}

def evaluate_squat_quality(
    score: float | None,
    features: dict[str, Any],
    landmarks_quality: dict[str, float],
) -> dict[str, Any]:
    """
    评估深蹲质量并生成建议。
    
    返回:
    {
        "quality_score": 85,           # 0-100
        "quality_level": "good",       # excellent/good/fair/poor/insufficient
        "quality_label": "良好",        # 中文标签
        "suggestions": [               # 改进建议列表
            "膝盖弯曲不够，尝试蹲得更低",
            "请保持全身入镜"
        ],
        "feature_checks": {            # 特征检查详情（调试用）
            "knee_angle_ok": false,
            "hip_height_ok": true,
            "visibility_ok": true,
        }
    }
    """
    pass

def evaluate_jumping_jack_quality(...) -> dict[str, Any]:
    """评估开合跳质量并生成建议。"""
    pass

def evaluate_high_knee_quality(...) -> dict[str, Any]:
    """评估高抬腿质量并生成建议。"""
    pass

# ... 其他动作的评估函数
```

#### 3.1.2 修改 realtime_engine.py
在 `push_frame()` 返回的 actions 中增加质量评估字段：

```python
# 当前返回格式
{
    "name": "squat",
    "count": 3,
    "stage": "down",
    "score": 0.84,  # 现有字段
}

# 新增字段
{
    "name": "squat",
    "count": 3,
    "stage": "down",
    "score": 0.84,
    "quality": {                    # 新增
        "quality_score": 84,
        "quality_level": "good",
        "quality_label": "良好",
        "suggestions": ["膝盖弯曲不够，尝试蹲得更低"],
        "feature_checks": {...}
    }
}
```

### Phase 5.2: 前端标准度和建议显示

#### 3.2.1 修改 DisplayView.vue（演示页）
在当前组合动作下方增加"动作质量"区域：

```html
<!-- 动作质量反馈（只在有组合动作时显示） -->
<div v-if="currentCompositeAction && actionQuality" class="quality-feedback">
  <div class="quality-score-badge" :class="`quality-${actionQuality.quality_level}`">
    <span class="score-number">{{ actionQuality.quality_score }}</span>
    <span class="score-label">{{ actionQuality.quality_label }}</span>
  </div>
  
  <div v-if="actionQuality.suggestions.length > 0" class="suggestions-list">
    <div class="suggestions-title">💡 改进建议</div>
    <ul>
      <li v-for="(suggestion, idx) in actionQuality.suggestions" :key="idx">
        {{ suggestion }}
      </li>
    </ul>
  </div>
</div>
```

#### 3.2.2 修改 PoseView.vue（调试页）
在现有的质量反馈基础上，增加详细的特征检查信息：

```html
<div class="quality-detail">
  <h3>动作质量详情</h3>
  <div class="quality-score">标准度: {{ actionQuality.quality_score }}/100</div>
  
  <!-- 特征检查明细 -->
  <div class="feature-checks">
    <div v-for="(passed, feature) in actionQuality.feature_checks" :key="feature"
         class="check-item" :class="{ passed, failed: !passed }">
      <span class="check-icon">{{ passed ? '✓' : '✗' }}</span>
      <span class="check-name">{{ featureNameMap[feature] }}</span>
    </div>
  </div>
  
  <!-- 建议列表 -->
  <div class="suggestions">
    <div v-for="(s, i) in actionQuality.suggestions" :key="i" class="suggestion-item">
      {{ s }}
    </div>
  </div>
</div>
```

#### 3.2.3 修改 useActionWs.ts
解析后端返回的 quality 字段并暴露给组件：

```typescript
export interface ActionQuality {
  quality_score: number;
  quality_level: string;
  quality_label: string;
  suggestions: string[];
  feature_checks: Record<string, boolean>;
}

// 新增 ref
const actionQuality = ref<ActionQuality | null>(null);

// 在 WebSocket 消息处理中更新
if (data.actions && data.actions.length > 0) {
  const primaryAction = data.actions[0];
  if (primaryAction.quality) {
    actionQuality.value = primaryAction.quality;
  }
}

// 导出
return {
  // ... 现有导出
  actionQuality,
};
```

---

## 四、验收标准

### 4.1 后端验收
- [ ] 每个动作的 quality 字段包含 quality_score、quality_level、quality_label、suggestions
- [ ] 标准动作得分 ≥ 85
- [ ] 不标准动作得分 < 75，且至少有 1 条建议
- [ ] 关键点不完整时有可见度相关建议
- [ ] 单元测试覆盖各动作的评估函数

### 4.2 前端验收（DisplayView）
- [ ] 当组合动作识别成功时，显示标准度徽章
- [ ] 分数 ≥ 85 显示绿色，60-84 黄色，< 60 红色
- [ ] 有建议时显示建议列表，最多显示前 3 条
- [ ] UI 简洁，不遮挡主要画面

### 4.3 前端验收（PoseView）
- [ ] 显示详细的特征检查结果（通过/未通过）
- [ ] 显示完整的建议列表
- [ ] 调试信息可折叠

### 4.4 实测验收
针对 Task9 已完成的动作进行逐个测试：

**深蹲 (squat)**
- [ ] 标准深蹲（膝盖 < 90°）得分 ≥ 85
- [ ] 浅蹲（膝盖 > 120°）得分 < 75，有"蹲得更低"建议
- [ ] 侧身深蹲能识别（双摄像头融合）

**开合跳 (jumping_jack)**
- [ ] 标准开合跳（双脚双臂完全打开）得分 ≥ 85
- [ ] 只动手不动脚得分 < 60，有"双脚打开幅度不够"建议
- [ ] 只动脚不动手得分 < 60，有"手臂打开幅度不够"建议

**高抬腿 (high_knee)**
- [ ] 标准高抬腿（大腿接近水平）得分 ≥ 85
- [ ] 抬腿不够高得分 < 75，有"膝盖抬得不够高"建议
- [ ] 左右次数差异大时有"左右腿不均衡"建议

**胯下击掌 (clap_under_knee)**
- [ ] 标准胯下击掌得分 ≥ 85
- [ ] 膝盖不够高或双手未合拢得分 < 75

**弓步压腿 (lunge)**
- [ ] 标准弓步保持得分 ≥ 85
- [ ] 前腿膝盖超过脚尖或后腿膝盖不够低有相应建议

---

## 五、实施优先级

### P0（必须完成）
1. 后端 quality_evaluator.py 模块
2. squat、jumping_jack、high_knee 三个动作的评估函数
3. realtime_engine 集成 quality 字段
4. DisplayView 显示标准度徽章和建议（简化版）

### P1（推荐完成）
5. PoseView 显示详细特征检查
6. clap_under_knee、lunge 两个动作的评估函数
7. 单元测试

### P2（后续优化）
8. 基于历史数据的个性化建议（例如"您的深蹲次数逐渐增加，但膝盖弯曲角度仍不够"）
9. 建议优先级排序（先解决最影响分数的问题）
10. 视频回放标注（标出动作不标准的关键帧）

---

## 六、技术细节

### 6.1 特征检查规则示例（深蹲）

```python
def _check_squat_features(features: dict, spec: dict) -> dict[str, bool]:
    """检查深蹲特征是否达标。"""
    checks = {}
    
    # 膝角检查
    knee_angle_min = features.get("mean_knee_angle_min")
    if knee_angle_min is not None:
        checks["knee_flexion_sufficient"] = knee_angle_min <= 140.0
        checks["knee_flexion_deep"] = knee_angle_min <= 90.0
    
    # 髋高检查
    hip_height_range = features.get("hip_height_range")
    if hip_height_range is not None:
        checks["hip_drop_sufficient"] = hip_height_range >= 0.30
    
    # 膝角变化幅度
    knee_angle_range = features.get("mean_knee_angle_range")
    if knee_angle_range is not None:
        checks["knee_range_sufficient"] = knee_angle_range >= 45.0
    
    # 可见度检查
    core_visible = features.get("mean_core_visible_ratio")
    if core_visible is not None:
        checks["visibility_ok"] = core_visible >= 0.7
    
    # 左右对称性（如果有左右膝角）
    left_knee = features.get("left_knee_angle_min")
    right_knee = features.get("right_knee_angle_min")
    if left_knee is not None and right_knee is not None:
        checks["symmetry_ok"] = abs(left_knee - right_knee) <= 20.0
    
    return checks
```

### 6.2 建议生成规则示例（深蹲）

```python
def _generate_squat_suggestions(checks: dict, features: dict) -> list[str]:
    """根据特征检查结果生成建议。"""
    suggestions = []
    
    if not checks.get("knee_flexion_sufficient", True):
        if features.get("mean_knee_angle_min", 180) > 140:
            suggestions.append("膝盖弯曲不够，尝试蹲得更低（目标：膝盖弯曲至90°）")
    
    if not checks.get("hip_drop_sufficient", True):
        suggestions.append("髋部下降幅度不够，重心要降低，感觉像坐在椅子上")
    
    if not checks.get("knee_range_sufficient", True):
        suggestions.append("动作幅度不足，站立和下蹲的姿态差异要更明显")
    
    if not checks.get("visibility_ok", True):
        suggestions.append("请保持全身入镜，特别是髋部、膝盖和脚踝")
    
    if not checks.get("symmetry_ok", True):
        left_knee = features.get("left_knee_angle_min", 0)
        right_knee = features.get("right_knee_angle_min", 0)
        if left_knee > right_knee + 20:
            suggestions.append("左膝弯曲不够，注意左右膝盖同步下蹲")
        elif right_knee > left_knee + 20:
            suggestions.append("右膝弯曲不够，注意左右膝盖同步下蹲")
    
    # 通用建议（当其他都达标但分数仍不高时）
    if not suggestions and features.get("score", 0) < 0.75:
        suggestions.append("整体动作基本正确，保持练习以提高稳定性")
    
    return suggestions[:3]  # 最多返回3条建议
```

---

## 七、后续扩展方向

1. **动作轨迹可视化**: 在画面上绘制关键点运动轨迹，标注不标准的部分
2. **语音反馈**: 当检测到动作不标准时，语音提示"膝盖再低一点"
3. **个性化阈值**: 根据用户历史表现动态调整评分标准
4. **对比学习**: 在界面上展示标准动作示范视频，与用户动作对比
5. **渐进式训练**: 根据当前水平推荐下一步练习目标

---

## 八、相关文件清单

### 新增文件
- `backend/action_engine/quality_evaluator.py` - 质量评估核心逻辑
- `backend/action_engine/tests/test_quality_evaluator.py` - 单元测试
- `task9-基础动作和组合动作新增/action-quality-and-feedback-plan.md` - 本文档

### 修改文件
- `backend/action_engine/realtime_engine.py` - 集成质量评估
- `frontend/src/composables/useActionWs.ts` - 解析 quality 字段
- `frontend/src/views/DisplayView.vue` - 显示标准度和建议
- `frontend/src/views/PoseView.vue` - 显示详细质量信息
- `task9-基础动作和组合动作新增/todolist.txt` - 更新 Phase 5 进度
