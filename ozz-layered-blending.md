# ozz-animation 分层姿态混合：关节遮罩与状态转移分析

分析对象：`ozz::animation::BlendingJob`
（`include/ozz/animation/runtime/blending_job.h`，`src/animation/runtime/blending_job.cc`）。
`MotionBlendingJob` 只混合根运动 delta，不含遮罩机制，不在本文范围。

## 1. 混合管线总览

`BlendingJob::Run()` 按固定顺序执行四个阶段（`blending_job.cc:431`）：

1. **BlendLayers**（`blending_job.cc:207`）：逐层累加加权变换，同时维护每个 SoA 关节槽的累计权重 `accumulated_weights[i]` 与全局累计权重 `accumulated_weight`。
2. **BlendRestPose**（`blending_job.cc:277`）：对累计权重不足阈值的关节，用静止姿态（rest pose）补足差额——这就是"低权重关节回到静止姿态"的唯一机制。
3. **Normalize**（`blending_job.cc:334`）：旋转做四元数归一化，平移/缩放除以累计权重。
4. **AddLayers**（`blending_job.cc:362`）：在归一化结果上叠加/扣除加性层。

关键不变量：输出关节数由 `rest_pose` 的 SoA 宽度决定，所有输入缓冲必须至少与之等大（`Validate()`，`blending_job.cc:57`）。

## 2. 局部遮罩（per-joint weights）

每层 `Layer::joint_weights` 是**可选**的 `span<const math::SimdFloat4>`（SoA 布局，一个槽位对应一个关节）：

- **空 span（默认）**：该层对所有关节隐式权重 1.0，且不计入 `num_partial_passes`。这是归一化的参考值（`blending_job.h:95-100`）。
- **非空 span**：进入"部分混合"路径，有效权重为
  `weight = layer.weight × Max0(joint_weights[i])`（`blending_job.cc:235`、`blending_job.cc:244`）。
- **负值按 0 处理**（`math::Max0`），**正值不截断**——允许超过 1.0，因为当所有层都带遮罩时归一化会把它拉回正确比例。

遮罩本身不参与空间衰减或层级传播：ozz 不提供"子树自动继承"，遮罩的层级语义完全由调用方构建。参考实现见 `samples/partial_blend/sample_partial_blend.cc:193-232`：用 `IterateJointsDF`（`skeleton_utils.h:71`）从子树根（如脊柱）深度优先遍历，把子树关节设为 `coeff`、其余设为 `1 - coeff`，形成上下半身互补遮罩。

## 3. 零关节权重的状态转移

关节权重为 0（或被 `Max0` 钳到 0）时发生的状态转移：

1. **贡献为零**：`OZZ_BLEND_1ST_PASS` / `OZZ_BLEND_N_PASS` 以 `weight = 0` 累加，该层的变换不进入输出，`accumulated_weights[i]` 也不增加。
2. **触发静止姿态兜底**：在 `BlendRestPose` 中，若该关节的累计权重低于阈值，则补入 `bp_weight = Max0(threshold - accumulated_weights[i])` 权重的 rest pose（`blending_job.cc:321-325`）。
3. **归一化分母被钳制**：`accumulated_weights[i] = Max(threshold, accumulated_weights[i])`，保证除法有界。

即"零权重关节"并非输出零变换，而是**转移到静止姿态**——这正是分层混合中遮罩外关节保持合理姿态的核心设计。注意一个边界情形：若某关节在所有遮罩层中权重都是 0，但存在无遮罩层（隐式权重 1），则该关节仍被无遮罩层完全驱动；遮罩只对本层生效，层间互不屏蔽。

## 4. 累计权重阈值

`BlendingJob::threshold`（默认 `0.1f`，必须 `> 0`，`blending_job.cc:64`）决定 rest pose 何时介入，有两条路径：

**全局路径（无任何遮罩层，`num_partial_passes == 0`）**——`blending_job.cc:283-308`：

- `bp_weight = threshold - accumulated_weight`（标量）。
- `bp_weight > 0` 时把 rest pose 以 `bp_weight` 混入，并把 `accumulated_weight` 钳到 `threshold`。
- 特例：一个有效层都没有时直接整体拷贝 rest pose，权重视为 1（`blending_job.cc:288-293`）。

**逐关节路径（存在至少一个遮罩层）**——`blending_job.cc:309-327`：

- 每个关节独立计算 `bp_weight = Max0(threshold - accumulated_weights[i])`。
- 一旦进入此路径，**所有关节**（包括遮罩恒为 1 的）都按逐关节逻辑兜底，全局 `accumulated_weight` 不再用于归一化。

状态转移是**软切换**而非硬门限：rest pose 权重随累计权重线性衰减到 0，在 `accumulated = threshold` 处恰好为 0，过渡连续（详见第 6 节）。

## 5. 多个遮罩叠加

多层遮罩在同一关节上**算术累加**：

```
accumulated_weights[i] = Σ_layers  Max0(layer.weight) × Max0(joint_weights[layer][i])
```

（`layer.weight <= 0` 的层被整体跳过，`blending_job.cc:218`。）

叠加语义要点：

- **互补遮罩**（如上下半身 `w` 与 `1-w`）叠加后每关节总权重恒等于层权重之和，归一化后各层贡献比例等于遮罩值之比——这是遮罩设计的推荐用法。
- **重叠遮罩**（两层在同一关节都 > 0）不会饱和：权重可超过 1，归一化阶段按比例分摊，效果等价于该关节上两层按权重比插值。
- **遮罩层与无遮罩层混用**合法：无遮罩层按隐式 1.0 计入每个关节的累计权重。
- 加性层（`additive_layers`）的遮罩只缩放加性增量（`OZZ_ADD_PASS` / `OZZ_SUB_PASS` 中的 `weight`），不参与累计权重与归一化，语义独立。

## 6. 归一化

`Normalize`（`blending_job.cc:334`）：

- **旋转**：始终 `NormalizeEst`（归一化累加后的四元数）。混合过程是加权 nlerp：每层累加前用 `Sign(Dot(acc, in))` 翻转输入四元数符号（`OZZ_BLEND_N_PASS`，`blending_job.cc:106-109`），保证所有四元数落在同一半球、走最短路径。
- **平移/缩放**：乘以 `1 / accumulated`（无遮罩层时全局标量；有遮罩层时逐关节 `one / accumulated_weights[i]`）。
- 由于第 4 节的钳制，分母恒 `≥ threshold > 0`，不存在除零。

## 7. 不同姿态状态下的旋转插值连续性

| 状态转移 | 连续性 | 机理 |
|---|---|---|
| 层权重在 `(0, +∞)` 内变化 | **连续** | 输出是权重的有理函数（加权和 / 累计权重），分母有正下界 |
| 累计权重穿越 `threshold` | **连续** | rest pose 权重 `threshold - acc` 在阈值处恰好为 0；分母 `max(threshold, acc)` 在阈值处两侧相等。导数不连续（折点），但姿态无跳变 |
| 层权重降到 `≤ 0`（层被跳过） | **连续** | 权重趋近 0 时该层贡献趋近 0，跳过瞬间输出无突变；但层数变化会切换首层/累加代码路径，首层四元数符号作为参考半球随之改变——因符号翻转前后表示同一旋转，姿态仍连续 |
| 遮罩值穿越 0（负值被 `Max0` 截断） | **连续** | `Max0` 本身连续；在 0 处导数不连续 |
| 四元数符号翻转（点积变号） | **连续** | `q` 与 `-q` 是同一旋转，翻转只改变累加的数值路径 |
| 加性层权重由正转负（add → subtract） | **连续** | 权重 → 0 时 `interp_quat` → 单位四元数，`OZZ_ADD_PASS` 与 `OZZ_SUB_PASS` 在 0 点汇合 |

结论：在所有权重连续变化的前提下，输出姿态对权重是**处处连续**的（部分折点处仅一阶导数不连续）。ozz 内部不存在会引起姿态跳变的硬切换；**跳变只能来自调用方离散地增删层或突变遮罩**，因此平滑过渡应由调用方在时间轴上对层权重/遮罩做 cross-fade 实现。

## 8. 阈值取值的收益与代价

| 阈值 | 收益 | 代价 |
|---|---|---|
| **大（如 1.0）** | 总权重恒被补足到 1，归一化分母稳定，输出永远是"满幅"有效姿态；权重微弱时不会出现被放大的数值噪声 | rest pose 明显渗入：层权重 0.5 时输出是一半动画一半静止姿态，动画被"稀释"、显得疲软；低权重淡入淡出区间动画表现力弱 |
| **小（默认 0.1）** | 动画保真度高，层权重略大于阈值即几乎完全由动画驱动；rest pose 只在权重趋近 0 时兜底 | 累计权重在阈值附近时归一化增益最大（分母最小 0.1），输入动画的噪声/误差被相对放大；权重远小于阈值时输出几乎全是 rest pose，动画"消失"得较早 |
| **≤ 0** | —— | 非法（`Validate()` 拒绝）：归一化将除以 0 或负数 |

阈值与归一化是耦合的：分母被钳到 `max(threshold, acc)`，阈值越大，低权重区间的增益越低、姿态越"稳"但越接近 rest pose。0.1 的默认值是在"权重趋零时平滑回到静止姿态"与"不过度稀释动画"之间的折中。

## 9. 不破坏既有行为的改进方案

以下方案均保持现有默认行为逐位不变：

1. **互补遮罩约定（纯调用方纪律，零代码改动）**：构建遮罩时保证同一关节上所有遮罩层权重之和为 1（如 `IterateJointsDF` 子树划分），使逐关节累计权重恒等于层权重之和，阈值兜底只在整体淡出时触发。这是收益最大、风险为零的"方案"。
2. **遮罩构建工具下沉**：在 `skeleton_utils` 增加从子树根列表生成互补 SoA 遮罩的 helper（把 `sample_partial_blend.cc` 中的 `WeightSetupIterator` 模式提炼为库函数）。纯新增 API，不触碰 `BlendingJob`。
3. **可选逐关节阈值**：为 `BlendingJob` 新增可选 `span<const math::SimdFloat4> joint_thresholds`，默认空 span 时走现有全部代码路径（全局 `threshold` 语义不变）；非空时在 `BlendRestPose` 逐关节分支中替换广播的阈值常量。默认构造的 job 行为与现在完全一致。
4. **文档化权重域**：在 `Layer::joint_weights` 注释中明确推荐 `[0,1]` 且互补遮罩和为 1；`Validate()` 的 `threshold > 0` 约束与 `Max0` 截断语义保持不变，避免改变任何现有输入的数值结果。
5. **过渡动画留给上层**：ozz 内部连续性已完备（第 7 节），淡入淡出应由调用方对 `Layer::weight` 做时间轴插值，不建议在 job 内引入历史状态（如上一帧权重），以保持 job 无状态、可重入的设计。
