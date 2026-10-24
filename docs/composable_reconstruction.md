# FISTA、H1 与 Haar 使用说明

## 接口与支持范围

`solvers.specs`、`ct_runtime` 和 Agent 共用 registry。`fista` 是 solver ID；H1 和 Haar 是正则配置。Tikhonov-CG、`tv_fista` 函数与 `TVFISTASolver` 保留。

| solver | regularizer | 约束 |
| --- | --- | --- |
| `fista` | `tikhonov`（L=I） | 无约束或可分离 box，含非负 |
| `fista` | `tv` | 无约束，有限 FGP 近似近端 |
| `fista` | `haar_l1` | 无约束；默认 level=2 |
| `tikhonov` | `h1` | 无约束；复用广义 Tikhonov-CG |
| `tikhonov` | `tikhonov` | 保留旧接口与默认值 |
| `tv_fista` | TV | 保留旧接口与历史 post-prox box projection 语义 |

迭代前拒绝 FISTA+H1、FISTA+其他广义 L、FISTA+Haar+非负/box、通用 FISTA+TV+box、H1-CG 的约束配置，以及 Haar 非法 level/尺寸。旧 TV 的投影组合继续可调用，但不将其表述为 TV 与 box 的精确复合近端；旧 Tikhonov 的 box 行为也不构成新增受约束 CG 保证。

Python 入口：

```python
from inv_framework.solvers import FISTASolver, TikhonovSolver, fista
from inv_framework.regularizers import TikhonovRegularizer, HaarL1Regularizer
from inv_framework.operators import GradientOperator

x = fista(A, y, regularizer=TikhonovRegularizer(), strength=0.01)
report = FISTASolver(HaarL1Regularizer(level=2), strength=0.01).solve_detailed(y, A)
h1 = TikhonovSolver(reg_strength=0.01,
    regularization_operator=GradientOperator(A.domain_shape))
x_h1 = h1.solve(y, A)
```

`fista()`、`FISTASolver.solve()`、`solve_detailed()`、旧 TV 裸函数和 wrapper 均委托 `solvers/fista.py` 的同一更新循环。Python 通用核心接受实现 `Regularizer` 与 `ProximalOperator` 的对象；正式 CLI/Agent 能力限制为表中已验证配置。

CLI 继续通过既有 solver 位置参数和 YAML `parameters.regularizer` 独立选择。新增示例位于 `configs/examples/`，不会加入算法文件清单：

```powershell
python -m inv_framework.cli run fista --case parallel_2d/disk_analytic_32 --config configs/examples/fista_haar_l1.yaml --out artifacts/p1_haar_example --device cpu
python -m inv_framework.cli run tikhonov --case parallel_2d/disk_analytic_32 --config configs/examples/tikhonov_h1.yaml --out artifacts/p1_h1_example --device cpu
```

Agent 使用同样的 solver ID、示例 config URI 和 `algorithm.parameters.regularizer`。参数推荐复用 registry defaults、合法范围和兼容性验证。FISTA 无需在请求阶段预先提供谱范数；运行时估计并检查步长。动态回溯及 H1 的 D 工作使用运行时预算，不以旧 TV 单次 forward 成本或恒等 L 成本作保守估计。

`fista` 新入口以 `strength` 为标准字段，支持旧 `reg_strength` 别名；CG 配置以 `reg_strength` 为标准字段，支持新 `strength`。两字段同时存在且不同会拒绝，包括配置与请求合并后发生的冲突；相同值接受。若在旧 Tikhonov YAML 上使用新字段，应移除旧字段或令两者一致。参数来源随别名转换保留。旧 `tv_fista` API 的 `reg_strength` 名称保持不变。

## 数学与参数约定

目标为 `0.5||Ax-y||² + strength R(x)`，lambda 仅在目标装配及对应 `t*lambda` 近端权重处施加一次，不进入正则对象自身。

- L=I：`R(x)=0.5||x||²`，`prox_(t R)(v)=v/(1+t)`；加 box 时对闭式解逐分量裁界。广义 L 不能调用这个闭式近端。
- H1：`R(x)=0.5||Dx||²`，梯度为 `D*Dx`。D 与 P0 TV 使用同一单位网格前向差分，垂直末行、水平末列为零。`GradientOperator` 将两个方向放在 batch 后的 axis 1，range 为 `(2,*domain_shape)`。CG 求解 `A*A + lambda D*D` 正规方程；需要该方程在求解子空间上正定，不能保证任意共同零空间问题唯一可解。
- Haar：仅二维正交 Haar，每个方向 `1/sqrt(2)` 归一化；按偶数原点 `(0,1),(2,3),…` 下采样，高通为 `(even-odd)/sqrt(2)`。周期 dyadic 网格不需要跨端点补值。每层活动块布局为 `[[LL,LH],[HL,HH]]`，字母先表示垂直、再表示水平方向；仅 LL 递归。所有细节权重为 1，最粗 LL 权重为 0。
- `level` 必须为正整数，H/W 为正且均被 `2**level` 整除；不补零、不裁剪。W 保留完整系数，`W*W=WW*=I`，因此整项近端是分析、细节软阈值、合成。batch/channel 独立，保持 dtype/device。

通用 FISTA 默认 `strength=0.01`、最大外层 50、tolerance=`1e-5`、power iterations=12；Haar 默认 level=2。它们是可复现的工程起点，不是已通过质量 pilot 确定的最优正则强度；使用前应说明数据尺度，质量比较须声明统一调参规则和预算。旧 YAML 的强度与默认值不变。默认值不构成调参结果。

省略步长时以确定性有限 power iteration 得到 `L_est`，初始 `t=0.99/L_est`；零估计取 1。该估计不是谱范数上界。每个更新按 batch 检查

```text
f(z) <= f(v) + <grad f(v),z-v> + ||z-v||²/(2t)
f(x) = 0.5||Ax-y||²
```

不满足则将步长减半，最多 60 次 trial；上界比较只容许 dtype 舍入量级误差。用户指定步长也作为初始 trial 检查。检查光滑项上界不依赖 TV gap；它不能把有限 FGP 变为精确近端。TV 的内层迭代与原 iterate-change 停止保持原数值递推，不增加 gap 证书，也不声明精确近端 FISTA 收敛率。TV mapping 是配置的近似近端下的固定点指标。

## 轨迹、停止与成本

新 FISTA 及带停止 policy 的 TV 轨迹均记录接受后返回的 x、完整复合目标和返回点 proximal-gradient mapping。沿用 CT mapping 归一化 `||G_t(x)|| / max(||x||/t,1e-12)`，并要求目标变化及可选 discrepancy 连续满足 policy。达到迭代上限仍为 `max_iterations`。无 policy 的旧 TV 继续记录 momentum 轨迹，单独在最终返回点验证 mapping；历史 iterate tolerance 提前停止标为 `stalled`，不凭该条件宣称最优性。

初始化范数估计、梯度、每次回溯 forward、终点 adjoint/prox 和 runtime endpoint confirmation 都经过现有计数器。返回点 prediction 可复用最后接受 trial 的 forward 结果。无回溯时，新 FISTA 每外层使用 2 forward、2 adjoint、2 proximal；自动估计另有最多 `power_iterations+1` 对调用。请求 runtime 停止 policy 时，既有 policy 初始化还可进行 8 次 power iteration 的独立范数诊断，这部分也计入 parameter-estimation phase。旧无 policy TV 每外层 2 forward、1 adjoint、1 proximal，终点另需 1 adjoint、1 proximal。拒绝的 trial 另计 forward/prox，不能只按外层次数解释成本。

资源包括真实 `prox_calls/prox_iterations/prox_seconds`、回溯 trial/rejection、TV 的 D/D* 次数，以及 Haar analysis/synthesis 次数和变换耗时。H1 的 D/D* 通过 `GradientOperator.stats()` 单列为 `auxiliary_D_*`，不占用 CT A/A* 预算；runtime 的 `ct_forward_calls/ct_adjoint_calls` 与辅助工作分别解释。这些额外字段不代表新的 CT 投影调用。

operator cap 在工作发生前检查。FISTA 预算中断保留最后接受点，并明确 `endpoint_mapping_complete`；裸函数和 tensor `solve()` 继续抛出预算异常，detailed 返回 `resource_exhausted`。runtime 保留 `partial_reconstruction.pt`、已完成迭代、轨迹与中断诊断；该文件不作为成功重建结果。NaN/Inf 和非有限目标/mapping 为 `numerical_error`。Agent 的现有 subprocess wall timeout 保持生效。
