# CT 离散算子验证与回归依据

当前 ParallelBeamRadon2D 的离散转置、双向 autograd 与公共差分由永久测试
保护。下列数值表保留 2026-10-22 初始 CPU 验证的观测值；后续 dtype、CUDA
和组合 solver 的当前行为见 [模块说明](../inv_framework/README.md) 与
[Agent 用法](agent_usage.md)。历史数值证据不充当最新安装或算法质量验收。

## 修正内容

反向旋转后的插值不是前向双线性插值矩阵的转置。原有自定义 backward 调用同一错误反投影，原测试只能证明两条错误路径一致。

现在先使用与前向相同的正角度网格，再按 `align_corners=False` 的坐标换算获得四邻点权重，使用 `scatter_add_` 将行求和的转置展开值累加至原图，屏蔽落在图像外的邻点。没有拟合缩放，没有改变前向离散化、角度、几何、归一化或公开 API。生产 adjoint 不执行额外 forward 或 VJP；一次 adjoint 仍是一次转置计算。自定义 backward 在两方向分别调用配对的 Function，支持被验证的一阶梯度和二阶 Hessian-vector 路径。

公共 `operators.gradient.gradient` 保留 TV 原有前向差分：垂直末行和水平末列为零。`gradient_adjoint` 仅累加有效差分边，忽略无效边界 dual 分量。TV 的两个内部静态入口指向同一公共函数；TV 定义、强度、value/proximal API、FGP 数值更新和停止逻辑保持现有语义。

Windows 回归实际触发资源记账的 invalid handle 异常信息，定位为 ctypes 未声明指针宽度的 process handle。`GetCurrentProcess` 和 `GetProcessMemoryInfo` 现使用明确的返回/参数类型，新增本机 working-set 回归；没有增加依赖或更改预算口径。

## 验证与结果

环境：Windows，Python 3.12.14，PyTorch 2.5.1，CPU；`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。`torch.cuda.is_available()` 为 false。CPU float32 初始转置阈值 `1e-5`、CUDA 测试阈值 `1e-4`、独立 float64 阈值 `1e-8` 保持原值。

| 尺寸 / 角度 | 显式矩阵转置相对误差 | 归一化内积误差 | L2 方向导数相对误差 |
| --- | ---: | ---: | ---: |
| 8² / 均匀 7 角度 | 1.330e-7 | 1.046e-8 | 3.080e-6 |
| 16² / 均匀 7 角度 | 1.122e-7 | 7.227e-9 | 5.580e-6 |
| 8² / [0,120°) 7 角度 | 1.173e-7 | 5.615e-9 | 7.439e-6 |
| 16² / [0,120°) 7 角度 | 1.171e-7 | 6.427e-9 | 2.122e-5 |
| 8² / 非均匀 7 角度 | 1.325e-7 | 1.966e-8 | 2.272e-5 |
| 16² / 非均匀 7 角度 | 1.211e-7 | 7.156e-9 | 3.235e-5 |

每种角度同时验证完整的 `adjoint(I)` 与基向量构造的显式矩阵转置、独立随机 x/y、原生 forward VJP、自定义 forward/adjoint backward、直接 scatter 计算的 VJP、batch=2/channel=3 和 forward 冻结回归。内积误差以 `||Ax|| ||y||` 归一化，避免随机点积接近零造成不稳定比值。L2 使用 float32 中心差分 `h=0.01`，方向导数阈值 `2e-3`。

独立 double 网格/角度的小矩阵测试误差为 `1.817e-16`，并通过原生 VJP 核对；该测试只在隔离测试中替换网格 dtype，当时公开 CT 路径固定 float32；当前网格 dtype 行为见下文。D 在任意 double dual 下通过内积/autograd 核对，覆盖非零末行/末列、1×N/N×1/1×1、常量、多 batch/channel。TV 两种模式验证二维 value、常量保持、降 TV、零强度和两像素闭式 proximal。

本次另从 Git 基点读取旧实现重新测量：8²/7 角度、种子 17 的旧伴随误差 `0.0629649758`，新伴随误差 `1.125489e-7`，forward 最大差值 `0.0`。同一随机输入的三步 Tikhonov/TV-FISTA 相对旧图像差异分别为约 4.45%/2.88%，对应 corrected transpose 对 CG 正规方程和数据梯度的影响，不要求旧错误结果保持不变。

### 冻结短轨迹

`test/data/p0_short_trajectories.json` 由 `tools/freeze_p0_baseline.py` 显式生成，pytest 只读取固定记录。参考使用冻结的旧原生 forward 作用于基向量构造 A；随后完全使用 NumPy float64 的 A、A.T、独立组装的差分矩阵与求解递推，不调用生产 adjoint、solver 或 regularizer。保存测量、实际角度、完整最终图像、逐步目标/残差和参数。

固定 8² 小图、7 均匀角度、零初始化、3 次迭代/epoch；Tikhonov lambda=0.2，TV lambda=0.08，TV step=0.01，prox=30 次且 tolerance=0，SART block_size=2。无调参。该实例 `||A||_1 ||A||_inf=57.671568` 是平方谱范数上界，TV step=0.01 满足光滑数据梯度稳定条件；这组参数仅作回归依据。

| 入口 | 相对参考图像误差 | 最终归一化残差 | forward / adjoint | 状态 |
| --- | ---: | ---: | ---: | --- |
| Tikhonov | 9.533e-7 | 0.062383813 | 5 / 5 | max_iterations |
| TV-FISTA | 7.192e-8 | 0.337198120 | 4 / 3 | max_iterations |
| CGLS | 1.916e-6 | 0.061651822 | 8 / 4 | max_iterations |
| SIRT | 9.616e-8 | 0.223378667 | 5 / 4 | max_iterations |
| SART | 8.877e-8 | 0.074309215 | 27 / 24 | max_iterations（maximum_epochs_reached） |
| FBP | 2.598e-7 | 0.102269687 | 1 / 1 | completed_valid |

裸 `solve()` 与 detailed 各自对照冻结参考；计数还与实际 `_radon_*_impl` spy 对照，包含子算子调用。TV 另核对 3 次 prox / 90 次内层迭代。FBP 本身为 0 次优化迭代，detailed 不提供轨迹/残差；表内包含单独执行并计账的一次终点 forward。TV legacy detailed 保留 momentum 状态轨迹，终点单独评估，未将它改成返回点轨迹。迭代上限没有被记为收敛。原有 dense-view disk FBP 回归为 29.04 dB，超过原阈值 20 dB。

### 已保存的验证证据

初始 CT 与 Agent 专项日志分别为
[CT 测试日志](../artifacts/ct_composable/p0/p0_ct_tests.log) 和
[Agent 测试日志](../artifacts/ct_composable/p0/p0_agent_tests.log)。这些是初始版本的
观测记录，不是当前全部套件的通过数量。当前回归仍由 `tests/test_p0_operators.py`、
`tests/test_p0_short_trajectories.py` 及 `tests/test_radon_adjoint.py` 保护。

## 当前行为与限制

Radon 网格按输入 dtype 构造，CPU/CUDA float32/float64 均有专项测试；
上面的初始 CPU 数值记录属于当时 float32 基线，不限制当前公开接口。
TV FGP 使用有限内层迭代，没有严格近端 gap 证书。当前 FISTA 步长与回溯约定见
[FISTA 用法](composable_reconstruction.md)，预算和固定来源状态见
[Agent 用法](agent_usage.md)。历史重建对应其记录的实现，不能与当前实现混作同版比较。

冻结参考由 `tools/freeze_p0_baseline.py` 和 `tests/p0_reference.py` 产生，
`test/data/p0_short_trajectories.json` 是永久回归输入，测试不会自动重写它。
原始修正前后观测见 [p0_before_after.json](../artifacts/ct_composable/p0/p0_before_after.json)。

## 2026-10-23 CUDA 专项验收

本轮 P0/P1/P2 CUDA 验收使用 GTX 1650 4GB、Python 3.12.14、PyTorch 2.5.1 / CUDA 12.1。实际导入当前 CT checkout，通过显式 PYTHONPATH 隔离已安装版本。原验收阈值、P0 冻结文件和历史 CPU 记录保留。

数值、正式 Agent/Adapter/CLI、预算/失败和资源验收的命令、结果、限制及逐配置成本见 [CUDA 验收报告](../artifacts/ct_composable/p2_cuda/report.md)。这是当时的专项记录；当前接口和固定版本状态以 [Agent 用法](agent_usage.md) 为准。
