# ADMM、PDHG 与梯度 Huber 使用说明

当前支持范围以[组件组合支持表](supported_combinations.md)为准；下文保留对应实现阶段的接口与数学说明。

## 接口与支持矩阵

`admm`、`pdhg` 的裸函数、`solve()` 和 `solve_detailed()` 共用 splitting.py 核心，runtime 构建器复用 registry。算法 YAML 位于 configs/algorithms。二次保真项和透射 NLL 的范围见 [保真项说明](data_fidelity.md)。

| 正则 | FISTA | ADMM / PDHG |
|---|---|---|
| 恒等 Tikhonov | 无约束、box、非负 | 无约束、非负 |
| TV（isotropic / anisotropic） | 无约束 | 无约束、非负 |
| H1 | 提前拒绝 | 无约束、非负 |
| 分量梯度 Huber | 提前拒绝 | 无约束、非负 |
| Haar-L1 | 无约束 | 无约束、非负 |

通用 FISTA 的无约束组合为 3 个，两种新 solver 各 5 个，共 13 个组合。新 solver 的 `min_value=None` 为无约束，`min_value=0` 为非负；任何其他下界或上界均提前拒绝。旧 TV-FISTA 的 legacy projection 行为保留。Haar 使用完整正交 W；LL 权重零，细节权重一，dual LL 通过 Moreau 固定为零。只支持已实现的 I、D、Haar，不接受任意广义 Tikhonov 变换。

## 数学与参数

目标为 `0.5||Ax-y||² + strength*h(Tx) + I_C(x)`。strength/reg_strength 遵循 [FISTA 用法](composable_reconstruction.md) 中的冲突拒绝和别名规则，强度只施加一次。

ADMM 无约束分裂 z=Tx；x 正规方程为 `(A*A+rho T*T)x=A*y+rho T*(z-u)`。非负时增加 q=x，把 `(z,q)` 作为一个联合块，其可分 proximal 分别为系数 proximal 和非负投影；x 步增加 `rho I` 和右端 `rho(q-v)`。这仍是两个分组块的 ADMM，未 clamp 无约束 x 解。返回可行 q，并在返回 q 上重新计算数据项、目标和终点 KKT。

rho 有限且正，默认 1 对应 unit-grid/正交变换系数的一单位增广尺度，是可复现初始化，不表示质量或速度最优；默认值不表示经过 rho 搜索。无约束正规方程正定的条件是 `ker(A) intersect ker(T)={0}`。I/W 可逆，所以 rho>0 即满足。连通二维网格的 D 核仅为常量，当前非零非负 CT 对常量有非零投影，因此满足交核条件。非负分裂的 rho I 直接保证正定。不添加 ridge。

CG 在当前 x 上 warm start，累计所有正规方程 A/A*；沿用 Torch 的矩阵自由 CG 递推方式。停止要求 `||r|| <= cg_tolerance/k² * max(||rhs||,1)`，默认 cg_tolerance=1e-6、上限 100。该目标随外层收紧，在有界 rhs 和固定正定系统的前提下为可求和误差控制；浮点递推残差并非精确解证书。非正曲率/非有限数值报告 numerical_error，内层上限报告 max_iterations/cg_inner_max_iterations，不接受未完成的外层更新。记录实际内层次数、时间及最终内层相对残差。

ADMM primal 为 `sqrt(||Tx-z||²+||x-q||²)`（无约束省略第二项）；dual 为 `rho||T*(z-z_old)+(q-q_old)||`。绝对阈值为 tolerance*sqrt(对应空间维数)，相对项分别为 tolerance*max(两侧堆叠变量范数) 和 tolerance*增广 dual adjoint 范数。只有两个条件同时连续通过才收敛。元数据 split_residual 是两项相对阈值的最大比值（通过界为 1）。

PDHG 使用 K=[A;T]，theta=1。L2 dual 更新 `(p+sigma*(A*xbar-y))/(1+sigma)`；正则 dual 使用 `v-sigma*prox_(strength*h/sigma)(v/sigma)`；原始步通过非负集合 projection。必须 tau,sigma>0 且 tau*sigma*B<1。默认二者为 `.99/sqrt(B)`。CT 非负矩阵以 `||A||1*||A||inf` 为平方范数上界，计算包括一次 A1 和一次 A*1；D 加 8，I/W 加 1。一般带符号小矩阵逐基向量计算绝对行列和，不能套用 A1；该通用参考路径限 4096 个域元素。没有将 power estimate 当成上界。

PDHG 的返回点指标为三个 primal-dual fixed-point residual 的联合范数，除以 max(1,||x||,||p||,||d||)，包含数据 dual、正则 dual 和 primal projection 条件。无 gap 声明。runtime 对返回点再次独立执行 fixed-point/KKT 与 objective 计算，不将一次迭代内记录当作独立终点验证。ADMM 使用其正则 dual rho*u 和返回点精确 L2 数据 dual 进行终点检查。

Huber 定义逐 D 分量：小于等于 delta 时 t²/(2delta)，否则 |t|-delta/2。gradient 使用公共 D/D*。系数 proximal 在 |v|<=delta+s 时为 delta*v/(delta+s)，否则 v-s sign(v)。它不是 h(Dx) 的图像域整项 proximal。delta 有限且正；默认 .01 来源为 unit-grid、归一化图像差分尺度的初始配置，未做 pilot、不宣称最优；修改图像或网格尺度时需要重新指定。代码提供 value、gradient 和 coefficient proximal，Moreau 在分裂正则块复用。

## 资源与中断

A/A* 沿用 CountingLinearOperator 字段，包含初始化、ADMM 每次 CG、PDHG 数据块及独立终点。辅助变换在 transform_forward_calls / transform_adjoint_calls / transform_seconds 记录，以 transform_kind 区分 I/D/W，CT 字段不混入 D/W。coefficient_prox_calls/seconds 另记，同时通过既有 record_prox 累计标准 proximal 计数；闭式 proximal 的内层迭代为 0。cg_iterations/cg_seconds 单列，不能当作外层数。runtime 独立终点的辅助工作增量加入 resources，并单列 endpoint_confirmation。

operator-call 中断返回最后完整外层状态；CG 和 dual 半步不提交。终点中断时 endpoint_diagnostics_complete=false，返回结果不携带旧点的 objective/residual。NaN/Inf 为 numerical_error。wall timeout 复用正式 Agent/subprocess 的 timeout 语义；不承诺被终止进程一定写出部分状态。
