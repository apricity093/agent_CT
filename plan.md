# CT solver 增量接入实验方案

## 0. 实验目的

将用户已经初步改写的 CT 求解算法，按当前项目 `inv_framework` 的既有接口和冻结规则，增量接入到当前仓库：

```text
D:\Pythonprojects\yan_1\invframework
```

本方案是可执行 runbook。执行时应按 phase 顺序推进，除非触发本文列出的 stop gate，否则不要中途停下等待确认。

## 1. 强制约束

执行前必须先阅读并遵守：

```text
D:\Pythonprojects\yan_1\invframework\AGENTS.md
```

关键约束：

- 2026-06-08 已存在的 `.py` 实现被冻结，不得改写、删除、重命名、移动或改变既有行为。
- 允许新增独立文件。
- 允许在既有 `.py` 文件末尾追加新函数、新类、新导出，但不得修改既有符号。
- 后续新增测试必须保存到：

```text
D:\Pythonprojects\yan_1\invframework\test
```

- 每次文件改动后必须更新：

```text
D:\Pythonprojects\yan_1\invframework\report.md
```

## 2. 来源路径与落地路径

下表中的“初步改写来源文件”是用户已经初步改写后的算法保存路径；“当前项目落地文件”是本次实验应写入的目标位置。

| 算法 | 函数 | Solver 类 | 初步改写来源文件 | 当前项目落地文件 | 接入方式 |
|---|---|---|---|---|---|
| CGLS | `cgls` | `CGLSSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\classical.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\classical.py` | 只追加 |
| LSQR | `lsqr` | `LSQRSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\classical.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\classical.py` | 只追加，且必须是真 LSQR |
| FDK | `fdk` | `FDKSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\classical.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\classical.py` | 只追加，backend-gated |
| SART | `sart` | `SARTSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\subset.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\subset.py` | 新增文件 |
| OS-SART | `ossart` | `OSSARTSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\subset.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\subset.py` | 新增文件 |
| MLEM | `mlem` | `MLEMSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\statistical.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\statistical.py` | 新增文件 |
| OSEM | `osem` | `OSEMSolver` | `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\statistical.py` | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\statistical.py` | 新增文件 |

内部辅助模块：

| 用途 | 当前项目落地文件 | 接入方式 |
|---|---|---|
| 新增 solver 共享 shape、初始化、subset 工具 | `D:\Pythonprojects\yan_1\invframework\inv_framework\solvers\_utils.py` | 新增文件 |

新增测试文件统一放在：

```text
D:\Pythonprojects\yan_1\invframework\test
```

不得把新增测试写入 `tests/`。

## 3. Stop gate

执行时遇到以下任一情况必须停止并向用户说明，不得自行绕过：

1. 需要改写 AGENTS.md 冻结清单中已有 `.py` 符号的既有实现，才能继续。
2. 初步改写来源文件无法读取，且无法仅凭当前项目接口安全实现同等算法。
3. LSQR 来源代码不是 Golub-Kahan bidiagonalization 风格的真实 LSQR，且当前执行者不能补出真实 LSQR。
4. FDK 来源代码试图用普通 backprojection、FBP 或 ASTRA `adjoint` 伪装成 FDK。
5. SART/OSEM 只能通过“切 measurement 但仍使用全角度 operator”的方式实现。
6. MLEM/OSEM 被要求声明为当前 log-domain CT 噪声模型的完整统计重建，但没有相应物理推导和测试。

如果只是初步改写代码接口不一致，应继续适配；这不属于 stop gate。

## 4. Phase 0: Preflight

目标：确认当前项目和来源文件状态，不做代码改动。

步骤：

1. 阅读 `AGENTS.md`、`plan.md`、`report.md`。
2. 确认当前项目根目录包含：

```text
README.md
tutorial.ipynb
examples/
tests/
test/
inv_framework/
AGENTS.md
plan.md
report.md
```

3. 读取当前项目接口文件：

```text
inv_framework/operators/base.py
inv_framework/operators/ct/radon_torch.py
inv_framework/solvers/base.py
inv_framework/solvers/classical.py
inv_framework/solvers/__init__.py
```

4. 尝试读取初步改写来源文件：

```text
D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\classical.py
D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\subset.py
D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\solvers\statistical.py
```

5. 如果来源路径因权限或路径问题无法读取，记录到 `report.md`，然后执行以下判断：

- 若算法可由当前项目接口和标准算法定义安全实现，则继续，但在最终报告中说明“未使用来源代码”。
- 若无法安全实现，则触发 stop gate。

## 5. Phase 1: 新增内部工具

新增：

```text
inv_framework/solvers/_utils.py
```

必须提供：

```python
require_linear_operator(operator, solver_name)
validate_measurement_shape(measurement, operator, solver_name)
prepare_initial_image(measurement, operator, x_init=None, initial_value=0.0)
apply_box_constraints(x, min_value=None, max_value=None)
make_angle_subsets(num_angles, block_size=None, subset_indices=None,
                   order_strategy="ordered", seed=None, device=None)
select_measurement_subset(measurement, indices)
make_subset_operator(operator, indices)
```

验收要求：

- `require_linear_operator` 对普通 `ForwardOperator` 抛 `TypeError`，错误信息包含 `LinearOperator`。
- `validate_measurement_shape` 检查 `measurement.shape[1:] == operator.range_shape`。
- `prepare_initial_image` 返回 `(B, *operator.domain_shape)`，并处理 `x_init` 的 shape、device、dtype。
- `make_angle_subsets` 支持 ordered/random，`seed` 不污染全局随机状态。
- `select_measurement_subset` 沿角度维 `dim=-2` 选择。
- `make_subset_operator` 策略：
  - 优先调用 `operator.subset(indices)`；
  - 若 operator 是 `ParallelBeamRadon2D`，用 `operator.angles[indices]` 构造新的 `ParallelBeamRadon2D`；
  - 其他情况抛 `NotImplementedError`。

不得修改 `LinearOperator` 基类。

## 6. Phase 2: classical 增量追加

只在文件末尾追加：

```text
inv_framework/solvers/classical.py
```

追加内容：

```python
cgls
lsqr
fdk
CGLSSolver
LSQRSolver
FDKSolver
```

不得改动现有：

```python
_require_linear
_ram_lak
_filter_sino
fbp
sirt
landweber
FBPSolver
SIRTSolver
LandweberSolver
```

### CGLS 执行要求

- 解 `min_x ||A x - y||_2^2`。
- 使用 `r = y - A x`、`s = A^T r`、共轭方向 `p`。
- 支持 `num_iterations`、`tol`、`x_init`、`min_value`、`max_value`。
- 默认 `x_init` 为零图像。

### LSQR 执行要求

- 必须实现真实 LSQR，即 Golub-Kahan bidiagonalization 迭代。
- 支持 `num_iterations`、`damping`、`atol`、`btol`、`x_init`、`min_value`、`max_value`。
- 若无法实现真实 LSQR，不追加 `lsqr` / `LSQRSolver`，并触发 stop gate 或在用户确认后改计划。

### FDK 执行要求

只做 backend gate：

```python
def fdk(operator, y, **kwargs):
    backend = getattr(operator, "fdk", None)
    if not callable(backend):
        raise NotImplementedError(...)
    return backend(y, **kwargs)
```

不得修改 `ASTRAOperator3D`，不得把 FBP 或 backprojection 当作 FDK。

## 7. Phase 3: subset solver

新增：

```text
inv_framework/solvers/subset.py
```

公开：

```python
sart
ossart
SARTSolver
OSSARTSolver
```

要求：

- 必须要求 `LinearOperator`。
- 每个 subset update 都必须使用子集 operator 的 `forward/adjoint`。
- 不允许只切 `measurement` 而继续使用全角度 operator。
- row normalization 使用 `sub_operator.forward(ones_domain)`。
- column normalization 使用 `sub_operator.adjoint(ones_subrange)`。
- normalization 小于 `eps` 时安全处理。
- 支持 `num_iterations`、`block_size`、`subset_indices`、`order_strategy`、`seed`、`relaxation`、`x_init`、`min_value`、`max_value`。

## 8. Phase 4: statistical solver

新增：

```text
inv_framework/solvers/statistical.py
```

公开：

```python
mlem
osem
MLEMSolver
OSEMSolver
```

要求：

- 必须要求 `LinearOperator`。
- `mlem` 保留 sensitivity、prediction ratio、multiplicative correction。
- `osem` 每个 subset 独立计算 prediction、ratio、sensitivity。
- 默认 `initial_value=1e-6`。
- 默认 `min_value=0.0`。
- 明确仅适用于非负线性投影数据；不宣称等价于当前 `PoissonLogDomainNoise` 的 log-domain CT 重建。

## 9. Phase 5: 导出

只追加：

```text
inv_framework/solvers/__init__.py
```

新增导出：

```python
cgls
lsqr
fdk
CGLSSolver
LSQRSolver
FDKSolver
sart
ossart
SARTSolver
OSSARTSolver
mlem
osem
MLEMSolver
OSEMSolver
```

不得重写已有 `__all__`，推荐追加：

```python
__all__ += [
    ...
]
```

不得修改：

```text
inv_framework/__init__.py
inv_framework/operators/base.py
inv_framework/operators/ct/radon_torch.py
inv_framework/operators/ct/__init__.py
inv_framework/operators/ct/astra_adapter.py
```

## 10. Phase 6: 新增测试

新增测试必须写入：

```text
test/
```

建议新增：

```text
test/test_classical_extra_solvers.py
test/test_subset_solvers.py
test/test_statistical_solvers.py
test/test_solver_contract.py
```

测试要求：

- `CGLSSolver` 在 dummy dense `LinearOperator` 上 residual 下降。
- `LSQRSolver` 在 dummy dense `LinearOperator` 上 residual 与 `torch.linalg.lstsq` 做 sanity check。
- `FDKSolver` 在无 `fdk()` backend 时抛 `NotImplementedError`。
- `FDKSolver` 在 dummy backend 上确实调用 `operator.fdk`。
- `SARTSolver` / `OSSARTSolver` 确实使用 subset operator。
- `ParallelBeamRadon2D` fallback subset operator 使用 `operator.angles[indices]`。
- `MLEMSolver` / `OSEMSolver` 输出非负且 shape 正确。
- 所有新增 solver 输出 `(B, *operator.domain_shape)`。
- `x_init` shape 错误时抛 `ValueError`。
- `x_init` 会迁移到 measurement 的 dtype/device。
- 所有需要 adjoint 的新增 solver 收到普通 `ForwardOperator` 时抛 `TypeError`。

不得新增测试到 `tests/`。

## 11. Phase 7: 测试命令

优先运行：

```powershell
D:\anaconda3\envs\fno\python.exe -m pytest -q tests test
```

如果该环境不可用，再尝试：

```powershell
python -m pytest -q tests test
```

必要时分组运行：

```powershell
D:\anaconda3\envs\fno\python.exe -m pytest -q tests
D:\anaconda3\envs\fno\python.exe -m pytest -q test
```

如果测试失败：

- 先判断是否是新增代码导致。
- 不得通过改写冻结既有实现来让测试通过。
- 如果失败暴露的是新增 solver 设计问题，应修新增文件或追加内容。
- 如果失败暴露的是环境缺依赖，应记录到 `report.md`。

## 12. Phase 8: report.md 记录

每个 phase 完成后都应更新：

```text
report.md
```

至少记录：

- 日期。
- 完成的 phase。
- 改动文件。
- 是否触及冻结清单。
- 是否仅 add-only。
- 测试命令。
- 测试结果。
- 未完成项和原因。

## 13. 最终交付口径

完成实验后，最终答复应包含：

1. 新增文件清单。
2. 修改过的既有文件清单，并说明是否 add-only。
3. 新增算法列表。
4. 每个算法是否需要 `LinearOperator`。
5. 每个算法是否需要额外 operator 能力。
6. LSQR 是否是真 LSQR。
7. FDK 的 backend-gated 行为。
8. MLEM/OSEM 的数据语义限制。
9. 测试命令和结果。
10. 未纳入算法及原因。

## 14. 本轮不纳入

```text
FISTA
ISTA
ASD-POCS
AwASD-POCS
OS-SART-TV
```

这些算法等待 TV、投影约束、TIGRE/backend 能力边界明确后再单独设计。
