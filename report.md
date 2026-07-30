# invframework change report

## 2026-06-08

- Files changed: `AGENTS.md`, `report.md`
- Summary: Added follow-up execution rules requiring all newly added tests to be stored under `D:\Pythonprojects\yan_1\invframework\test`, while preserving existing frozen `tests/` contents. Added the requirement that every project file change be recorded in this root `report.md`.
- Frozen baseline impact: No frozen `.py` implementation was modified.
- Tests: Not run; documentation-only change.

## 2026-06-08

- Files changed: `plan.md`, `report.md`
- Summary: Rewrote `plan.md` as an executable CT solver integration experiment runbook. Clarified that `D:\Pythonprojects\yan_1\ctalgorithm\inv_framework\...` paths are initial rewritten source files, while `D:\Pythonprojects\yan_1\invframework\...` paths are current project landing files. Updated the test plan to require new tests under `test/`, added phase-by-phase execution gates, and made LSQR/FDK/subset/MLEM-OSEM compatibility rules explicit.
- Frozen baseline impact: No frozen `.py` implementation was modified.
- Tests: Not run; documentation-only change.

## 2026-06-08

- 完成阶段：Phase 0 preflight；Phase 1 新增内部工具。
- 改动文件：`inv_framework/solvers/_utils.py`，`report.md`。
- 改动摘要：读取并确认当前项目规则、执行方案和初步改写来源文件；新增 `_utils.py`，提供新增 CT solver 所需的 `LinearOperator` 校验、measurement shape 校验、`x_init` 初始化/迁移、box clamp、角度 subset 构造、measurement subset 选择和 subset operator 适配。`ParallelBeamRadon2D` 的 subset fallback 被隔离在 helper 中，未修改 `LinearOperator` 基类。
- 是否触及冻结清单：未改写冻结清单中的既有 `.py` 实现；本阶段仅新增独立 `.py` 文件。
- 测试：尚未运行；后续 Phase 7 统一运行。

## 2026-06-08

- 完成阶段：Phase 2 classical 增量追加。
- 改动文件：`inv_framework/solvers/classical.py`，`report.md`。
- 改动摘要：在 `classical.py` 文件末尾追加 CGLS、LSQR、FDK 函数及 `CGLSSolver`、`LSQRSolver`、`FDKSolver` wrapper。CGLS 使用显式 `A^T` 的共轭梯度最小二乘更新；LSQR 使用 Golub-Kahan bidiagonalization 迭代；FDK 仅调用 `operator.fdk(y, **kwargs)` backend，无 backend 时抛 `NotImplementedError`。
- 是否触及冻结清单：触及冻结清单文件 `inv_framework/solvers/classical.py`，但仅在 2026-06-08 既有实现之后追加新内容，未改写既有函数/类/导入/默认参数/控制流。
- 测试：尚未运行；后续 Phase 7 统一运行。

## 2026-06-08

- 完成阶段：Phase 3 subset solver。
- 改动文件：`inv_framework/solvers/subset.py`，`report.md`。
- 改动摘要：新增 SART、OS-SART 函数及 `SARTSolver`、`OSSARTSolver` wrapper。实现中要求 `LinearOperator`，并通过 `_utils.make_subset_operator()` 获取子集 operator，每个 subset update 都使用子集 operator 的 `forward/adjoint`，未采用“只切 measurement 但继续用全角度 operator”的方式。
- 是否触及冻结清单：未改写冻结清单中的既有 `.py` 实现；本阶段仅新增独立 `.py` 文件。
- 测试：尚未运行；后续 Phase 7 统一运行。

## 2026-06-08

- 完成阶段：Phase 4 statistical solver。
- 改动文件：`inv_framework/solvers/statistical.py`，`report.md`。
- 改动摘要：新增 MLEM、OSEM 函数及 `MLEMSolver`、`OSEMSolver` wrapper。MLEM/OSEM 均要求 `LinearOperator`，使用非负线性投影数据下的 multiplicative update；OSEM 逐 subset 计算 prediction、ratio 和 sensitivity，并通过 `_utils.make_subset_operator()` 使用子集 operator。
- 是否触及冻结清单：未改写冻结清单中的既有 `.py` 实现；本阶段仅新增独立 `.py` 文件。
- 测试：尚未运行；后续 Phase 7 统一运行。

## 2026-06-08

- 完成阶段：Phase 5 solver 导出。
- 改动文件：`inv_framework/solvers/__init__.py`，`report.md`。
- 改动摘要：在 `solvers/__init__.py` 文件末尾追加 CGLS、LSQR、FDK、SART、OS-SART、MLEM、OSEM 的函数和 Solver 类导入，并通过 `__all__ += [...]` 追加公开导出。
- 是否触及冻结清单：触及冻结清单文件 `inv_framework/solvers/__init__.py`，但仅在末尾追加导入和导出项，未删除、改名或改变已有导出含义。
- 测试：尚未运行；后续 Phase 7 统一运行。

## 2026-06-08

- 完成阶段：Phase 6 新增测试。
- 改动文件：`test/test_classical_extra_solvers.py`，`test/test_subset_solvers.py`，`test/test_statistical_solvers.py`，`test/test_solver_contract.py`，`report.md`。
- 改动摘要：在 `test/` 目录新增 CGLS/LSQR/FDK、SART/OS-SART、MLEM/OSEM 和新增 solver contract 测试。测试覆盖 residual 下降、LSQR dense lstsq sanity check、FDK backend gate、subset operator 调用、`ParallelBeamRadon2D` subset fallback、非负输出、输出 shape、`x_init` shape/dtype 处理和普通 `ForwardOperator` 拒绝行为。
- 是否触及冻结清单：未改写冻结清单中的既有测试；新增测试均保存到 `test/`。
- 测试：尚未运行；即将进入 Phase 7。

## 2026-06-08

- 完成阶段：Phase 7 全量测试；Phase 8 最终记录。
- 改动文件：eport.md
- 改动摘要：确认全量测试通过（18 passed，包括 freeze 清单中的既有测试和新增 solver 测试）；补全最终交付记录。
- 测试命令：D:\anaconda3\envs\fno\python.exe -m pytest -q tests test
- 测试结果：18 passed，耗时 33.15s
- 是否触及冻结清单：未改写冻结清单中的既有 .py 实现。
- 实验状态：全部 Phase 完成，无 stop gate 触发。

## 2026-07-29

- 完成内容：新增 Tikhonov 与 TV 正则化重建能力。
- 改动文件：`inv_framework/regularizers/__init__.py`、`inv_framework/regularizers/base.py`、`inv_framework/regularizers/tikhonov.py`、`inv_framework/regularizers/tv.py`、`inv_framework/solvers/regularized.py`、`inv_framework/solvers/__init__.py`、`inv_framework/README.md`、`test/test_regularized_solvers.py`、`report.md`。
- 改动摘要：新增 `TikhonovRegularizer`，表达 `0.5 ||Lx||_2^2`，默认 `L=I`，并允许通过额外 `LinearOperator` 表达广义 Tikhonov；新增 `tikhonov()` 和 `TikhonovSolver`，使用 batched conjugate gradients 求解正规方程。新增 `TVRegularizer`，支持二维 isotropic/anisotropic TV，并使用双变量 Fast Gradient Projection 求解 ROF proximal；新增 `tv_fista()` 和 `TVFISTASolver`，外层使用 FISTA，未指定步长时通过 power iteration 估计 `||A||^2`。
- 来源与能力边界：算法语义参考 CIL 的 FISTA/TotalVariation、TIGRE 的 TV-FISTA 控制流程和 TomoPy 的 `tikh`/`tv` 入口；本实现为框架原生纯 PyTorch 版本，不宣称与 TIGRE CUDA `im3ddenoise/minTV` 或 TomoPy C backend 在几何、插值和尺度上逐点数值等价。
- 是否触及冻结清单：触及冻结清单文件 `inv_framework/solvers/__init__.py`，但只在文件末尾追加 Tikhonov/TV-FISTA 的导入和 `__all__` 项；未改写、删除或移动任何 2026-06-08 既有实现。其余新增 Python 实现和测试均位于新文件中。
- 测试：`D:\anaconda3\envs\fno\python.exe -m pytest -q test\test_regularized_solvers.py`，结果 `9 passed`；`D:\anaconda3\envs\fno\python.exe -m pytest -q`，结果 `27 passed`。

## 2026-07-30

- 完成内容：初始化本地 Git 仓库并同步至 GitHub 仓库 `apricity093/agent_CT`。
- 改动文件：`report.md`；另新增 Git 元数据目录 `.git/`。
- 改动摘要：提交前检查了忽略规则、敏感信息与大文件；使用 `main` 作为本地默认分支，配置目标 GitHub 远程仓库，并将本地 `main` 推送为 `origin/main`。
- 是否触及冻结清单：未改写、删除或移动任何冻结清单中的既有 `.py` 实现。
- 测试：未运行；本次仅执行版本控制初始化与远程同步，不涉及 Python 代码变更。
