# CT Agent 使用说明

主项目的 CTBackend 与 ct_selection 精确核对 17 项能力：
`fbp, sirt, landweber, cgls, lsqr, sart, os_sart, mlem, osem, tikhonov,
tv_fista, fista, admm, pdhg, constrained_pdhg, asd_pocs_tv, fdk`。
registry、算法 YAML 与构建器必须一致；额外项和缺项均拒绝。

执行链为能力发现、结构化候选过滤、参数初始化及校验、公开 case staging、
CTAdapter、torch-first 子进程、库内 CLI/runtime、原生 solver、终点确认及独立评价。
旧 `pdhg` 是惩罚目标，`reg_strength` 保持原义；新约束算法使用独立 ID。

## 组件发现与请求组装

包含组件接口的主项目版本支持以下命令；均在主项目根目录执行，
不属于 `invct` 子命令，安装 CT 包本身也不会安装主项目 CLI：

```bash
inverse-agent list-ct-components --ct-repo external/ct-benchlab
inverse-agent enumerate-ct-compositions --request request.json --ct-repo external/ct-benchlab
inverse-agent compose-ct-request --request request.json --composition algorithm.json --ct-repo external/ct-benchlab
```

`algorithm.json` 是 AlgorithmRequest JSON，例如
`{"algorithm_id":"fista","components":{"regularizer":"tv"},"parameters":{"strength":0.002}}`。
省略的组件字段继承实现。显式 null solver 用于 FBP 等直接方法，null regularizer
要求算法支持无正则项；null 或省略 operator 使用 case 几何。solver、regularizer
与 operator 无需同时显式提供，但组合必须满足 registry 的兼容性约束。

枚举返回各合法 prior 的候选请求、参数建议与拒绝原因；组装返回已通过参数校验的
完整 RunRequest，不执行重建、调参或重试。保存组装结果后，应对同一请求文件执行
`inverse-agent check-capability`、`validate-run`，再执行 `run-experiment`。
为每次实验设置独立 run_id 和空 output_dir。切换组合后旧执行计划被清除；
需要审计包时，在组装后调用 `prepare_ct_request()`。

Python 入口为 `IterisExperimentService.list_ct_components()`、
`enumerate_ct_compositions(request)`、`compose_ct_request(request, algorithm)`。
当前可执行 operator 为二维平行束 `parallel_beam_radon_2d` 与三维锥束
`astra_fdk_3d`，受实际 case 几何约束。库中尚未接入 runtime 的 operator、独立
去噪和 operator-only 操作不在该入口的执行范围内。投影数、角度、分辨率或噪声
改变时，需要准备对应 case 数据，不能仅改参数或 metadata。

## 约束 TV 请求

约束目标为 `min TV(x), ||M(Ax-y)||2 <= epsilon, x>=0`。支持二维方形
parallel-beam、单 batch/channel、line_integral、非空二值 mask、单位网格
isotropic/anisotropic TV。epsilon 必须有限非负，epsilon_source 必须是显式非空
来源字符串。Agent 不自动估计半径。默认 YAML 半径仅演示数值接口，不是噪声标定。

在完整 RunRequest 中可提供以下约束字段：

```json
{
  "objective_type": "residual_ball_constrained_tv",
  "epsilon": 0.31,
  "epsilon_source": "explicit observation-unit radius",
  "tv_mode": "anisotropic",
  "min_value": 0,
  "max_value": null
}
```

该片段属于 `problem.unknown.constraints`。epsilon/source/TV/边界也可在
`algorithm.parameters` 提供；问题约束覆盖演示配置并记录为 problem_constraint，
与显式算法参数冲突时拒绝。objective_type 按结构化目标过滤候选。
WLS/NLL 约束、替代 prior、自由惩罚强度、负下界、上界及不支持的几何/观测域
在迭代前拒绝。算法参数、数学条件和独立库示例见 [约束 TV 用法](constrained_tv.md)。

## 状态、预算与评价

`converged` 要求约束 PDHG 原生可行性和 primal-dual fixed-point 判据；
ASD 可行停滞为 `stalled`，没有全局最优保证。`max_iterations`、约束
`violated`、`resource_exhausted`、`numerical_error` 与 `invalid_parameters`
分别保留。进程级 success 和独立 endpoint passed 均不表示可行或最优。

`budget.max_wall_seconds` 是可选有限非负 runtime 上限，默认 None，覆盖初始化、
参数估计、ASD 归一化、迭代及终点。在 A/A* 调用前和终点边界检查；单次操作
不能抢占。forward/adjoint 预算与 runtime wall 中断保留最后完整外层点，
未确认终点 objective/residual 为空，不能用历史点或独立评价值补造。
SART 同样保留最后完整 sweep。外层上限由 max_iterations 传递。
当配置迭代数小于预算上限时，终点确认与状态归类使用配置的实际迭代上限；
较宽的预算不会把正常达到配置上限误判为轨迹提前结束。

`budget.timeout_s` 是含子进程启动的硬超时，返回 timeout/resource_exhausted；
强制终止不能保证部分点。若需要可恢复中断，应给硬超时留出 runtime 预算之外的
启动和落盘时间。Windows 使用 staging_only，不提供操作系统文件系统隔离保证。

truth 只用于观测生成和独立 image.rmse/image.ssim 评价。公开 staging 只有
零 truth shape 占位、观测和有效 mask，没有真实 truth、y_clean 或评价 ROI。
参数选择、权重与停止不读取 truth。

## 安装与调用

交付安装应使用远程真实固定 commit 的非 editable 包：

```powershell
python -m pip install --no-deps "inv-framework @ git+https://github.com/apricity093/agent_CT.git@<CT_COMMIT>"
```

`<CT_COMMIT>` 必须替换为已可从远程获取的完整 SHA。Python 包携带 17 个
`configs/algorithms/*.yaml`；数据集仍从显式 catalog/data-root 提供。
开发 checkout 和已安装包可能不同；包版本 0.1.0 本身不能确定源码版本。

主项目 CLI 的 `--ct-repo` 可指向包含 inv_framework 和 configs 的安装根目录
（通常为该环境的 Lib/site-packages），请求中的 config URI 使用该目录的 YAML。
输入 catalog 可以来自独立数据目录。先 list-algorithms 和 validate-run，再
run-experiment。实际子进程的 runtime_import.json 记录解释器、模块路径、
sys.path 与 direct_url；subprocess.json 记录命令和环境。

Windows 沿用基础 Conda 环境的 Library/bin PATH、torch-first 启动以及
OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS 均为 1。隔离安装时
PYTHONPATH 只包含主项目 src，移除 CT checkout，避免覆盖已安装来源。
主项目 `tests/test_ct_p6_agent.py` 依据实际导入的 inv_framework 定位后端，
使用 8²/7 views synthetic case 覆盖新算法和旧 PDHG/SART/FBP；数据模板只作输入。

## 固定版本状态

本版本包含完整的 17 项 registry、算法配置和 Agent 数值接口。主项目在
`pyproject.toml` 与 README 中固定本库提交；安装验证应使用该远程提交构建的包，
移除 PYTHONPATH 中的 CT checkout，并运行上述真实 Agent 测试。
固定来源安装检查与严格数值精度验收分别记录；P7/P8 未启动。

## 现有版本证据与限制

主项目通过忽略的 `external/` 路径使用这个独立仓库，不是 submodule/subtree。
本次扩展基于 `011d48d69578f3608000663db87f4ab357b81527`。
历史基础环境实际安装来源为 `0c787441...`，
见 [base-install.json](../artifacts/ct_composable/p6/acceptance-20261023/base-install.json)。
包版本 `0.1.0`、checkout HEAD、已安装 distribution metadata 三者不能互相替代。

已有本地 wheel 验证的 [audit.json](../artifacts/ct_composable/p6/acceptance-20261023/audit.json)、
[安装日志](../artifacts/ct_composable/p6/acceptance-20261023/install-delivery.log) 与
[真实调用记录](../artifacts/ct_composable/p6/acceptance-20261023/installed-delivery.log)
保留原始来源、17 个配置与六次代表调用证据；当时的临时 wheel/安装环境已不在当前目录，
不能按日志中的路径重新使用它们。这些本地验证不等于远程固定 commit 验收。
针对性测试证据分别见同目录的 final-tests.log（37 项）、endpoint-tests.log（6 项）
和 tests-3.log（31 项），集合重叠，不相加。失败尝试及真实预算中断记录仍保留。

约束 TV 的严格长程精度仍有未通过和未完成项，见
[数值限制](constrained_tv.md)。32² 质量实验的已接受端点成本不含早期
SART 未落盘的完整失败成本，见 [质量示例](quality_example.md)。当前主项目组织
phantom 的 FDK 原生观测范数与残差异常仍待核查；独立图像指标正常不代表
数据一致性通过。正式结果入口由主项目 README 指向，历史 32²/CUDA/安装证据
不充当最新 17 项组织结果或普遍质量排名。
