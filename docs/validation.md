# 数值与接口验证

测试从本仓库根目录运行，开发依赖使用 `python -m pip install ".[dev]"`。
ASTRA CUDA、LEAP、非 Cartesian MRI 与 VarNet 分别需要对应后端、扩展包或权重；
完整安装要求见[依赖与安装](../DEPENDENCIES.md)。

## 测试入口

| 范围 | 测试位置 | 验证内容 |
| --- | --- | --- |
| CT 离散算子 | `tests/test_radon_adjoint.py`、`tests/test_operator_references.py` | 投影伴随、显式矩阵与导数参考 |
| CT 变分组件 | `tests/test_combination_extension.py`、`tests/test_fista_fidelity_matrix.py`、`tests/test_splitting_fidelity_matrix.py` | 求解器、保真项、正则项组合 |
| 约束 TV | `tests/test_constrained_ct_reference.py`、`tests/test_constrained_public_api.py` | 残差球、可行性与优化精度 |
| CT API 与数据 | `tests/test_public_task_contract.py`、`tests/test_ellipses_runtime.py` | 任务契约、椭圆案例、停止与资源释放 |
| 3D FDK | `tests/test_fdk_backend_adapters.py`、`tests/ct_fdk_simulation/` | 三维几何、后端与重建 |
| MRI | `tests/mri/` | Fourier/NUFFT、复数伴随、CG、Haar、TV、GRAPPA、ESPIRiT 与 VarNet |
| Deconvolution | `tests/deconvolution/` | 卷积伴随、Wiener、RL、近端、Hyper-Laplacian 与盲核约束 |
| 文件任务与公共数据 | `tests/core/` | 输入输出、API/CLI 一致性、无真值执行、独立评价 |

例如运行 CT 伴随与任务契约：

```bash
python -m pytest tests/test_radon_adjoint.py tests/test_public_task_contract.py
```

`tools/compare_varnet_reference.py` 使用明确提供的上游源码及同一 checkpoint
比较网络输出。小矩阵、解析输入和 `tests/core/fixtures/` 的临时数据用于
数学与文件契约验证。基准生成与运行命令见[公共基准](public_benchmarks.md)。

## 结果解释与限制

执行成功、优化收敛和图像质量分别评价。`max_iterations` 不代表收敛；
ASD 的可行停滞没有全局最优保证。有限内层 TV/Haar 近端达到迭代上限或
返回精度限制时，应读取原生停止原因。高精度约束 CT 参考测试对收敛与
TV gap 有独立要求，不能用短任务成功替代；这些严格回归存在未解决的
精度失败，不能声称全套测试通过。LEAP 的真实后端验证尚未完成。

CT 保真项的严格数学参考使用 float64，以满足 ADMM 逐次收紧的真实内层残差
目标；float32 请求仍按其实际停止状态返回，有限输出不代表通过这些精度要求。

ASTRA FDK 使用 ASTRA 自有数据缓冲执行 FDK_CUDA，保持调用者观测不变，
输出恢复调用者设备和精度。ASTRA 对象由所属执行路径在 finally 中释放。

二维椭圆与公共图像基准使用模型匹配的合成观测；MR 模体、理想 Airy 核及
VarNet 模体推理不代表临床或真实显微成像质量。PSNR/SSIM 不能替代算子和
优化机制检查。评价参考只用于重建后的独立评价，默认参数不代表最优调参。


## 2026-11-10 CT 验收

接口与独立安装验证复用已通过记录，未重复执行。四个二维椭圆及三维案例均保留原数据和 catalog 注册。本次环境为 `D:/anaconda3/envs/inverse-agent-round1/python.exe`：NumPy 2.5.2、PyTorch 2.5.1、ASTRA 2.2.0，PyTorch CUDA/ASTRA CUDA 均可用，GPU 为 GTX 1650 / WDDM。

所有下列结果相对 [outputs/ct_acceptance_20261110](../outputs/ct_acceptance_20261110/)。请求、有效参数、精度、轨迹、指标与 NPY 重建均保留。

### 验收对应关系

| 要求 | 现有入口及本次补齐结果 | 结论 |
| --- | --- | --- |
| 四场景二维 | tools/accept_ct_numerics.py 经 run_ct_task；numerics/、variational/ | 通过 |
| counts/I0路由 | test_ellipses_source_and_photon_observations；实际Poisson参数和保真构造 | 通过 |
| 角度子集 | tests/ct_workflow/test_subset_scheduling.py；真实OS-SART配置 | 通过 |
| 几何/TV/Haar伴随 | test_operator_references.py、test_matched_cone.py、check_ct_mathematics.py | 通过 |
| 自动步长和范数 | test_shared_mathematics.py 的范数/步长测试、解析解对照 | 通过 |
| 完整3D FDK保护 | math_verified/mathematics.json、fdk_verified.log、fdk_repeat_verified.log | 通过 |
| 正式50次GPU生命周期 | check_gpu_lifecycle.py、gpu_lifecycle_verified.json | 本案例通过 |
| 严格CT/TV gap历史声明 | test_constrained_ct_reference.py、references.log | 2项仍失败 |

### 二维实际结果

后端 torch，设备 cuda:0，输出 `(1,1,64,64)` / float32，形状正确且有限。PSNR按案例 data_range=0.02计算，单位dB；SSIM无量纲；RMSE为对象线性衰减尺度。Poisson近端内部float64，返回float32。所有任务 converged=false，success不代表收敛。

案例前缀均为 `parallel_2d__ellipses_`。FBP默认scale=null；SART/OS-SART采用ordered、relaxation=0.1、预算5 epochs，分别block_size=1 / subset_count=10。两项PDHG均预算200、tolerance=1e-5、isotropic TV strength=0.01，Poisson任务显式min_value=0。

| 案例后缀 | 算法 | 次数 / 停止原因 | PSNR | SSIM | RMSE | 结果JSON |
| --- | --- | --- | --- | --- | --- | --- |
| dense_clean | FBP | 0 / direct_reconstruction_valid | 30.054817 | 0.74086237 | 0.0006284766 | numerics/dense_fbp.json |
| sparse_gaussian | SART | 5 / maximum_epochs_reached | 27.500353 | 0.67246878 | 0.0008433587 | numerics/sparse_sart.json |
| sparse_gaussian | OS-SART | 5 / maximum_epochs_reached | 22.380140 | 0.36467841 | 0.0015206280 | numerics/sparse_os_sart.json |
| sparse_poisson | PDHG + transmission_poisson_nll + TV | 200 / maximum_iterations_reached | 32.004936 | 0.88828230 | 0.0005020917 | variational/poisson_pdhg.json |
| limited_angle | PDHG + L2 + TV | 200 / maximum_iterations_reached | 26.459053 | 0.69575876 | 0.0009507741 | variational/limited_pdhg.json |

Poisson metadata为log_projection，数据满足y=-log(max(counts,1)/I0)。build_fidelity明确将公共counts/I0构造为TransmissionPoissonFidelity，通过transmission scale连接对象与线积分，没有替换为L2。实际Poisson/有限角tau=sigma分别为0.021375333341207、0.015717731380341344。角度子集回归验证每个请求角度恰被覆盖，全部子算子执行forward/adjoint；真实OS-SART仍按subset_count=10调度。

### 伴随与自动步长

seed=71，分母为实际非零abs(<Ax,y>)。下表为float32。ASTRA二维CUDA使用ASTRA线性投影矩阵及其GPU转置。

| 算子 | CPU相对内积误差 | CUDA相对内积误差 |
| --- | --- | --- |
| radon | 7.58268968e-08 | 1.10201228e-07 |
| limited | 0 | 2.00820594e-07 |
| astra2d | 1.79925976e-07 | 2.1528254e-07 |
| tv | 0 | 3.49707165e-07 |
| haar | 8.15808363e-08 | 0 |
| 圆轨迹cone_3d匹配射线积分 | CPU显式矩阵回归另已通过 | 1.42538795e-07 |

小矩阵解析参考A=diag(1,2,3,4)，Tikhonov strength=0.1，float64，未提供步长：FISTA在314次收敛，解误差6.223e-9，step=0.061875，step*||A||²=0.99；PDHG在136次收敛，解误差5.845e-10，tau=sigma=0.24011026878596964，组合范数平方上界17，乘积0.9801<1。两者轨迹有限且达到解析解。

实际默认路径优先计算确定的Schur范数上界；FISTA不能求上界时使用幂迭代估计并检查光滑主函数，PDHG使用投影与正则变换组合上界。未把上界称为幂迭代结果。test_gradient_haar_norms另验证四种实/复精度的幂迭代估计相对显式谱范数误差<1%；test_steps_and_scaled_stacks验证安全条件、过大步长主函数校正和零算子。共5项通过，证据在shared_math/numerics.json。

圆锥线性接口采用分块三线性中点射线积分及同权重散射转置，采样为半体素量级，无稠密三维矩阵；保留原生FDK_CUDA重建。该离散投影与ASTRA原生FP3D不逐点相同，原数据不变，投影残差按当前公开算子解释。CPU/CUDA、float32/float64的独立显式矩阵及自动微分4项回归通过（cone_regression.log）。

### 三维FDK与GPU生命周期

实际ID为`cone_3d/spheres_astra_12`。ASTRA CUDA、Ram-Lak、short_scan=false、voxel_supersampling=1；输出`(1,12,12,12)` / float32且有限，success，iterations=0，direct_reconstruction_valid，converged=false。调用者观测逐元素不变，本次ASTRA对象释放完成。FDK后端8项回归通过，新增同输入重复调用一致性与输入不变断言的单项回归通过。

体指标：PSNR=25.417250，SSIM=0.86791807（全部轴向切片均值），RMSE=0.0010719327。

| 中心切片 | PSNR | SSIM | RMSE |
| --- | --- | --- | --- |
| axial | 22.859875 | 0.88916099 | 0.0014389186 |
| coronal | 22.917156 | 0.96810085 | 0.0014294606 |
| sagittal | 22.918036 | 0.96810651 | 0.0014293159 |

进程级测量使用Windows `Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory.DedicatedUsage`，按当前PID汇总bytes。独立预热3次后，同进程正式连续run_ct_task 50次，实际执行FDK_CUDA。每次校验形状与有限值、删除结果、同步CUDA后采样，核对ASTRA data2d/data3d/projector/algorithm/matrix管理器恢复基线；未使用额外astra.clear()或重启进程，没有用PyTorch allocator冒充进程显存。

预热后起点、结束及同步采样峰值均为 **71,970,816 bytes**，51个测量点一致，无随次数持续增长。峰值指调用结束后采样峰值，不是内核执行中的瞬时最大占用。对象每次均释放。结论限于本次小型FDK路径，不扩展为所有算法和GPU的无限期保证。

### 修复与剩余限制

本次修复：公开cone线性接口采用共享射线权重及精确转置；FDK使用ASTRA自有缓冲，保证输入保护和重复调用稳定。新增/相关回归结果见上文；数据集及二维求解器未修改。

严格CT/TV、二维几何参考、角度子集和光子数据定向运行总计 **32 passed、2 failed、0 skipped**，1400.46秒。保留原阈值和全部断言。

- `test_nonconstant_ct[False-cpu]`：PDHG在20000次后max_iterations，残差0.2000012696，TV=5.6225454282，TV gap=1.88072384e-4，优化残差6.46885690e-5；未达到converged及gap<1e-5。
- `test_nonconstant_ct[False-cuda]`：独立SciPy SLSQP参考在392次以`More than 3*n iterations in LSQ subproblem`失败，未进入待验收求解器；不能把近似参考记为成功。
- 有限角严格PDHG的CPU/CUDA均通过：13925/13926次，gap=8.37357e-6 / 8.37147e-6。ASD的stalled状态保留，不宣称全局最优。

这两个稀疏严格失败与既有记录一致，本次三维修改不涉及对应二维求解路径。保留为未解决的历史优化精度/参考求解限制；不扩大二维求解器重构。LEAP真实后端仍未验收。规定的二维场景、伴随、自动步长、完整FDK保护和50次生命周期证据已补齐，但严格回归未全部通过，**不能声明全方案验收通过**。

禁止行为检查覆盖`inv_framework/ tools/ tests/ configs/`的Python/YAML/TOML代码，搜索sha256、checksum、manifest、environment_2d无匹配；规范术语不作为生成行为。本次未新增哈希、快照或额外审计体系。
