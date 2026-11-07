# 数值与接口验证

测试从本仓库根目录运行，开发依赖使用 `python -m pip install ".[dev]"`。
ASTRA CUDA、LEAP、非 Cartesian MRI 与 VarNet 分别需要对应后端或权重。

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

ASTRA FDK 使用独立工作缓冲，避免原地滤波改变调用者观测；GPU-link 与
CPU-copy 路径不保证数值相同。ASTRA 对象由所属执行路径在 finally 中释放。

二维椭圆与公共图像基准使用模型匹配的合成观测；MR 模体、理想 Airy 核及
VarNet 模体推理不代表临床或真实显微成像质量。PSNR/SSIM 不能替代算子和
优化机制检查。评价参考只用于重建后的独立评价，默认参数不代表最优调参。
