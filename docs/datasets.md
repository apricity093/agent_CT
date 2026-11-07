# CT 案例与运行输出

MRI 与反卷积数据见[公共基准](public_benchmarks.md)。

默认目录为 `tests/data/`，可用 `INVFRAMEWORK_CT_CASE_ROOT` 或 CLI
`--data-root` 指定其他目录。`catalog.json` 供 Agent 发现案例。

## 二维椭圆数据

来源为 `ct-ellipses-new-20261101`。`reference_images/` 保留原始
`train.npy`（400 张）、`val.npy`（80 张）、`test.npy`（40 张），均为
64×64 float32 图像。标准案例使用 `test[0]`，不将测试图像用于调参。
真值为原始图像乘以 0.02；该固定衰减尺度避免单位幅值在像素求和投影下
造成几乎全部光子耗尽。真值、干净投影和观测使用相同几何与归一化。

| case_id | 角度数/范围 | 观测 |
| --- | --- | --- |
| `parallel_2d__ellipses_dense_clean` | 180 / [0°,180°) | 干净线积分 |
| `parallel_2d__ellipses_sparse_gaussian` | 32 / [0°,180°) | 标准差为干净投影 RMS 的 1% |
| `parallel_2d__ellipses_sparse_poisson` | 32 / [0°,180°) | I0=100000，Poisson(I0 exp(-Ax)) |
| `parallel_2d__ellipses_limited_angle` | 60 / [0°,120°) | 干净线积分 |

投影由 `ParallelBeamRadon2D` 生成，属于模型匹配测试，不是独立物理正演验证。
高斯和泊松噪声种子分别为 20261102、20261103。泊松负对数转换以 1
截断计数，原始计数保持不变；数值稳定性保护继续保留。

`arrays.h5` 中 `truth/x` 是 phantom，`measurement/y_clean` 是干净
sinogram，`measurement/y_observed` 是观测 sinogram；泊松场景还保存
`measurement/counts`（transmission_counts）、`measurement/I0`（blank_scan）
和 `measurement/weights`。`case.json` 包含 `angles_rad`、`detector_count`、
`image_shape`、张量布局、观测模型和来源。案例保存不计算内容摘要。

重新转换（支持 tar 或 tar.gz，不要求文件扩展名）：

```bash
python tests/generate_ellipses_cases.py /path/to/ct-ellipses-new-20261101.tar
python -m inv_framework.cli data validate
```

可用 `--sample` 指定测试样本索引，默认 0；索引记录在每个案例的 provenance 中。
使用 `--data-root` 可另建实验目录。生成器保留目标目录 catalog 中已有的三维记录。

## 三维案例

`cone_3d/spheres_astra_12` 的数组位于
`cases/cone_3d__spheres_astra_12/`，case_id 为 `cone_3d/spheres_astra_12`。
FDK 仍使用 ASTRA CUDA，执行后释放资源，并提供三切片评估。

## 执行与输出

公开任务入口为 `inv_framework.api.run_ct_task`；文件输出入口为 `invct`。成功的文件输出运行保存
`reconstruction.pt`、`metrics.json`、`diagnostics.json`、`comparison.png`。
诊断保留参数来源、迭代数、停止原因、收敛状态和 wall-clock runtime。
MLEM/OSEM 的发射计数模型不适用于本数据集的透射计数；透射泊松任务使用
支持 `transmission_poisson_nll` 的可组合求解器。
