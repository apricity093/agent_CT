# ct 最小 CPU 示例

在算法库根目录执行（使用已安装依赖的 Python）：

```powershell
python -m inv_framework.runner --config examples/minimal_cpu/ct/request.yaml --output outputs/minimal_cpu/ct
```

已有同名输出时，将 --output 改成新的目录，保留原结果。本次真实结果已保存在
[结果目录](../../../outputs/minimal_cpu/ct/result.json) 及同目录 reconstruction.npy。
Windows 已验证解释器为 `D:/anaconda3/envs/inverse-agent-round1/python.exe`。

## 输入与完整配置

直接使用附带的 [input.npz](input.npz)，无需生成或下载权重。
[完整配置 request.yaml](request.yaml) 是唯一配置来源；[数据规范](../../../docs/ct/data_format.md) 说明每个字段。
输入生成方法保存在 [generate_inputs.py](../generate_inputs.py)，库根目录可执行
`python -m examples.minimal_cpu.generate_inputs`，但它拒绝覆盖已存在的输入/配置。
32×32 双椭圆，衰减幅度 0.014+0.006。使用现有 ParallelBeamRadon2D 在 CPU 合成 90 个 [0,pi) 均匀角度的无噪投影，不使用 tests/data。

| NPZ key | shape | dtype |
| --- | --- | --- |
| observation | `[1, 1, 90, 32]` | `float32` |
| reference | `[1, 1, 32, 32]` | `float32` |

配置的输入与 reference 引用均相对 request.yaml 所在目录；配置可用绝对路径从其他 cwd 调用。
CLI --output 相对调用 cwd。reference 只用于事后评价，不进入求解器。

## 实测结果

2026-11-10，在上述解释器通过公开 CLI 执行，device=cpu：

- status=success，converged=false，iterations=0，stopping_reason=direct_reconstruction_valid。
- 输出 `reconstruction.npy`，shape=`[1, 1, 32, 32]`，dtype=`float32`；所有元素有限。
- result.json 的 output_artifacts 含实际 shape、dtype、axes、表示、单位和尺度；数组路径相对 result.json。outputs 保留相同数组路径。
- runtime_sec=0.147378 秒（库内计时，不包含整个解释器启动过程）。
- rmse=0.0014566453, psnr=22.753523, ssim=0.65001041；PSNR 单位 dB，SSIM 无量纲，RMSE 使用对象单位。

原生任意衰减尺度，reference 最大值 0.02，data_range=0.02；无 HU 换算。

FBP 是直接法，无迭代预算或提前停止。iterations=0、converged=false 与 success 一致；停止原因 direct_reconstruction_valid 表示有效完成。插值、离散滤波和有限采样产生误差，不期待精确反演。

success 表示执行完成并返回有效输出；converged 和 stopping_reason 分别描述停止判据。
耗尽预算也可以 success + converged=false。耗时与精确浮点值不是跨机器硬门槛。
移除 evaluation 后仍能重建，且不计算参考图像 PSNR/SSIM/RMSE。
