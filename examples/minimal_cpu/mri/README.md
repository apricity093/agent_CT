# mri 最小 CPU 示例

在算法库根目录执行（使用已安装依赖的 Python）：

```powershell
python -m inv_framework.runner --config examples/minimal_cpu/mri/request.yaml --output outputs/minimal_cpu/mri
```

已有同名输出时，将 --output 改成新的目录，保留原结果。本次真实结果已保存在
[结果目录](../../../outputs/minimal_cpu/mri/result.json) 及同目录 reconstruction.npy。
Windows 已验证解释器为 `D:/anaconda3/envs/inverse-agent-round1/python.exe`。

## 输入与完整配置

直接使用附带的 [input.npz](input.npz)，无需生成或下载权重。
[完整配置 request.yaml](request.yaml) 是唯一配置来源；[数据规范](../../../docs/mri/data_format.md) 说明每个字段。
输入生成方法保存在 [generate_inputs.py](../generate_inputs.py)，库根目录可执行
`python -m examples.minimal_cpu.generate_inputs`，但它拒绝覆盖已存在的输入/配置。
32×32 复数椭圆对象、线性相位；两个显式线圈灵敏度图逐像素归一化 sum(abs(S)^2)=1。复用现有 imaging_inputs fixture 的 centered ortho FFT 合成方式，每第三行不采样，mask 显式展开到全部 batch/coil。

| NPZ key | shape | dtype |
| --- | --- | --- |
| kspace | `[1, 2, 32, 32]` | `complex128` |
| sampling_mask | `[1, 2, 32, 32]` | `uint8` |
| sensitivity_maps | `[1, 2, 32, 32]` | `complex128` |
| reference | `[1, 32, 32]` | `complex128` |

配置的输入与 reference 引用均相对 request.yaml 所在目录；配置可用绝对路径从其他 cwd 调用。
CLI --output 相对调用 cwd。reference 只用于事后评价，不进入求解器。

## 实测结果

2026-11-10，在上述解释器通过公开 CLI 执行，device=cpu：

- status=success，converged=true，iterations=10，stopping_reason=converged。
- 输出 `reconstruction.npy`，shape=`[1, 32, 32]`，dtype=`complex128`；所有元素有限。
- result.json 的 output_artifacts 含实际 shape、dtype、axes、表示、单位和尺度；数组路径相对 result.json。outputs 保留相同数组路径。
- runtime_sec=0.248056 秒（库内计时，不包含整个解释器启动过程）。
- rmse=0.0019723317, psnr=53.185251, ssim=0.99966225；PSNR 单位 dB，SSIM 无量纲，RMSE 使用对象单位。

输出复数对象；评价 abs(output) 对 abs(reference)，data_range=0.9，不对齐相位或拟合尺度。

顶层 max_iterations=50 是外层 CG 预算，tolerance=1e-6 是相对正规方程残差阈值；不是参考误差。lam=0.001。实测 10 次提前停止，effective_params 和 effective_stopping_policy 可核对实际生效值。

success 表示执行完成并返回有效输出；converged 和 stopping_reason 分别描述停止判据。
耗尽预算也可以 success + converged=false。耗时与精确浮点值不是跨机器硬门槛。
移除 evaluation 后仍能重建，且不计算参考图像 PSNR/SSIM/RMSE。
