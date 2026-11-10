# deconvolution 最小 CPU 示例

在算法库根目录执行（使用已安装依赖的 Python）：

```powershell
python -m inv_framework.runner --config examples/minimal_cpu/deconvolution/request.yaml --output outputs/minimal_cpu/deconvolution
```

已有同名输出时，将 --output 改成新的目录，保留原结果。本次真实结果已保存在
[结果目录](../../../outputs/minimal_cpu/deconvolution/result.json) 及同目录 reconstruction.npy。
Windows 已验证解释器为 `D:/anaconda3/envs/inverse-agent-round1/python.exe`。

## 输入与完整配置

直接使用附带的 [input.npz](input.npz)，无需生成或下载权重。
[完整配置 request.yaml](request.yaml) 是唯一配置来源；[数据规范](../../../docs/deconvolution/data_format.md) 说明每个字段。
输入生成方法保存在 [generate_inputs.py](../generate_inputs.py)，库根目录可执行
`python -m examples.minimal_cpu.generate_inputs`，但它拒绝覆盖已存在的输入/配置。
32×32 正值对象（0.1/0.9），已知单位和 3×3 PSF，原点 [1,1]。使用 imaging_inputs fixture 的独立周期移位求和合成 H(x)，固定 NumPy seed=42，观测 Poisson(30*H(x)+0.2)。

| NPZ key | shape | dtype |
| --- | --- | --- |
| observation | `[32, 32]` | `int64` |
| psf | `[3, 3]` | `float64` |
| reference | `[32, 32]` | `float64` |

配置的输入与 reference 引用均相对 request.yaml 所在目录；配置可用绝对路径从其他 cwd 调用。
CLI --output 相对调用 cwd。reference 只用于事后评价，不进入求解器。

## 实测结果

2026-11-10，在上述解释器通过公开 CLI 执行，device=cpu：

- status=success，converged=true，iterations=38，stopping_reason=converged。
- 输出 `reconstruction.npy`，shape=`[32, 32]`，dtype=`float64`；所有元素有限。
- result.json 的 output_artifacts 含实际 shape、dtype、axes、表示、单位和尺度；数组路径相对 result.json。outputs 保留相同数组路径。
- runtime_sec=0.292234 秒（库内计时，不包含整个解释器启动过程）。
- rmse=0.27453435, psnr=10.312916, ssim=0.43719761；PSNR 单位 dB，SSIM 无量纲，RMSE 使用对象单位。

gain=30（计数/对象强度）、background=0.2（每像素期望计数）。输出已经是对象 x，不再扣背景或除 gain。直接与 reference 比较，data_range=0.9。

顶层 max_iterations=50 是 RL 预算，tolerance=0.003 是相对迭代变化阈值。实测 38 次提前停止；effective_params.max_iter/tolerance 和 effective_stopping_policy 记录映射。nonnegative=true 是 RL 固有约束。低计数噪声明显，RL 无正则化，继续迭代可能放大噪声；收敛阈值不保证高 PSNR。

success 表示执行完成并返回有效输出；converged 和 stopping_reason 分别描述停止判据。
耗尽预算也可以 success + converged=false。耗时与精确浮点值不是跨机器硬门槛。
移除 evaluation 后仍能重建，且不计算参考图像 PSNR/SSIM/RMSE。
