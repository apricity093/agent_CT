# Deconvolution inputs and evaluation

## 用户数据接入与完整配置

独立配置模板：[request_template.yaml](request_template.yaml)，适用路径为 **已知 PSF 的 Poisson Richardson–Lucy**。
它不引用最小 CPU 示例，也不附带合成数据。复制到自己的任务目录后，按注释填写
真实输入路径、尺寸和物理参数，即可通过公开 runner 执行；无需案例注册。

1. 将模板复制为任务目录中的 `request.yaml`，把自己的 NPY/NPZ 文件放在该目录或填写绝对路径。
2. 替换全部采集相关字段。按实际数据填写 PSF 原点、边界、gain 和 background。盲反卷积使用不同 task_type、kernel_shape 和输入约定，不能在已知 PSF 模板中只更换算法名。
3. 参数和预算数值是合法配置示意，不是所有数据的推荐值。未显式设置的参数使用当前安装版本的算法默认值，可查询 `list_capabilities("deconvolution")`。
4. 无参考图像时保留 evaluation 为缺省；有参考时按模板注释增加整个映射，填写真实 data_range，不写 `evaluation: null`。
5. 选择新的结果目录执行：

```powershell
python -m inv_framework.runner --config <任务目录>/request.yaml --output <新的结果目录>
```

文件配置的相对路径只相对配置文件所在目录，不回退 cwd；CLI 输出目录相对调用目录。
NPY 引用为 `{path: data/array.npy}`，不带 key；NPZ 引用为
`{path: data/arrays.npz, key: array_name}`。输入 shape/dtype、模型和评价约定见下文。
“完整”指所选任务必需信息完整，不要求同时包含所有算法的互斥输入。

已填好小数据的 CPU 接入实例另见 [最小示例](../../examples/minimal_cpu/deconvolution/README.md)。
该实例用于快速验证接口，不作为本规范或模板的前置依赖。

## 输入字段

| Field | Required | Shape / axes | dtype | Units, meaning and constraints |
| --- | --- | --- | --- | --- |
| inputs.observation | Always | 2D/3D spatial, optional last channel | finite real | Counts or Gaussian signal intensities; Poisson nonnegative |
| inputs.psf | Known PSF | spatial kernel only | finite real | Dimensionless coefficients; RL nonnegative with positive sum |
| inputs.initial_estimate | Optional known-PSF iterative | observation shape | finite real | Object x units; RL nonnegative |
| inputs.initial_psf | Optional blind | kernel_shape | finite real | Initial kernel, subject to blind method constraints |
| model.axes | Always | `[y,x]`, `[z,y,x]`, optionally last channel | string list | No implicit batch or channel mixing |
| model.psf_origin | Known PSF | one index per spatial kernel axis | int list | Index of zero displacement, within support |
| model.boundary | Always | scalar | string | periodic wraps; zero assumes zero outside image |
| model.noise_model | Explicit generation assumption | scalar | string | poisson or gaussian; fidelity must be supported |
| model.gain | RL optional, default 1 | scalar | positive finite real | Counts per unit blurred object intensity |
| model.background | RL optional, default 0 | scalar | nonnegative finite real | Additive expected counts per pixel |
| model.kernel_shape | Blind | two odd sizes >=3 within image | int list | Estimated kernel support, centered output origin |
| evaluation.reference / data_range | Optional; range required with reference | output shape / scalar | finite real / positive real | Same object units as reconstruction |

Known PSFs are not normalized automatically. Unit-sum kernels preserve constants
with periodic boundaries. Origin is explicit, not inferred from maximum or FFT
shift. Forward displacement is kernel_index-origin, with matched adjoint for
each boundary. Zero boundaries reduce edge sensitivity; RL uses gain*H^T(ones).

## Poisson model, outputs and evaluation scale

RL uses `mean=gain*H(x)+background`. A positive initial_value is used unless
initial_estimate is supplied. Mean uses an epsilon floor in the update and
objective; multiplicative updates retain nonnegativity. Output is object x;
do not apply `(x-background)/gain` to it. This example uses seed 42, gain=30,
background=0.2, a unit-sum 3x3 PSF and an object valued 0.1 or 0.9. Evaluation
compares output directly with reference at data_range=0.9, without rescaling.

Older public Poisson benchmarks use another parameterization:
`tools/generate_public_benchmarks.py` generates Poisson(40*H(object)+0.2), while
`standard_public_*_poisson_richardson_lucy.yaml` and `tools/benchmark_runs.py:task`
omit gain (default 1) and set background=0.2. Their unknown is x_counts=40*object.
Independent evaluation in Bench.run divides output by photon_gain=40 only; it
does not subtract background again. The runner does not perform this conversion.
Runner evaluation of that unchanged config requires a counts-scale reference
and data_range=40, or independent division by 40 before unit-scale evaluation.
The minimal example explicitly sets gain; the fixture in
tests/core/fixtures/imaging_inputs.py also sets gain=30 and evaluates object scale.

Known-PSF output is reconstruction with observation shape; the known input PSF
is not returned as an estimate. Native RL and Wiener-Hunt convert inputs to
float64 and return float64, including integer/float32 input. No [0,1] clipping,
display normalization or rescaling occurs. Blind output additionally includes
dimensionless nonnegative unit-sum psf with kernel_shape and centered origin
kernel_shape//2. Native blind image and kernel outputs are float64. Inspect
actual dtype and compute precision for other algorithm paths.

output_artifacts records path relative to result.json (null in memory), shape,
dtype, axes, representation, units and scale. Each output has its own `.npy`.
RL stops when norm(x_new-x)/max(norm(x),epsilon) <= tolerance. Top-level
max_iterations=50 and tolerance=0.003 map to max_iter/tolerance; nonnegative=true
is intrinsic. The recorded run stops at 38; this criterion does not certify
recovery of the noise-free object. Longer unregularized RL can amplify noise.
Budget exhaustion may still be success with converged=false. Wiener-Hunt instead
requires Gaussian Hx, periodic boundary, gain=1, background=0; it does not use
RL's Poisson model.

inputs contains finite real observation, psf, and optional initial_estimate (iterative methods), as arrays or NPY/NPZ path/key references. model.axes is [y,x], [z,y,x],
[y,x,channel] or [z,y,x,channel]. PSF contains spatial axes only. No channel mixing
or batch dimension is implicit. psf_origin explicitly indexes each PSF axis.
model.boundary is periodic or zero. noise_model is gaussian (Wiener/HQS), poisson (RL/RL-TV), or either for tv_admm; RL accepts scalar positive gain and nonnegative background in count units.
Neither PSF nor data is silently normalized. Pixel spacing in the Laplacian is one.

Evaluation in runner is optional: evaluation: {reference: {path: truth.npy},
data_range: 1.0}. Real RMSE, PSNR and local SSIM use original intensities, without
fitting to truth. SSIM uses population moments, window up to seven per spatial
axis, valid centers, independent channels. Exact zero RMSE omits infinite PSNR
from finite JSON. No truth reaches the solver. Without reference, execution still
returns reconstruction and data diagnostics. Direct domain APIs do not evaluate
or save; inv_framework.evaluation.array_metrics.evaluate_array is independently
callable on returned arrays with spatial_axes.

Blind methods require task_type=blind_deconvolution. inputs contains only
observation and optional initial_psf; true PSF/image are never solver inputs.
model.axes is [y,x] or [y,x,channel]; shared spatially invariant Gaussian blur,
periodic boundary, gain=1, background=0. model.kernel_shape contains two odd
sizes >=3 within the observation. The output origin is kernel_shape//2 (an
explicit psf_origin, if supplied, must match). Output psf is a nonnegative
unit-sum spatial array; reconstruction preserves the observation shape.
Kernel support, residual, mass/minimum, scale/iteration history, inner solver
termination and final recovery settings are diagnostics. Original outputs remain
unaligned; any shift-aligned scoring belongs exclusively to independent evaluation.
