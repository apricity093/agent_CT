# MRI data conventions

## 用户数据接入与完整配置

独立配置模板：[request_template.yaml](request_template.yaml)，适用路径为 **二维 Cartesian CG-SENSE**。
它不引用最小 CPU 示例，也不附带合成数据。复制到自己的任务目录后，按注释填写
真实输入路径、尺寸和物理参数，即可通过公开 runner 执行；无需案例注册。

1. 将模板复制为任务目录中的 `request.yaml`，把自己的 NPY/NPZ 文件放在该目录或填写绝对路径。
2. 替换全部采集相关字段。同步修改 spatial_shape、输入的 batch/coil 轴、mask 和灵敏度图；不能仅将 sampling 改为 noncartesian 而缺少 trajectory 及其单位。校准任务的输入输出另见下文。
3. 参数和预算数值是合法配置示意，不是所有数据的推荐值。未显式设置的参数使用当前安装版本的算法默认值，可查询 `list_capabilities("mri")`。
4. 无参考图像时保留 evaluation 为缺省；有参考时按模板注释增加整个映射，填写真实 data_range，不写 `evaluation: null`。
5. 选择新的结果目录执行：

```powershell
python -m inv_framework.runner --config <任务目录>/request.yaml --output <新的结果目录>
```

文件配置的相对路径只相对配置文件所在目录，不回退 cwd；CLI 输出目录相对调用目录。
NPY 引用为 `{path: data/array.npy}`，不带 key；NPZ 引用为
`{path: data/arrays.npz, key: array_name}`。输入 shape/dtype、模型和评价约定见下文。
“完整”指所选任务必需信息完整，不要求同时包含所有算法的互斥输入。

已填好小数据的 CPU 接入实例另见 [最小示例](../../examples/minimal_cpu/mri/README.md)。
该实例用于快速验证接口，不作为本规范或模板的前置依赖。

## 输入字段

| Field | Required | Shape / axes | dtype | Units, meaning and constraints |
| --- | --- | --- | --- | --- |
| inputs.kspace | Reconstruction | Cartesian `(B,C,*spatial)`; non-Cartesian `(B,C,M)` | finite complex | Native Fourier signal; batch B, coil C, samples M |
| inputs.sampling_mask | Cartesian | exactly kspace shape | binary real/bool | Acquired-sample indicator |
| inputs.sensitivity_maps | SENSE/CG/CS/composition | `(B,C,*spatial)` | finite complex | Dimensionless sensitivity; determines object scale |
| inputs.trajectory | Non-Cartesian | `(M,spatial_dimension)` | finite real | Units declared below; shared by batch and coil |
| inputs.data_weights | WLS; optional Gaussian NLL | exactly kspace shape | nonnegative real | Inverse signal variance |
| inputs.initial_estimate | Optional iterative | `(B,*spatial)` | finite complex | Output object scale |
| inputs.calibration_data | ESPIRiT/GRAPPA | `(B,C,acs_y,acs_x)` | finite complex | Fully acquired native ACS rectangle |
| model.spatial_shape | Required | 2 or 3 positive dimensions | int list | Voxel grid, e.g. `[32,32]` |
| model.axes | Required | `[batch,coil,y,x]` or `[batch,coil,z,y,x]` | string list | Explicit axis order |
| model.sampling | Required | scalar | string | cartesian or noncartesian |
| model.fft_centered / fft_norm | Optional, defaults true / ortho | scalars | bool/string | Only true and ortho accepted |
| model.trajectory_units | Non-Cartesian | scalar | string | radians/voxel, cycles/voxel, grid_indices |
| evaluation.reference / data_range / mode | Optional; range required with reference | output shape / scalar / string | finite array / positive real / string | Matched object or RSS target; magnitude or complex mode |

Example maps have `sum_coil abs(S)^2=1` per voxel. The API does not normalize
caller maps. Rescaling S changes object scale and the meaning of regularization;
reference must use the same convention. Physical signal calibration is external.

## Output scale and actual precision

CG-SENSE returns complex `reconstruction` `(B,*spatial)` in sensitivity/FFT scale.
Native classical SENSE converts k-space to complex128; this example computes and
returns complex128, including when input is complex64. Native zero-filled SENSE
is complex128, RSS is float64. GRAPPA returns complex128 `kspace` `(B,C,y,x)` and
float64 RSS `reconstruction` `(B,y,x)`. ESPIRiT returns complex128
`sensitivity_maps` `(B,C,y,x)` and float64 `eigenvalues` `(B,y,x)`, not an image.
Selected eigenvectors have unit coil norm on retained support, zero outside the
crop and a phase anchor. VarNet uses a separate float32 tensor path; consult
actual artifact dtype for each output. Shared solver precision is in diagnostics.

RSS is `sqrt(sum_coil abs(coil_image)^2)` without object phase; ideal fully sampled
intensity is object magnitude times the square root of coil power. Use a matching
RSS reference if sensitivities are not normalized. Maps, complex objects and RSS
are distinct outputs.

Every output is saved as a named `.npy`. output_artifacts reports actual shape,
dtype, axes, representation, units and scale; path is relative to result.json
(null in memory). Signal units are arbitrary unless externally calibrated; maps
and eigenvalues are dimensionless. No intensity rescaling occurs. This example
evaluates magnitude against reference key `reference` with data_range=0.9.
Without reference there is no PSNR/SSIM/RMSE; data residuals remain available.

The top-level policy maps max_iterations=50 to max_iter and tolerance=1e-6 to
the CG relative normal-equation residual, not image error. With lam=0.001, the
recorded run stops at 10 iterations. Success denotes valid execution; exhausted
budgets can succeed with converged=false.

inputs: kspace, sampling_mask (Cartesian), trajectory (non-Cartesian),
sensitivity_maps, optional data_weights and initial_estimate. Values are native
arrays or explicit NPY/NPZ path/key references. All observations are finite.
Model axes must be [batch,coil,y,x] or [batch,coil,z,y,x]; spatial_shape declares
the corresponding positive dimensions. Cartesian kspace/maps are
(batch,coil,*spatial_shape), non-Cartesian kspace is (batch,coil,sample).
Reconstruction and initial_estimate are (batch,*spatial_shape).

model.sampling is cartesian or noncartesian. fft_centered=true and fft_norm=ortho
are the only conventions. FFT = fftshift(fftn(ifftshift(x),norm='ortho')) over
spatial axes only. Cartesian mask has exactly kspace shape and binary entries;
broadcast explicitly before calling. Observed zeros remain sampled; unobserved
entries are ignored regardless of their value. data_weights have kspace shape,
are real nonnegative inverse-variance weights, and default to ones.

Trajectory is (sample,spatial_dimension), shared across batches and coils.
trajectory_units is radians/voxel, cycles/voxel or grid_indices. At entry cycles
multiply by 2*pi, centered grid indices multiply by 2*pi/spatial_shape. Converted
coordinates must be in [-pi,pi). Spatial coordinates are n-floor(N/2), with
negative exponential forward and conjugate positive exponential adjoint.
Physical inverse-length coordinates require caller conversion with voxel size.
Install the mri extra for NUFFT. There is no second NUFFT backend.
Density compensation is not implicitly applied: explicitly pass it as data_weights
if the intended objective uses it. No implicit calibration or gridding baseline.

Evaluation via runner is optional: evaluation: {reference: {path: truth.npy},
data_range: 1.0, mode: magnitude}. Magnitude mode compares abs(reconstruction)
with abs(reference), without scaling or phase alignment; complex mode returns
complex RMSE and PSNR without SSIM. Use a matching RSS or object reference for
the selected coil combination. Real SSIM uses local population moments with
window up to seven spatial pixels/voxels and averages valid centers, independently
across batch. Exact zero RMSE omits infinite PSNR from finite JSON. Truth is
passed exclusively to the evaluator after reconstruction.

Calibration/GRAPPA additionally accept calibration_data (batch,coil,acs_y,acs_x), a fully acquired Cartesian ACS rectangle with native complex values. ESPIRiT accepts only that input, with task_type=calibration. GRAPPA accepts kspace, sampling_mask and calibration_data only. CS uses the same encoding inputs as cg_sense; no implicit calibration occurs.

VarNet accepts only kspace and sampling_mask, native complex (batch,coil,y,x),
at least two coils. Masks have exactly this shape and must be constant across
coils and readout y; columns x are the phase encoding dimension. The explicit
params.num_low_frequencies declares a fully sampled centered ACS band starting
at (width-num_low_frequencies+1)//2. Default 24 is an example acquisition setting,
not automatic calibration inference; change it to the actual acquisition width.
params.checkpoint is a local file name; file-task relative checkpoint paths resolve
against that task file. No array reference or checkpoint is fetched at execution.
Outputs are reconstruction (batch,y,x) magnitude RSS and complex kspace on the
original grid. Missing checkpoint reports missing_checkpoint. Structure mismatches
report invalid_parameters. See ../public_benchmarks.md for preparation.
