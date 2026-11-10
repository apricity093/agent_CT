# CT 外部数据规范

## 用户数据接入与完整配置

独立配置模板：[request_template.yaml](request_template.yaml)，适用路径为 **二维平行束线积分 FBP**。
它不引用最小 CPU 示例，也不附带合成数据。复制到自己的任务目录后，按注释填写
真实输入路径、尺寸和物理参数，即可通过公开 runner 执行；无需案例注册。

1. 将模板复制为任务目录中的 `request.yaml`，把自己的 NPY/NPZ 文件放在该目录或填写绝对路径。
2. 替换全部采集相关字段。同步修改 domain_shape、range_shape、angles_rad 与投影轴序；模板中的 8 个角度只是字段示意，不代表推荐采集方案。counts 输入或 cone_3d 任务需依据下文切换观测模型、几何与适用算法，不可只换文件名。
3. 参数和预算数值是合法配置示意，不是所有数据的推荐值。未显式设置的参数使用当前安装版本的算法默认值，可查询 `list_capabilities("ct")`。
4. 无参考图像时保留 evaluation 为缺省；有参考时按模板注释增加整个映射，填写真实 data_range，不写 `evaluation: null`。
5. 选择新的结果目录执行：

```powershell
python -m inv_framework.runner --config <任务目录>/request.yaml --output <新的结果目录>
```

文件配置的相对路径只相对配置文件所在目录，不回退 cwd；CLI 输出目录相对调用目录。
NPY 引用为 `{path: data/array.npy}`，不带 key；NPZ 引用为
`{path: data/arrays.npz, key: array_name}`。输入 shape/dtype、模型和评价约定见下文。
“完整”指所选任务必需信息完整，不要求同时包含所有算法的互斥输入。

已填好小数据的 CPU 接入实例另见 [最小示例](../../examples/minimal_cpu/ct/README.md)。
该实例用于快速验证接口，不作为本规范或模板的前置依赖。

## 输入字段

## 输入字段

B=batch，C=独立图像通道，V=角度数，N=图像边长。所有数组有限、实数。
NPY 引用 `{path: image.npy}` 不接受 key；NPZ 必须有 key，例如
`{path: input.npz, key: observation}`。文件请求相对配置目录，字典请求相对 cwd；
路径不存在即失败，不回退 cwd。几何数组在 YAML 中直接提供。

| 字段 | 必需性 | shape / axes | dtype | 单位、物理含义及约束 |
| --- | --- | --- | --- | --- |
| inputs.observation | 必需 | 2D `(B,C,V,N)`；3D `(B,rows,V,cols)` | finite real | 线积分或声明的观测域，与 range_shape 一致 |
| inputs.public_measurement.counts | 纯透射 Poisson 必需 | 同 observation | real，整数值 | 非负光子计数 I，可为零 |
| inputs.public_measurement.I0 | 透射模型必需 | 同 observation | real | 正入射光子期望数，逐射线展开 |
| inputs.public_measurement.c_obs | 混合读出噪声使用 | 同 observation | real | 含电子噪声的计数读数，可负，不与 counts 同时提供 |
| inputs.public_measurement.sigma | Gaussian NLL 所需 | 同 observation | positive real | 标准差，与 observation 同单位 |
| inputs.public_measurement.weights | WLS 所需 | 同 observation | nonnegative real | 逆方差权重 |
| inputs.public_measurement.sigma_e | 混合噪声使用 | 同 observation | nonnegative real | 电子噪声标准差，计数单位 |
| inputs.valid_measurement_mask | 可选 | 同 observation 或去掉 batch | binary | 1 为有效射线 |
| model.geometry | 必需 | 字典，见下表 | 数值/list | 投影几何 |
| model.measurement | 必需 | 字典 | strings/numbers | 观测域及噪声 |
| evaluation.reference | 可选，仅评价 | `(B,*domain_shape)` | finite real | 与重建同尺度、同 shape |
| evaluation.data_range | 有 reference 时必需 | 标量 | positive real | 固定评价动态范围 |

## 几何和观测模型

| geometry 字段 | parallel_2d | cone_3d |
| --- | --- | --- |
| type | `parallel_2d` | `cone_3d` |
| domain_shape | `[C,N,N]`，方形 | `[N,N,N]`，立方体 |
| range_shape | `[C,V,N]`，探测器数等于 N | `[rows,V,cols]` |
| angles_rad | V 个弧度角，按投影顺序 | V 个弧度角，按投影顺序 |
| detector_rows / detector_cols | 不使用 | 与 range_shape 一致的正整数 |
| detector_spacing_row / detector_spacing_col | 不支持任意探测器间距 | 正间距 |
| source_origin_distance / origin_detector_distance | 不使用 | 正距离，与间距使用一致长度单位 |

2D torch 旋转插值后沿行求和，像素与探测器采用匹配的离散单位，没有物理像素间距
乘子。额外 spacing 标签不代表已实施的几何变换。ASTRA 支持 2D CPU/CUDA，LEAP 有
平行束绑定；3D 公开任务为 ASTRA CUDA FDK。本接口不承诺 fan-beam、任意体素尺寸或
任意 3D 迭代重建。

本例 `noise_model=gaussian, observation_model=gaussian, observation_domain=line_integral`，
投影实际未添加噪声，observation 为 Ax。透射使用 `noise_model=poisson,
observation_model=transmission_poisson, observation_domain=log_projection`，正 floor 与 scale：
`I ~ Poisson(I0*exp(-scale*Ax))`，对数投影为 `-log(max(I,floor)/I0)/scale`。
混合噪声用 c_obs 代替 I，并声明 noise_model=poisson_gaussian。截断仅用于生成有限对数值，
不修改原始 counts/c_obs。导入不会替调用者生成 observation，须提供对应的对数投影。

L2/WLS 处理投影域；transmission_poisson_nll 使用原始整数 counts 与 I0，不适用于 c_obs。
当前纯 Poisson fidelity 的值为 `I0*exp(-scale*p)+counts*scale*p`（省略常数），不对 counts
或指数统一裁剪；proximal 在 float64 中求解并验证残差。导出对数的 floor 与 NLL 是两回事。
具体适用模型及有限极小值条件由边界校验与能力查询说明。

## 输出、精度与评价

输出 reconstruction：2D `(B,C,N,N)`，3D `(B,N,N,N)`，无显示归一化。
物理单位由调用者离散投影标定决定。本例任意衰减尺度、参考最大值 0.02，不能标为 HU。
外部 torch 路径保留 float64 输入，否则转 float32；其他后端输入转 float32。
读取 diagnostics.computation_precision 中 measurement/solver_iterate 和每个 output_artifacts.dtype，
不要假设所有算法固定 float32。本例 float32 输入、计算与输出。

output_artifacts 的 path 相对 result.json（内存调用为 null），shape/dtype/axes 描述实际数组；
representation=real，units/scale 表示原生模型尺度。落盘为 reconstruction.npy，无额外缩放。
本例 reference 来自 input.npz，data_range=0.02，RMSE/PSNR/SSIM 在此尺度计算。
外部输入无 evaluation.reference 时不计算参考图像指标，仍可有投影残差。参考仅在重建后读取；
catalog 路径保留可选真值评价。3D RMSE/PSNR 覆盖体积，SSIM 平均轴向切片，正交切片接口仍保留。
FBP iterations=0、converged=false、stopping_reason=direct_reconstruction_valid；success 表示有效完成。
