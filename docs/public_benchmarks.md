# 标准轻量展示与基准数据

MRI 与 deconvolution 的主展示和质量基准使用各领域 catalog 中的
`primary_benchmark_group=public_standard`。当前只维护这组标准数据及对应结果。
CT 数据见[案例说明](datasets.md)。

## 采用的数据与来源

反卷积使用 scikit-image 0.26.0 的 camera、astronaut 和 cell，明确转换为
256×256、浮点 [0,1]；astronaut 先转标准亮度灰度，所有 resize 使用抗混叠。
camera 为 Lav Varshney 摄影的 CC0 版本；astronaut 来自 NASA，公开域、无已知
版权限制；cell 为 CC0 定量相位图。来源和转换记录在各 case.json。
cell 被作为非负强度模板合成 Poisson 观测，不宣称它是荧光光子采集。
见 [scikit-image 数据说明](https://scikit-image.org/docs/stable/api/skimage.data)。

8 个 Levin09 核来自 [DPIR 的公开核目录](https://github.com/cszn/DPIR/tree/master/kernels)，
原作者为 Levin、Weiss、Durand、Freeman，CVPR 2009。
MATLAB HDF5 cell 转成 NPY 语义时转置存储轴，不旋转、不重新居中、不改变支持。
原始支持为 19、17、15、27、13、21、23、23 像素的方阵，总质量约为 1。
NPZ 合计 10,736 字节；原始下载 MAT 为 32,224 字节，不需要下载整个 DPIR。
保留该来源仓库的 MIT 许可全文于 sources/DPIR_LICENSE.txt，并保留原始论文归属；
不将来源仓库的许可另行表述为原作者单独签发的数据许可。
当前固定例选 kernel 1 用于 camera 开发例、kernel 2 用于 astronaut 评价例；
其他核可编辑配置选取，没有把所有图像与核做笛卡尔积。

cell 使用 15×15 理想二维 Airy 核，形状为 [2J1(r)/r]²，中心取极限 1，
第一暗环半径为 3 像素，截断后归一化。它是理想标量衍射 PSF，
没有折射率失配、离焦或像差，不称为完整 Gibson–Lanni 显微镜模型。
基础观测仍为模型匹配的周期卷积，加 Gaussian 或 Poisson 噪声，
不是 Levin 原始实拍模糊图像对。真实核只传入非盲任务，盲任务只获得观测和核支持。

MRI 使用 [phantominator 0.7.0](https://github.com/mckib2/phantominator) 的
MR Shepp–Logan 标准组织参数模体（Gach 等，2008），来源生成器 MIT。
根据 M0、T1、T2 生成 M0(1−exp(−TR/T1))exp(−TE/T2) 对比度，
显式保存 B0、TR、TE、z 截面和复数相位约定，不使用 CT 衰减模体冒充 MR 组织。
这是解析 MR 模体，不是真实脑切片或 BrainWeb。两种截面/对比度是标准模型的
开发与评价条件，不是独立临床受试者。

4 个理想圆形接收线圈由 Biot–Savart 中点积分产生：环分为 64 段，中心环半径 1.6，
线圈半径 .55，FOV 为 [−1,1]²，复数 maps 为 Bx+iBy 并归一化为单位 RSS。
它没有 RF 屏蔽、线圈耦合和高场介电效应，不声称校准一定稳定。
观测使用独立 NumPy 中心正交 FFT，径向组使用显式 DFT；已有重建实现不变。
Cartesian 规则 R4 外围采样含实际提供的连续 ACS；ACS 严格来自公开含噪观测。

## fastMRI 和 VarNet

fastMRI GitHub 的测试 fixture 会生成随机 mock 数据，不能标成真实膝关节采集。
参见 [官方测试生成器](https://github.com/facebookresearch/fastMRI/blob/main/tests/create_temp_data.py)。
官方数据的研究/教育用途及再分发条件独立于代码 MIT 许可，见
[NYU 数据条款](https://fastmri.med.nyu.edu/)。没有可自由再分发的真实多线圈
单切片时，不将它加入版本库。标准模体上的 VarNet 仅作为推理流程示例，不提供真实膝关节质量结论。

已提供 `tools/prepare_fastmri_slice.py`，供以后将用户自行取得授权的完整
多线圈训练/验证 HDF5 提取一片到本地输出目录。保留原始网格，不缩成 128，
采用明确 R4 列掩膜和 24 列 ACS，参考为完整采集的原生 FOV RSS。
该评价不是官方裁剪后的 challenge 分数；data_range 为参考峰值，
不拟合或归一化重建输出。输入及派生切片不应未经许可再分发。

## 复现和独立任务

数据生成依赖可通过 `python -m pip install ".[benchmark-data]"` 安装。
部分 scikit-image 图像在首次准备时按需获取。领域数据位于 `tests/data/`；
运行已有 NPY/NPZ 任务不需要模体生成器。以下命令从本仓库根目录执行：

```powershell
python tools/fetch_public_sources.py       # 仅需重新准备公开核时联网
python tools/generate_public_benchmarks.py
python tools/run_public_benchmarks.py
```

结果保存在指定输出目录，默认 `outputs/public_benchmarks`；
`report.md`、`summary.json` 与图像随运行输出保存。
已有结果可直接阅读[基准报告](../outputs/public_benchmarks/report.md)。仅从已有记录重新生成报告：

```powershell
python tools/run_public_benchmarks.py --report-only
```

该入口只写指定输出目录的 report.md，不执行算法，不改写数据、图片或实验配置。
验证入口见[验证说明](validation.md)。每次调用保存独立 task.yaml、
result.json、evaluation.json、数组与比较图；standard_ 开头的可编辑配置位于
configs/mri/benchmarks 和 configs/deconvolution/benchmarks，不要求登记外部输入。
结果中 success 与 converged 分开，上限保留为 max_iterations。
固定参数仅作为展示条件，不从评价真值搜索更好的分数。

自然图像评价不等于所有真实成像系统；理想 Airy 不等于真实显微镜；解析 MR 模体
不验证临床泛化；已知真核的有限预算非盲结果是参照，不能称为理论上界。

取得授权 HDF5 后，可在本地准备一张切片：

```powershell
python tools/prepare_fastmri_slice.py D:/authorized/knee.h5 --slice 10 --output outputs/fastmri_local --checkpoint outputs/weights/knee_leaderboard_state_dict.pt
python -m inv_framework.runner --config outputs/fastmri_local/varnet.yaml --output outputs/fastmri_local/run
```

outputs/fastmri_local 已排除 Git 跟踪；其他输出位置由调用者决定，仍须遵守原始数据条款。

## 案例与算法对应关系

### MRI

三个标准采集条件都使用 phantominator MR Shepp–Logan 组织参数、
模拟复数相位和 4 个 Biot–Savart 环形线圈，不是真实受试者数据。

| 案例 | 条件 |
|---|---|
| standard_mr_dev | 128×128 Cartesian 开发条件，R4 外围采样，32 列 ACS |
| standard_mr_cartesian | 256×256 Cartesian 评价条件，R4 外围采样，48 列 ACS |
| standard_mr_radial | 128×128 径向评价条件，32 辐条，每辐条 160 点 |

| 算法 ID | 使用的数据和输入 |
|---|---|
| zero_filled | 两个 Cartesian 案例的公开 k-space、掩膜与已知 maps |
| cg_sense | 三个案例；Cartesian 分别使用已知 maps 与显式 ESPIRiT 校准 maps |
| cs_tv | 两个 Cartesian 案例、已知 maps、复数 TV |
| cs_wavelet | 两个 Cartesian 案例、已知 maps、2 层 Haar |
| grappa | 两个 Cartesian 案例，公开 ACS 与采样 k-space，不输入 maps |
| espirit | 两个 Cartesian 案例的公开 ACS，输出 maps，再由调用者明确启动 SENSE |
| varnet | standard_mr_cartesian；只输入 k-space、列掩膜，官方 knee_leaderboard_state_dict.pt 权重 |

VarNet 的当前定位是权重加载、推理和调用流程验证。没有真实 fastMRI 膝关节
质量分数，也没有接入 SigPy 教程真实脑部数据。现有模拟输入足以进行接入冒烟测试，
不用于声明预训练模型的临床效果。权重保留在 outputs/weights，不自动下载。

### Deconvolution

三个案例均为 256×256 标准公开图像，观测采用合成周期卷积和噪声。
Gaussian σ=.005；Poisson 均值为 40Hx+.2，统一在 q=40x 单位求解并除以 40 评价。

| 案例 | 公开图像与 PSF |
|---|---|
| public_camera_levin | camera，Levin09 核 1，19×19，开发条件 |
| public_astronaut_levin | astronaut 转灰度，Levin09 核 2，17×17，评价条件 |
| public_cell_airy | cell，理想 Airy 核 15×15，评价条件 |

| 算法 ID | 使用的数据和输入 |
|---|---|
| wiener_hunt | 三个案例的 Gaussian 观测与真实 PSF |
| hyper_laplacian | 三个案例的 Gaussian 观测与真实 PSF |
| tv_admm | 三个案例的 Gaussian、Poisson 观测与真实 PSF |
| richardson_lucy | 三个案例的 Poisson 观测与真实 PSF |
| rl_tv | 三个案例的 Poisson 观测与真实 PSF |
| blind_dcp | camera、astronaut 的 Gaussian 观测与核支持，不输入真实 PSF |
| blind_l0 | 与 blind_dcp 相同观测和信息条件 |

cell 是定量相位图，用作非负强度模板，不是荧光采集。Airy 是理想衍射核，
不是完整显微镜模型。8 个 Levin 核保存在 sources/levin_kernels.npz，
当前代表实验使用两个，不穷举组合。真实核只在非盲求解或独立评价中使用。
