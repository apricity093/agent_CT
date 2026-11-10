# 文档

本目录涵盖 CT、MRI 与反卷积的任务接口、数据规范、算法使用和数值验证。
首次接入可依次阅读安装说明、任务 API 和对应领域的数据格式；查询具体方法时使用算法索引。

## 安装与任务接入

- [依赖与安装](../DEPENDENCIES.md)：CT、MRI、反卷积的基础依赖、extras、CUDA/ASTRA/LEAP、模型与数据要求。
- [架构与术语](architecture.md)：执行流程、模块职责和领域词汇。
- [任务 API](agent_algorithm_api.md)：独立调用、文件任务、参数、结果与模块边界。
- [最小调用示例](algorithm_examples.md)：三域公开 Python 入口及保真项、正则项的调用示例。

## 领域手册

各领域的算法页说明领域入口和适用条件，数据格式页定义数组布局与物理模型，任务页说明执行范围。
配置模板需按实际采集数据填写路径、尺寸和物理参数，适用任务见表中标注。

| 领域 | 算法 | 数据格式 | 任务 | 用户数据配置模板 |
| --- | --- | --- | --- | --- |
| CT | [算法](ct/algorithms.md) | [输入](ct/data_format.md) | [任务](ct/tasks.md) | [二维平行束 FBP](ct/request_template.yaml) |
| MRI | [算法](mri/algorithms.md) | [输入](mri/data_format.md) | [任务](mri/tasks.md) | [二维 Cartesian CG-SENSE](mri/request_template.yaml) |
| 反卷积 | [算法](deconvolution/algorithms.md) | [输入](deconvolution/data_format.md) | [任务](deconvolution/tasks.md) | [已知 PSF 的 Poisson Richardson–Lucy](deconvolution/request_template.yaml) |

## 算法、组合与数学

- [算法索引](algorithms/README.md)：按方法家族查询三域算法；各方法页列出输入输出、运行示例、默认参数、合法范围和建议搜索范围。
- [CT 算法总说明](algorithms.md)：CT 组件组合、观测域、数学约定、约束 TV 与来源许可。
- [公开组合规则](composition.md)：各领域可用的求解器、保真项与正则项组合，以及参数覆盖优先级。
- [可组合求解器](composable_solvers.md)：共享 ADMM、PDHG、FISTA 的目标、正则定义、子问题求解及诊断。
- [二次问题基线](quadratic_baselines.md)：Landweber、CGLS、LSQR、CG 的加权目标、正则、步长和停止条件。
- [共享数学说明](shared_mathematics.md)：差分、Haar、复数伴随、范数界、CG 与近端精度约定。
- [参数搜索约定](parameter_search.md)：建议搜索范围、固定采集条件和任务上下文候选筛选。

## 数据、来源与验证

- [CT 数据](datasets.md)：二维椭圆案例、三维锥束案例、生成命令与输出格式。
- [公共基准](public_benchmarks.md)：MRI 与反卷积的数据来源、算法对应关系及复现命令。
- [算法来源](source_references.md)：论文、参考实现及本项目的实现差异。
- [验证说明](validation.md)：数值与接口测试入口、验收记录、结果解释和已知限制。
