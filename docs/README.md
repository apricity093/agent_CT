# 文档

## 通用入口

- [架构与术语](architecture.md)：执行流程、模块职责和领域词汇。
- [依赖与安装](../DEPENDENCIES.md)：CT、MRI、反卷积的基础依赖、extras、CUDA/ASTRA/LEAP、模型与数据要求。
- [任务 API](agent_algorithm_api.md)：独立调用、文件任务、参数、结果与模块边界。
- [CT 算法](algorithms.md)：组件组合、数学约定、约束 TV 与来源许可。
- [CT 数据](datasets.md)：二维椭圆案例、三维锥束案例与输出格式。
- [公共基准](public_benchmarks.md)：MRI 与反卷积的数据来源、算法对应关系及复现命令。
- [算法来源](source_references.md)：论文、参考实现及本项目的实现差异。
- [验证说明](validation.md)：数值与接口测试入口、结果解释和适用限制。

## 领域手册

| 领域 | 算法 | 数据格式 | 任务 |
| --- | --- | --- | --- |
| CT | [算法](ct/algorithms.md) | [输入](ct/data_format.md) | [任务](ct/tasks.md) |
| MRI | [算法](mri/algorithms.md) | [输入](mri/data_format.md) | [任务](mri/tasks.md) |
| Deconvolution | [算法](deconvolution/algorithms.md) | [输入](deconvolution/data_format.md) | [任务](deconvolution/tasks.md) |

- [Public algorithm index](algorithms/README.md), [composition boundary](composition.md), [parameter search](parameter_search.md), [minimal public calls](algorithm_examples.md).
