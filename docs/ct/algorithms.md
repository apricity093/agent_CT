# CT algorithms and parameters

The 17 executable task IDs, fidelity/regularizer requirements, parameter meanings
and bounds are returned by `inv_framework.ct.api.list_capabilities()` from the
existing CT registry. Defaults come only from `configs/algorithms/*.yaml`.
See the [existing algorithm table](../../README.md#支持的算法) and
[public API](../agent_algorithm_api.md) for domain conversions and combinations.
DIP/INR/DPS remain library solvers; they are not additional task capabilities.
