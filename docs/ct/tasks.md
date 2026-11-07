# CT tasks

Reconstruct attenuation from explicit parallel-2D or cone-3D acquisition data.
Use `inv_framework.ct.api` for discovery, validation and one solve. The existing
catalog route remains optional; external arrays/files use `CTTaskRequest.inputs`.
Truth is optional evaluation data. No reconstruction parameter is inferred from
truth. [Public API](../agent_algorithm_api.md) describes statuses and stopping.
