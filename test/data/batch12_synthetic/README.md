# Batch 12 synthetic status evidence

This directory contains only deterministic, ordinary-CT quick-validation
evidence. It does not use USCT code or data and is not a reconstruction-quality
benchmark.

Regenerate the evidence from the CT repository root with:

```powershell
D:\anaconda3\envs\inverse-agent-round1\python.exe test\batch12_synthetic_matrix.py --protocol test\data\batch12_synthetic\protocol.json --output test\data\batch12_synthetic
```

`normal`, `max_iterations`, and `nonfinite` iterative records invoke the real
solver detailed APIs. Invalid records invoke the shared registry validation.
Stall and divergence records use explicitly labelled test-only fault
trajectories through the production status classifier, so fault injection
cannot leak into runtime code. Tolerance and terminal-trajectory records test
the shared classification and the real solver evidence respectively.

FBP has no optional external reconstruction backend, so its `unavailable`
scenario is explicitly `not_applicable`. FDK exercises an actual missing-backend
`unavailable` result. Iteration-only states are also explicitly
`not_applicable` for both direct algorithms.

`checksums.json` covers the exact protocol and results bytes. The focused test
reruns the complete matrix and requires the checked-in result and checksums to
match byte-for-byte.
