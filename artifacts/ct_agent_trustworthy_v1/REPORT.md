# Batch 13 budgeted ordinary-CT benchmark

- Fixed defaults: 27 transmission + 2 count-domain records; 29/29 accepted.
- FDK: `unavailable` (FDK requires PyTorch CUDA, astra-toolbox, and ASTRA CUDA).
- Protocol study: 7 tunable algorithms, four unique candidates each; one 3-fold history reused by four accepted protocol views.
- Robustness: 144 transmission + 32 count-domain records; 176/176 accepted.
- Parameter modes: fixed defaults, metadata recommendation, bounded held-out tuning, and separately tagged oracle upper bound.
- Metrics remain on five independent axes; no aggregate score is produced.
- Observed wall: 647.732s; peak working set: 271470592 bytes; ignored runtime size: 51539607 bytes.
