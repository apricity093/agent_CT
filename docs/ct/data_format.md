# External CT input

`CTTaskRequest(case_id="my_label", algorithm="sirt", inputs=...)` accepts:

- `observation`: finite real array or NPY/NPZ reference. Parallel shape
  `(B,C,views,detectors)`; cone `(B,detector_rows,views,detector_cols)`.
- `geometry`: `type`, `domain_shape`, `range_shape`, `angles_rad` in radians.
  Parallel domain `(C,N,N)`, range `(C,views,N)`; detector pitch and pixel pitch
  are one matching length unit in the existing projector. No arbitrary detector
  pitch is claimed by the parallel API. Cone domain `(N,N,N)`, range
  `(rows,views,cols)`; geometry additionally requires `detector_rows`,
  `detector_cols`, `detector_spacing_row`, `detector_spacing_col`,
  `source_origin_distance`, `origin_detector_distance` in matching length units.
  Cone requires ASTRA CUDA FDK.
- `measurement`: `noise_model`, `observation_model`, `observation_domain`, plus
  model parameters. Plain Gaussian line integrals use `gaussian`, `gaussian`,
  `line_integral`. Transmission uses `poisson`, `transmission_poisson`,
  `log_projection`, with positive `floor` and `scale` (normally 1).
- `public_measurement`: optional `counts`, `I0`, `sigma`, `weights`, `c_obs`,
  `sigma_e` arrays matching observation exactly. Pure transmission Poisson NLL
  requires nonnegative integer counts and positive I0; line observations are
  `-log(max(counts,floor)/I0)/scale`. Counts and readout c_obs are distinct.
- `valid_measurement_mask`: optional binary projection mask.
- `truth`: optional `(B,*domain_shape)` array; when supplied, positive
  `data_range` is required for metrics. Attenuation/line integrals retain their
  original scale; no display normalization occurs. Runner users supply truth
  under `evaluation.reference` instead of reconstruction inputs.

All CT arrays enter the established float32 path. Do not pass complex arrays.
The runner combines `inputs` (observation/public arrays/mask) and `model`
(geometry/measurement metadata) into the same domain request. See
`configs/ct/benchmarks/external_sirt.yaml`; create the referenced observation
with your acquisition/exporter. No local case registration is required.
