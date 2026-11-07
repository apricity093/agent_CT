# Inverse imaging experiments

## Language

**Observation**: Measured data available to reconstruction: CT line integrals/counts,
MRI complex k-space, or a blurred intensity/count image.

**Physical model**: Geometry and acquisition conventions defining how an unknown
object produces the observation. Sampling masks and PSF origins are explicit.

**Calibration**: Estimation of MRI coil sensitivities from actual calibration
observations; its output is maps rather than a reconstructed object.

**Blind deconvolution**: Joint estimation of an image and a spatially invariant
PSF from a blurred observation and declared kernel constraints.

**Evaluation reference**: Optional truth used after reconstruction for quality
measurement. It is not an observation, initialization or calibration input.

**Experiment**: One explicitly specified reconstruction, calibration or blind
estimation task. Selection of subsequent experiments belongs to its caller.
