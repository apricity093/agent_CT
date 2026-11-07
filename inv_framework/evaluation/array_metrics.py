"""Reference-only evaluation, without intensity fitting or solver access."""
import numpy as np
from scipy.ndimage import uniform_filter
from inv_framework.experiments import read_array


def evaluate_array(reconstruction, evaluation, *, spatial_axes, mri=False):
    truth = read_array(evaluation["reference"])
    truth = truth.astype(np.complex128 if np.iscomplexobj(truth) else np.float64, copy=False)
    mode = evaluation.get("mode", "magnitude" if mri else "real")
    if mode not in ({"magnitude", "complex"} if mri else {"real"}):
        raise ValueError("Unsupported evaluation mode")
    x = np.asarray(reconstruction, dtype=np.complex128 if np.iscomplexobj(reconstruction) else np.float64)
    if truth.shape != x.shape or not np.isfinite(truth).all():
        raise ValueError("Evaluation reference must be finite and match reconstruction")
    if mode == "magnitude":
        x, truth = abs(x), abs(truth)
    if mode == "real" and np.iscomplexobj(truth):
        raise ValueError("Real evaluation requires real reference")
    data_range = float(evaluation["data_range"])
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("evaluation.data_range must be positive")
    mse = float(np.mean(abs(x-truth)**2))
    # An exact match has infinite PSNR; omit it from finite JSON metrics.
    metrics = {"rmse": float(np.sqrt(mse))}
    if mse > 0:
        metrics["psnr"] = float(10*np.log10(data_range**2/mse))
    if mode != "complex":
        size = [1]*x.ndim
        for a in spatial_axes:
            size[a] = min(7, x.shape[a] if x.shape[a] % 2 else x.shape[a]-1)
        mean_x, mean_y = uniform_filter(x, size), uniform_filter(truth, size)
        vx = uniform_filter(x*x, size)-mean_x**2
        vy = uniform_filter(truth*truth, size)-mean_y**2
        cov = uniform_filter(x*truth, size)-mean_x*mean_y
        c1, c2 = (.01*data_range)**2, (.03*data_range)**2
        score = ((2*mean_x*mean_y+c1)*(2*cov+c2))/((mean_x**2+mean_y**2+c1)*(vx+vy+c2))
        crop = tuple(slice(n//2, -n//2+1 or None) if n > 1 else slice(None) for n in size)
        metrics["ssim"] = float(score[crop].mean())
    return metrics
