"""Known-PSF quadratic inversion and Poisson EM updates."""
import numpy as np


def wiener_hunt(operator, y, balance):
    laplacian = np.zeros([y.shape[a] for a in operator.axes])
    for axis, size in enumerate(laplacian.shape):
        shape = [1] * laplacian.ndim
        shape[axis] = size
        laplacian += (2 - 2*np.cos(2*np.pi*np.arange(size)/size)).reshape(shape)
    shape = [1] * y.ndim
    for a in operator.axes:
        shape[a] = y.shape[a]
    denominator = abs(operator.otf)**2 + balance * laplacian.reshape(shape)**2
    numerator = operator.otf.conj() * np.fft.fftn(y, axes=operator.axes)
    # Zero modes have no information or penalty: choose the minimum-norm solution.
    spectrum = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator > 0)
    return np.fft.ifftn(spectrum, axes=operator.axes).real


def richardson_lucy(operator, y, initial, gain, background, max_iter, tolerance, epsilon):
    x = initial.copy()
    sensitivity = gain * operator.adjoint(np.ones_like(y))
    history = []
    reason = "max_iterations"
    for iteration in range(1, max_iter+1):
        mean = np.maximum(gain*operator.forward(x)+background, epsilon)
        numerator = gain*operator.adjoint(y/mean)
        update = np.divide(numerator, sensitivity, out=np.zeros_like(x), where=sensitivity > 0)
        next_x = x * np.maximum(update, 0)
        change = np.linalg.norm(next_x-x) / max(np.linalg.norm(x), epsilon)
        x = next_x
        mean = np.maximum(gain*operator.forward(x)+background, epsilon)
        history.append(float(np.sum(mean-y*np.log(mean))))
        if change <= tolerance:
            reason = "converged"
            break
    return x, iteration, reason, history
