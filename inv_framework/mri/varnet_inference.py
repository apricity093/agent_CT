"""Explicit checkpoint preparation and 2D Cartesian VarNet inference."""
from pathlib import Path
import time
import numpy as np
import torch
from .models.varnet import VarNet
from .models.complex_math import ifft2c, complex_abs, rss


def load_checkpoint(network, path, device):
    """Accept raw fastMRI state or Lightning state_dict with uniform varnet. prefix."""
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get('state_dict', payload)
    if state and all(key.startswith('varnet.') for key in state):
        state = {key[len('varnet.'):]: value for key, value in state.items()}
    network.load_state_dict(state, strict=True)


def infer(arrays, model, p, device):
    if set(arrays) != {'kspace', 'sampling_mask'}:
        raise ValueError('VarNet inputs require only kspace and sampling_mask; sensitivity estimation is internal')
    y, mask = arrays['kspace'], arrays['sampling_mask']
    shape = tuple(model['spatial_shape'])
    if model['sampling'] != 'cartesian' or len(shape) != 2:
        raise ValueError('VarNet requires 2D Cartesian data')
    if y.ndim != 4 or min(y.shape) < 1 or y.shape[1] < 2 or y.shape[2:] != shape or not np.iscomplexobj(y) or not np.isfinite(y).all():
        raise ValueError('VarNet kspace must be finite complex (batch,coil,y,x), at least two coils')
    if mask.shape != y.shape or not np.isin(mask, [0, 1]).all() or not np.all(mask == mask[:, :1, :1, :]):
        raise ValueError('VarNet requires binary Cartesian column mask shared over coils and readout')
    low = p['num_low_frequencies']
    start = (shape[1]-low+1)//2
    if low > shape[1] or not np.all(mask[..., start:start+low]):
        raise ValueError('num_low_frequencies must describe a fully sampled centered ACS band')
    structure = {key: p[key] for key in ('num_cascades','chans','pools','sens_chans','sens_pools')}
    if any(np.prod([int(np.ceil(n/2**pools)) for n in shape]) <= 1 for pools in (p['pools'], p['sens_pools'])):
        raise ValueError('Spatial grid is too small for InstanceNorm at the configured pooling depth')
    if not p['checkpoint'] or not Path(p['checkpoint']).is_file():
        raise FileNotFoundError('VarNet requires an explicitly prepared matching checkpoint; no weights are downloaded during execution')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    load_started = time.perf_counter()
    network = VarNet(**structure).to(device)
    try:
        load_checkpoint(network, p['checkpoint'], device)
    except RuntimeError as error:
        raise ValueError(f'Checkpoint does not match VarNet architecture: {error}') from error
    network.eval()
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    loading_sec = time.perf_counter()-load_started
    data = torch.view_as_real(torch.as_tensor(np.asarray(y*mask, dtype=np.complex64), device=device))
    sampled = torch.as_tensor(mask[:, :1, :1, :, None].astype(bool), device=device)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        maps = network.sens_net(data, sampled, low)
        predicted = data.clone()
        for cascade in network.cascades:
            predicted = cascade(predicted, data, sampled, maps)
        image = rss(complex_abs(ifft2c(predicted)), 1)
        residual = torch.linalg.vector_norm((predicted-data)*sampled).item()
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    inference_sec = time.perf_counter()-inference_started
    return {'reconstruction': image.cpu().numpy(),
            'kspace': torch.view_as_complex(predicted.contiguous()).cpu().numpy()}, dict(
        architecture=structure, checkpoint=str(Path(p['checkpoint']).resolve()), device=device,
        dtype='complex64', num_low_frequencies=low, coil_combination='rss',
        model_loading_sec=loading_sec, inference_sec=inference_sec,
        data_residual=residual, residual_domain='sampled_predicted_coil_kspace',
        input_convention='batch,coil,y,x; centered ortho FFT; Cartesian column sampling')
