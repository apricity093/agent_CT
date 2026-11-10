"""Matched circular cone projection by midpoint ray integration.

Trilinear voxel weights are shared by projection and transpose. Rays are
processed in bounded chunks; no dense projection matrix is constructed.
"""
import math
import torch


def cone_apply(tensor, volume_geometry, projection_geometry, forward):
    shape = (volume_geometry['GridSliceCount'], volume_geometry['GridRowCount'],
             volume_geometry['GridColCount'])
    rows, cols = projection_geometry['DetectorRowCount'], projection_geometry['DetectorColCount']
    angles = torch.as_tensor(projection_geometry['ProjectionAngles'], device=tensor.device,
                             dtype=tensor.dtype)
    sino_shape = (rows, len(angles), cols)
    bounds = volume_geometry['option']
    low = tensor.new_tensor([bounds['WindowMinX'], bounds['WindowMinY'], bounds['WindowMinZ']])
    high = tensor.new_tensor([bounds['WindowMaxX'], bounds['WindowMaxY'], bounds['WindowMaxZ']])
    spacing = (high-low)/tensor.new_tensor(shape[::-1])
    # Half-voxel midpoint quadrature, independent of input values.
    samples = math.ceil(float(((high-low)/spacing.min()).norm())*2)
    fractions = (torch.arange(samples, device=tensor.device, dtype=tensor.dtype)+.5)/samples
    output = tensor.new_zeros((tensor.shape[0], *(sino_shape if forward else shape)))
    flat_input, flat_output = tensor.reshape(tensor.shape[0], -1), output.reshape(tensor.shape[0], -1)
    ray_count = math.prod(sino_shape)
    for start in range(0, ray_count, 1024):
        ids = torch.arange(start, min(start+1024,ray_count), device=tensor.device)
        column, angle_index, row = ids % cols, (ids//cols) % len(angles), ids//(cols*len(angles))
        angle = angles[angle_index]
        radial = torch.stack((angle.sin(), -angle.cos(), torch.zeros_like(angle)), -1)
        horizontal = torch.stack((angle.cos(), angle.sin(), torch.zeros_like(angle)), -1)
        source = radial*projection_geometry['DistanceOriginSource']
        detector = -radial*projection_geometry['DistanceOriginDetector']
        detector += horizontal*((column-(cols-1)/2)*projection_geometry['DetectorSpacingX'])[:,None]
        detector[:,2] += (row-(rows-1)/2)*projection_geometry['DetectorSpacingY']
        direction = detector-source
        parallel = direction == 0
        safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
        first, last = (low-source)/safe_direction, (high-source)/safe_direction
        entry = torch.minimum(first,last).masked_fill(parallel, -torch.inf).amax(-1).clamp_min(0)
        exit = torch.maximum(first,last).masked_fill(parallel, torch.inf).amin(-1).clamp_max(1)
        outside = (parallel & ((source<low)|(source>high))).any(-1)
        length = (exit-entry).clamp_min(0).masked_fill(outside,0)
        points = source[:,None,:]+(entry[:,None]+length[:,None]*fractions)[...,None]*direction[:,None,:]
        # ASTRA rows run from positive to negative world Y.
        grid = (points-low)/spacing-.5
        grid[...,1] = shape[1]-1-grid[...,1]
        base = grid.floor().long()
        blend = grid-base
        ray_weight = length*direction.norm(dim=-1)/samples
        for dz in (0,1):
            for dy in (0,1):
                for dx in (0,1):
                    offset = base.new_tensor([dx,dy,dz])
                    index = base+offset
                    valid = ((index>=0)&(index<index.new_tensor(shape[::-1]))).all(-1)
                    weight = torch.where(offset.bool(), blend, 1-blend).prod(-1)*ray_weight[:,None]*valid
                    linear = ((index[...,2].clamp(0,shape[0]-1)*shape[1]
                              +index[...,1].clamp(0,shape[1]-1))*shape[2]
                              +index[...,0].clamp(0,shape[2]-1)).flatten()
                    if forward:
                        flat_output[:,ids] += (flat_input[:,linear].reshape(tensor.shape[0],len(ids),samples)*weight).sum(-1)
                    else:
                        values = (flat_input[:,ids,None]*weight).reshape(tensor.shape[0],-1)
                        flat_output.scatter_add_(1,linear.expand(tensor.shape[0],-1),values)
    return output
