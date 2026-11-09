"""Small representative checks of the optional discrete definitions."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from inv_framework.core.operator import GradientOperator
from inv_framework.core.regularizer import TVRegularizer
from inv_framework.core.imaging_operators import KnownConvolution
from inv_framework.solvers.composition import CompositeSolver


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def homoscedastic_equivalence(image):
    from inv_framework.core.fidelity import QuadraticFidelity
    from inv_framework.core.imaging_operators import CartesianSense
    x=image[:,0]
    op=CartesianSense(torch.ones_like(image),torch.ones_like(image.real))
    prediction=op.forward(x)
    rng=torch.Generator().manual_seed(29)
    observed=prediction+.002/np.sqrt(2)*(torch.randn(prediction.shape,generator=rng,dtype=torch.float64)+1j*torch.randn(prediction.shape,generator=rng,dtype=torch.float64))
    c=2/.002**2;lam=.01
    plain=QuadraticFidelity(observed);weighted=QuadraticFidelity(observed,torch.full_like(observed.real,c))
    reg=float(TVRegularizer('isotropic').value(x))
    a=float(plain.value(prediction))+lam*reg;b=float(weighted.value(prediction))+c*lam*reg
    objective_error=abs(b-c*a)/max(1,abs(b))
    gradient_error=float((weighted.gradient(prediction)-c*plain.gradient(prediction)).norm())
    assert objective_error<1e-12 and gradient_error<1e-12
    return dict(complex_sigma=.002,weight=c,unweighted_tv=lam,weighted_tv=c*lam,
        objective_relative_error=objective_error,gradient_difference=gradient_error,
        interpretation='same minimizers after joint fidelity and regularizer scaling; no duplicate quality solve')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    y,x=torch.meshgrid(torch.linspace(-1,1,16,dtype=torch.float64),torch.linspace(-1,1,16,dtype=torch.float64),indexing='ij')
    image=(x+y>-.3).double()[None,None]
    phase=torch.exp(1j*(2*x+y**2))[None,None]
    complex_image=image*phase
    iso=TVRegularizer('isotropic');aniso=TVRegularizer('anisotropic')
    tv=dict(isotropic=float(iso.value(image)),anisotropic=float(aniso.value(image)),
            complex_tv=float(iso.value(complex_image)),magnitude_tv=float(iso.value(complex_image.abs())))
    assert tv['anisotropic']>tv['isotropic'] and tv['complex_tv']>tv['magnitude_tv']
    # A 3D ramp has exact constant physical derivatives away from terminal faces.
    z,y3,x3=torch.meshgrid(torch.arange(5,dtype=torch.float64),torch.arange(6,dtype=torch.float64),torch.arange(7,dtype=torch.float64),indexing='ij')
    volume=(2*z+3*y3+4*x3)[None,None]
    unit=GradientOperator(volume.shape[1:],axes=(-3,-2,-1))
    spaced=GradientOperator(volume.shape[1:],axes=(-3,-2,-1),spacing=(2.,3.,4.))
    derivative=spaced.forward(volume)
    assert torch.equal(derivative[:,:,:,:-1,:-1,:-1],torch.ones_like(derivative[:,:,:,:-1,:-1,:-1]))
    generator=torch.Generator().manual_seed(17)
    dual=torch.randn(derivative.shape,generator=generator,dtype=torch.float64)
    left=torch.vdot(derivative.ravel(),dual.ravel());right=torch.vdot(volume.ravel(),spaced.adjoint(dual).ravel())
    spacing=dict(unit_squared_norm=float(unit.forward(volume).square().sum()),
        spaced_squared_norm=float(derivative.square().sum()),adjoint_relative_error=float(abs(left-right)/abs(left)),spacing=[2,3,4])
    psf=torch.tensor([[.1,.2,.1],[.05,.1,.05],[.1,.2,.1]],dtype=torch.float64)
    blurred={};reconstructions={}
    for boundary in ('periodic','zero'):
        op=KnownConvolution((16,16),psf,[1,1],boundary)
        blurred[boundary]=op.forward(image[:,0])
    # Same observation, fixed prior, initialization and numerical accuracy; only H changes.
    residuals={}
    for boundary in ('periodic','zero'):
        op=KnownConvolution((16,16),psf,[1,1],boundary)
        solved=CompositeSolver('admm',regularizer='tv',strength=.01,boundary='neumann',
            num_iterations=2000,tolerance=1e-5,cg_tolerance=1e-6,cg_max_iterations=300).solve_detailed(blurred['zero'],op)
        reconstructions[boundary]=solved.reconstruction.numpy()
        residuals[boundary]=dict(status=solved.status,reason=solved.stopping_reason,iterations=solved.actual_iterations,
            primal=solved.metadata['primal_residual'],dual=solved.metadata['dual_residual'])
    boundary=dict(forward_difference=float((blurred['periodic']-blurred['zero']).norm()),
        reconstruction_difference=float(np.linalg.norm(reconstructions['periodic']-reconstructions['zero'])),
        physical=['periodic','zero'],regularizer_boundary='neumann',residuals=residuals)
    np.savez(args.output/'definitions_arrays.npz',**reconstructions,phase=complex_image.numpy())
    save(args.output/'definitions.json',dict(tv=tv,spacing=spacing,convolution=boundary,
        homoscedastic_mri=homoscedastic_equivalence(complex_image)))


if __name__=='__main__':main()
