"""Resident Torch physical operators for the shared Cartesian/known-PSF path."""
import torch
from inv_framework.core.operator import LinearOperator


class CartesianSense(LinearOperator):
    def __init__(self, maps, mask):
        self.maps, self.mask = maps, mask
        self.domain_shape = tuple(maps.shape[2:])
        self.range_shape = tuple(maps.shape[1:])

    def fourier(self, x, inverse=False):
        transform = torch.fft.ifftn if inverse else torch.fft.fftn
        return torch.fft.fftshift(transform(torch.fft.ifftshift(x,dim=(-2,-1)),
            dim=(-2,-1),norm='ortho'),dim=(-2,-1))

    def forward(self,x):
        return self.mask*self.fourier(self.maps*x[:,None])

    def adjoint(self,y):
        return (self.maps.conj()*self.fourier(self.mask*y,True)).sum(1)

    def squared_norm_upper_bound(self,x):
        return float(self.maps.abs().square().sum(1).max())


class KnownConvolution(LinearOperator):
    def __init__(self, shape, psf, origin, boundary):
        self.domain_shape = self.range_shape = tuple(shape)
        self.psf,self.origin,self.boundary=psf,origin,boundary
        self.nonnegative_matrix=bool((psf>=0).all())
        if boundary=='periodic':
            kernel=torch.zeros(shape,dtype=psf.dtype,device=psf.device)
            for i in range(psf.shape[0]):
                for j in range(psf.shape[1]):
                    kernel[(i-origin[0])%shape[0],(j-origin[1])%shape[1]]+=psf[i,j]
            self.otf=torch.fft.fftn(kernel)

    def _apply(self,x,adjoint):
        if self.boundary=='periodic':
            h=self.otf.conj() if adjoint else self.otf
            return torch.fft.ifftn(torch.fft.fftn(x,dim=(-2,-1))*h,dim=(-2,-1)).real
        out=torch.zeros_like(x)
        for i in range(self.psf.shape[0]):
            for j in range(self.psf.shape[1]):
                source=[slice(None)]*x.ndim;target=source.copy()
                for a,d in zip((-2,-1),(i-self.origin[0],j-self.origin[1])):
                    target[a]=slice(max(d,0),min(x.shape[a],x.shape[a]+d))
                    source[a]=slice(max(-d,0),min(x.shape[a],x.shape[a]-d))
                if adjoint:source,target=target,source
                out[tuple(target)]+=self.psf[i,j]*x[tuple(source)]
        return out

    def forward(self,x):return self._apply(x,False)
    def adjoint(self,y):return self._apply(y,True)
    def squared_norm_upper_bound(self,x):
        return float(self.otf.abs().square().max()) if self.boundary=='periodic' else float(self.psf.abs().sum())**2
