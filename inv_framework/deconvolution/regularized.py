"""Periodic TV ADMM, smoothed Dey RL-TV, and gradient Hyper-Laplacian HQS."""
import numpy as np
from inv_framework.core.numpy_math import gradient, gradient_adjoint, shrink, laplacian_spectrum


def poisson_prox(v, y, rho):
    a=v-1/rho
    root=np.sqrt(a*a+4*y/rho)
    # Stable positive root, including y=0.
    return np.where(a>=0,(a+root)/2,np.divide(2*y/rho,root-a,out=np.zeros_like(a),where=root-a>0))


def tv_admm(op,y,initial,p,poisson,background):
    axes=op.axes; rho=p['rho']; x=initial.copy()
    nonnegative=p.get('nonnegative')
    if nonnegative is None:nonnegative=poisson
    z=op.forward(x); d=gradient(x,axes); v=x.copy()
    uz=np.zeros_like(z); ud=np.zeros_like(d); uv=np.zeros_like(x)
    spectrum=abs(op.otf)**2+laplacian_spectrum(x.shape,axes)+1
    history=[]; reason='max_iterations'
    for iteration in range(1,p['max_iter']+1):
        rhs=op.adjoint(z-uz)+gradient_adjoint(d-ud,axes)+v-uv
        x=np.fft.ifftn(np.fft.fftn(rhs,axes=axes)/spectrum,axes=axes).real
        hx=op.forward(x); gx=gradient(x,axes)
        oldz,oldd,oldv=z,d,v
        z=poisson_prox(hx+uz+background,y,rho)-background if poisson else (y+rho*(hx+uz))/(1+rho)
        d=shrink(gx+ud,p['lam']/rho,True)
        projection_count=int(np.count_nonzero(x+uv<0)) if nonnegative else 0
        v=np.maximum(x+uv,0) if nonnegative else x+uv
        uz+=hx-z; ud+=gx-d; uv+=x-v
        primal=float(np.sqrt(np.linalg.norm(hx-z)**2+np.linalg.norm(gx-d)**2+np.linalg.norm(x-v)**2))
        dual=float(rho*np.linalg.norm(op.adjoint(z-oldz)+gradient_adjoint(d-oldd,axes)+v-oldv))
        history.append(dict(primal_residual=primal,dual_residual=dual))
        if max(primal,dual)<=p['tolerance']*max(1,np.linalg.norm(x)):
            reason='converged';break
    # v applies the selected image constraint; splitting errors remain separate.
    return v,iteration,reason,dict(residual_history=history,solver='admm',nonnegative_projection_count=projection_count,nonnegative=nonnegative)


def rl_tv(op,y,initial,p,background):
    x=initial.copy(); sensitivity=op.adjoint(np.ones_like(y)); history=[]; reason='max_iterations'
    for iteration in range(1,p['max_iter']+1):
        g=gradient(x,op.axes)
        tv_gradient=gradient_adjoint(g/np.sqrt(np.sum(g*g,axis=0)+p['tv_epsilon']**2),op.axes)
        denominator=sensitivity+p['lam']*tv_gradient
        if np.any(denominator<=0):
            return x,iteration-1,'numerical_error',dict(message='RL-TV denominator is nonpositive; multiplicative update undefined',minimum_denominator=float(denominator.min()),residual_history=history)
        mean=np.maximum(op.forward(x)+background,p['epsilon'])
        next_x=x*np.maximum(op.adjoint(y/mean),0)/denominator
        change=float(np.linalg.norm(next_x-x)/max(np.linalg.norm(x),p['epsilon']))
        x=next_x; history.append(change)
        if change<=p['tolerance']:
            reason='converged';break
    return x,iteration,reason,dict(residual_history=history,solver='dey_smoothed_multiplicative')


def hyper_scalar(v, weight, exponent):
    """Global minimizer of .5*(w-v)^2 + weight*abs(w)^p, 0<p<1."""
    if weight == 0:
        return v
    a=abs(v)
    turning=(weight*exponent*(1-exponent))**(1/(2-exponent))
    lower=np.full_like(a,turning); upper=np.maximum(a,turning)
    active=a>turning+weight*exponent*turning**(exponent-1)
    # Only the larger positive root can be a local minimum; compare it with zero.
    for _ in range(70):
        mid=(lower+upper)/2
        positive=mid+weight*exponent*mid**(exponent-1)>a
        upper=np.where(positive,mid,upper);lower=np.where(positive,lower,mid)
    root=(lower+upper)/2
    choose=active & (.5*(root-a)**2+weight*root**exponent < .5*a*a)
    return np.sign(v)*np.where(choose,root,0)


def hyper_hqs(op,y,initial,p):
    x=initial.copy(); axes=op.axes; lap=laplacian_spectrum(x.shape,axes)
    rhs=op.otf.conj()*np.fft.fftn(y,axes=axes)
    history=[]; beta=p['beta_initial']; reason='max_iterations'
    for iteration in range(1,p['max_iter']+1):
        g=gradient(x,axes); z=hyper_scalar(g,p['lam']/beta,p['p'])
        denominator=abs(op.otf)**2+beta*lap
        numerator=rhs+beta*np.fft.fftn(gradient_adjoint(z,axes),axes=axes)
        spectrum=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
        next_x=np.fft.ifftn(spectrum,axes=axes).real
        change=float(np.linalg.norm(next_x-x)/max(1,np.linalg.norm(x))); x=next_x
        history.append(dict(beta=beta,relative_change=change,splitting_residual=float(np.linalg.norm(gradient(x,axes)-z))))
        if beta>=p['beta_max'] and change<=p['tolerance']:
            reason='converged';break
        beta=min(beta*p['beta_factor'],p['beta_max'])
    return x,iteration,reason,dict(residual_history=history,solver='hqs',final_beta=history[-1]['beta'])
