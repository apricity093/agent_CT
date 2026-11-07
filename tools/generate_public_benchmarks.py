"""Prepare lightweight public-image and standard MR-phantom benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
from scipy.special import j1
from skimage import data, color, transform
from phantominator import mr_shepp_logan
ROOT = Path(__file__).resolve().parents[1]


def fft(a):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))


def direct_dft(coils, trajectory):
    h,w=coils.shape[-2:]
    ey=np.exp(-1j*trajectory[:,0,None]*(np.arange(h)-h//2))
    ex=np.exp(-1j*trajectory[:,1,None]*(np.arange(w)-w//2))
    return np.einsum('sy,cyx,sx->cs',ey,coils,ex,optimize=True)/np.sqrt(h*w)


def save_case(root, case_id, arrays, metadata, records):
    folder=root/'cases'/case_id
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'public.npz',**arrays)
    (folder/'case.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    records.append(dict(case_id=case_id,path=f'cases/{case_id}',**metadata))
    return folder


def loop_maps(n, coils):
    """Biot-Savart midpoint sum for circular receive loops, unit RSS Bx+iBy."""
    y,x=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n),indexing='ij')
    points=np.stack([x,y,np.zeros_like(x)],axis=-1)
    maps=[]
    for angle in np.arange(coils)*2*np.pi/coils:
        radial=np.array([np.cos(angle),np.sin(angle),0])
        tangent=np.array([-np.sin(angle),np.cos(angle),0])
        vertical=np.array([0,0,1])
        field=np.zeros_like(points)
        for phi in (np.arange(64)+.5)*2*np.pi/64:
            position=1.6*radial+.55*(np.cos(phi)*tangent+np.sin(phi)*vertical)
            dl=.55*2*np.pi/64*(-np.sin(phi)*tangent+np.cos(phi)*vertical)
            displacement=points-position
            field+=np.cross(dl,displacement)/np.linalg.norm(displacement,axis=-1)[...,None]**3
        maps.append(field[...,0]+1j*field[...,1])
    maps=np.asarray(maps)
    return maps/np.sqrt(np.sum(abs(maps)**2,axis=0))


def airy_kernel(size=15, first_zero=3.):
    y,x=np.mgrid[:size,:size]-(size//2)
    radius=3.8317059702075125*np.hypot(x,y)/first_zero
    amplitude=np.ones_like(radius)
    np.divide(2*j1(radius),radius,out=amplitude,where=radius!=0)
    kernel=amplitude**2
    return kernel/kernel.sum()


def generate(data_root):
    root=Path(data_root)
    with np.load(root/'deconvolution/sources/levin_kernels.npz') as source:
        kernels=[source[f'kernel_{i+1}'] for i in range(8)]
    images={'camera':data.camera(),'astronaut':color.rgb2gray(data.astronaut()),'cell':data.cell()}
    for domain in ('mri','deconvolution'):
        catalog_path=root/domain/'catalog.json'
        catalog=json.loads(catalog_path.read_text(encoding='utf8'))
        records=[]
        if domain=='deconvolution':
            settings=[('public_camera_levin','camera','dev',kernels[0],'Levin09 kernel 1',.005),
                      ('public_astronaut_levin','astronaut','eval',kernels[1],'Levin09 kernel 2',.005),
                      ('public_cell_airy','cell','eval',airy_kernel(),'ideal Airy first zero=3 pixels',.005)]
            for index,(cid,name,split,kernel,kname,sigma) in enumerate(settings):
                raw=images[name]
                image=transform.resize(raw,(256,256),anti_aliasing=True,preserve_range=True)
                if raw.dtype==np.uint8:image/=255
                seed=12000+index;rng=np.random.default_rng(seed)
                clean=convolve2d(image,kernel,mode='same',boundary='wrap')
                clean=np.maximum(clean,0)
                arrays=dict(gaussian=clean+rng.normal(0,sigma,clean.shape),counts=rng.poisson(40*clean+.2),psf=kernel)
                meta=dict(split=split,seed=seed,benchmark_group='public_standard',spatial_shape=[256,256],data_range=1.,
                          source=f'scikit-image 0.26.0 data.{name}',source_url=f'https://scikit-image.org/docs/0.26.x/api/skimage.data.html#skimage.data.{name}',
                          image_rights='CC0' if name!='astronaut' else 'NASA public domain; no known copyright restrictions',
                          object_kind=name,kernel=kname,kernel_shape=list(kernel.shape),psf_origin=[s//2 for s in kernel.shape],
                          boundary='periodic',gaussian_sigma=sigma,photon_gain=40.,background=.2,
                          observation='Synthetic periodic blur and Gaussian/Poisson noise applied to a public image; not an acquired blurred pair',
                          image_conversion='anti-aliased resize to 256 square; astronaut RGB to standard luminance; uint8 divided by 255',
                          interpretation='Cell is quantitative phase, used only as a nonnegative intensity surrogate; not measured fluorescence',
                          public_inputs=list(arrays),reference='reference.npz: object and true_psf; blind tasks never load PSF')
                folder=save_case(root/domain,cid,arrays,meta,records)
                np.savez_compressed(folder/'reference.npz',object=image,true_psf=kernel)
        else:
            settings=[('standard_mr_dev',128,'dev',-.25,.02,1.0,'cartesian'),
                      ('standard_mr_cartesian',256,'eval',-.15,.06,2.0,'cartesian'),
                      ('standard_mr_radial',128,'eval',-.15,.06,2.0,'noncartesian')]
            for index,(cid,n,split,z,te,tr,sampling) in enumerate(settings):
                density,t1,t2=mr_shepp_logan((n,n,1),B0=3.,zlims=(z,z))
                density,t1,t2=(a[...,0] for a in (density,t1,t2))
                magnitude=np.zeros((n,n));support=(density>0)&(t1>0)&(t2>0)
                magnitude[support]=density[support]*(1-np.exp(-tr/t1[support]))*np.exp(-te/t2[support])
                # Fixed physical signal scale, never fitted to reconstruction or truth.
                yy,xx=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n),indexing='ij')
                truth=magnitude*np.exp(.2j*np.sin(2*xx)*np.cos(yy))
                maps=loop_maps(n,4);seed=13000+index;rng=np.random.default_rng(seed)
                model=dict(axes=['batch','coil','y','x'],spatial_shape=[n,n],sampling=sampling,fft_centered=True,fft_norm='ortho')
                arrays=dict(sensitivity_maps=maps[None])
                if sampling=='cartesian':
                    low=32 if n==128 else 48;start=(n-low+1)//2
                    columns=np.zeros(n,bool);columns[::4]=True;columns[start:start+low]=True
                    mask=np.broadcast_to(columns,(1,4,n,n)).copy()
                    clean=fft(maps*truth)[None]
                    noisy=clean+.002/np.sqrt(2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
                    observed=noisy*mask
                    arrays.update(sampling_mask=mask,calibration_data=observed[...,start:start+low])
                else:
                    theta=np.arange(32)*np.pi/32;r=np.linspace(-np.pi,np.pi,160,endpoint=False)
                    trajectory=np.stack([(np.sin(theta)[:,None]*r).ravel(),(np.cos(theta)[:,None]*r).ravel()],axis=1)
                    clean=direct_dft(maps*truth,trajectory)[None]
                    observed=clean+.002/np.sqrt(2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
                    arrays['trajectory']=trajectory;model['trajectory_units']='radians/voxel'
                arrays['kspace']=observed.astype(np.complex64)
                # Calibration is sliced from exactly the stored acquired array.
                if sampling=='cartesian':arrays['calibration_data']=arrays['kspace'][...,start:start+low]
                meta=dict(split=split,seed=seed,benchmark_group='public_standard',object_kind='standard_mr_shepp_logan',spatial_shape=[n,n],data_range=1.,
                          source='phantominator 0.7.0 MR Shepp-Logan; Gach et al. 2008',source_url='https://github.com/mckib2/phantominator',image_rights='MIT generator',
                          model=model,coil_count=4,complex_noise_sigma=.002,source_z=z,TR_sec=tr,TE_sec=te,B0_tesla=3.,
                          signal_model='M0*(1-exp(-TR/T1))*exp(-TE/T2), spoiled spin-echo-like contrast without scanner simulation',
                          maps_model='Biot-Savart sum over 64 segments of 4 circular loops; ring radius1.6, loop radius.55, FOV[-1,1]^2, Bx+iBy, unit RSS',
                          interpretation='Standard analytic MR phantom with simulated receive encoding; not BrainWeb or real knee data',
                          public_inputs=list(arrays),reference='reference.npz: complex object, M0, T1, T2')
                if sampling=='cartesian':meta.update(acs_columns=[start,start+low],num_low_frequencies=low,acs_source='same public noisy acquisition')
                folder=save_case(root/domain,cid,arrays,meta,records)
                np.savez_compressed(folder/'reference.npz',object=truth[None].astype(np.complex64),M0=density,T1=t1,T2=t2)
        catalog.update(primary_benchmark_group='public_standard',cases=records)
        catalog_path.write_text(json.dumps(catalog,indent=2),encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'tests/data')
    generate(parser.parse_args().data_root)
