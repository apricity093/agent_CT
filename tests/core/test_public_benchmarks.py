"""Public source provenance and model-matched acquisition checks."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import convolve2d

ROOT=Path(__file__).resolve().parents[2]


def cases(domain):
    root=ROOT/'tests/data'/domain
    catalog=json.loads((root/'catalog.json').read_text())
    assert catalog['primary_benchmark_group']=='public_standard'
    assert len(catalog['cases'])==3
    assert all(c.get('benchmark_group')=='public_standard' for c in catalog['cases'])
    assert {p.name for p in (root/'cases').iterdir()}=={Path(c['path']).name for c in catalog['cases']}
    return root,[c for c in catalog['cases'] if c.get('benchmark_group')=='public_standard']


def test_public_deconvolution_observation_and_kernel_origin():
    root,records=cases('deconvolution')
    assert {c['object_kind'] for c in records}=={'camera','astronaut','cell'}
    source=np.load(root/'sources/levin_kernels.npz')
    assert len(source.files)==8
    for case in records:
        public=np.load(root/case['path']/'public.npz');reference=np.load(root/case['path']/'reference.npz')
        k=public['psf'];np.testing.assert_allclose(k.sum(),1,atol=1e-12)
        assert k.min()>=0 and list(k.shape)==case['kernel_shape']
        assert case['psf_origin']==[n//2 for n in k.shape]
        clean=convolve2d(reference['object'],k,mode='same',boundary='wrap')
        noise=np.random.default_rng(case['seed']).normal(0,case['gaussian_sigma'],clean.shape)
        np.testing.assert_allclose(public['gaussian'],clean+noise,atol=1e-13)
        assert public['counts'].min()>=0


def test_standard_mr_acquired_acs_and_fft():
    root,records=cases('mri')
    assert len(records)==3
    for case in records:
        public=np.load(root/case['path']/'public.npz');reference=np.load(root/case['path']/'reference.npz')['object']
        maps=public['sensitivity_maps']
        np.testing.assert_allclose(np.sum(abs(maps)**2,axis=1),1,atol=1e-12)
        assert np.isfinite(public['kspace']).all()
        if case['model']['sampling']=='cartesian':
            lo,hi=case['acs_columns'];mask=public['sampling_mask']
            assert mask[...,lo:hi].all()
            np.testing.assert_array_equal(public['calibration_data'],public['kspace'][...,lo:hi])
            coils=maps*reference
            clean=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(coils,axes=(-2,-1)),norm='ortho'),axes=(-2,-1))
            rng=np.random.default_rng(case['seed'])
            noise=.002/np.sqrt(2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
            np.testing.assert_allclose(public['kspace'],(clean+noise)*mask,atol=2e-6,rtol=1e-6)


def test_local_slice_import_preserves_native_grid_and_reference(tmp_path):
    import h5py
    import runpy
    import yaml
    prepare=runpy.run_path(str(ROOT/'tools/prepare_fastmri_slice.py'))['prepare']
    rng=np.random.default_rng(811)
    # Random test fixture checks file conventions only, never MRI quality.
    raw=(rng.normal(size=(2,4,64,80))+1j*rng.normal(size=(2,4,64,80))).astype(np.complex64)
    source=tmp_path/'mock.h5'
    with h5py.File(source,'w') as handle:handle['kspace']=raw
    prepare(source,1,tmp_path/'slice',tmp_path/'weights.pt')
    public=np.load(tmp_path/'slice/public.npz')
    assert public['kspace'].shape==(1,4,64,80)
    np.testing.assert_array_equal(public['kspace'],raw[1:2]*public['sampling_mask'])
    np.testing.assert_array_equal(public['calibration_data'],raw[1:2,...,28:52])
    coils=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(raw[1:2],axes=(-2,-1)),norm='ortho'),axes=(-2,-1))
    expected=np.sqrt(np.sum(abs(coils)**2,axis=1))
    np.testing.assert_allclose(np.load(tmp_path/'slice/reference.npy'),expected,rtol=1e-6)
    config=yaml.safe_load((tmp_path/'slice/varnet.yaml').read_text())
    assert set(config['inputs'])=={'kspace','sampling_mask'}
    assert config['model']['spatial_shape']==[64,80]
