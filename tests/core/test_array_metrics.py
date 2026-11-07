import numpy as np
import pytest
from inv_framework.evaluation.array_metrics import evaluate_array


@pytest.mark.parametrize('dtype',[np.uint8,np.uint16])
def test_integer_metrics_match_float(dtype):
    rng=np.random.default_rng(7)
    limit=np.iinfo(dtype).max
    truth=rng.integers(0,limit+1,size=(16,18),dtype=dtype)
    changed=np.roll(truth,1,axis=0)
    for reconstruction in (truth,changed):
        actual=evaluate_array(reconstruction,dict(reference=truth,data_range=limit),spatial_axes=(0,1))
        expected=evaluate_array(reconstruction.astype(float),
            dict(reference=truth.astype(float),data_range=limit),spatial_axes=(0,1))
        assert actual==pytest.approx(expected)
        if reconstruction is truth:
            assert actual['rmse']==0
            assert actual['ssim']==pytest.approx(1)
        else:
            assert actual['rmse']>0


def test_mri_magnitude_and_complex_metrics():
    truth=(np.arange(144).reshape(12,12)+1).astype(np.complex64)*(1+1j)
    reconstruction=1j*truth
    magnitude=evaluate_array(reconstruction,dict(reference=truth,data_range=204),
        spatial_axes=(0,1),mri=True)
    assert magnitude['rmse']==0
    assert magnitude['ssim']==pytest.approx(1)
    complex_metrics=evaluate_array(reconstruction,
        dict(reference=truth,data_range=204,mode='complex'),spatial_axes=(0,1),mri=True)
    assert 'ssim' not in complex_metrics
    assert complex_metrics['rmse']==pytest.approx(np.sqrt(np.mean(abs(
        reconstruction.astype(complex)-truth.astype(complex))**2)))


def test_real_metrics_reject_complex_reference():
    with pytest.raises(ValueError,match='real reference'):
        evaluate_array(np.ones((8,8)),dict(reference=np.ones((8,8),complex),data_range=1),
            spatial_axes=(0,1))
