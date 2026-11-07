"""Explicit preparation of the small Levin kernel source; no runtime downloads."""
import io
from pathlib import Path
import urllib.request
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def fetch():
    destination=ROOT/'tests/data/deconvolution/sources'
    destination.mkdir(parents=True,exist_ok=True)
    base='https://raw.githubusercontent.com/cszn/DPIR/master/'
    with urllib.request.urlopen(base+'kernels/Levin09.mat',timeout=30) as response:
        raw=response.read()
    with h5py.File(io.BytesIO(raw)) as source:
        # MATLAB cell data is transposed in HDF5. No resize, rotation or recentering.
        kernels={f'kernel_{i+1}':np.asarray(source[ref]).T for i,ref in enumerate(source['kernels'][:].ravel())}
    np.savez_compressed(destination/'levin_kernels.npz',**kernels)
    urllib.request.urlretrieve(base+'LICENSE',destination/'DPIR_LICENSE.txt')
    print('Saved eight Levin09 kernels and DPIR license',len(raw),'source bytes')


if __name__=='__main__':fetch()
