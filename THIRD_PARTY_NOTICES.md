# fastMRI network components

inv_framework/mri/models/varnet.py and unet.py are adapted from
facebookresearch/fastMRI (archived main, retrieved 2026-11-06).
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# Public benchmark data

The public_standard benchmark uses scikit-image 0.26.0 sample images:
camera by Lav Varshney (CC0), astronaut from NASA (public domain, no known
copyright restrictions), and cell by the quantitative phase imaging contributors
listed in skimage.data.cell (CC0). Image conversions and source URLs are retained
in each case.json and docs/public_benchmarks.md. Cell is not fluorescence data.

Levin09 kernels are attributed to Anat Levin, Yair Weiss, Fredo Durand and
William T. Freeman, CVPR 2009, obtained from cszn/DPIR/kernels/Levin09.mat.
The DPIR repository MIT notice, Copyright (c) 2020 Kai Zhang, is preserved in
tests/data/deconvolution/sources/DPIR_LICENSE.txt. Kernels retain original support;
no separate original-author data license is asserted by this notice.

The MR tissue phantom is generated with phantominator 0.7.0 (MIT), Nicholas
McKibben, implementing Gach, Tanase and Boada's 2008 MR Shepp-Logan standard.
This dependency is used for data preparation, not the reconstruction runtime.
No real fastMRI patient data is redistributed. fastMRI source-code licensing
does not replace the dataset use agreement.
