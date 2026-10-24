# Constrained TV source and variant correspondence

Mathematical source: Sidky and Pan, Physics in Medicine and Biology 53 (2008),
4777–4807, DOI [10.1088/0031-9155/53/17/021](https://doi.org/10.1088/0031-9155/53/17/021),
section 2.4.1 ([PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2630711/)).
The original uses ray-wise ART, positivity and adaptive normalized TV descent.
It solves a residual-tolerance constrained TV problem. Intermediate TV steps
can leave the feasible set; subsequent data updates restore consistency.
Local feasibility-preserving backtracking changes that trajectory, and therefore
has no inherited global optimality guarantee.

Engineering reference: CERN TIGRE **v2.6**, commit
`dfa6c6522aeaa0f2576e0c8500ad7deedfc1e64f`,
[Python/tigre/algorithms/pocs_algorithms.py](https://github.com/CERN/TIGRE/blob/dfa6c6522aeaa0f2576e0c8500ad7deedfc1e64f/Python/tigre/algorithms/pocs_algorithms.py),
`initASD_POCS.__init__`, `run_main_iter`, `ASD_POCS.__init__`.
The matching upstream MATLAB `ASD_POCS.m` and `LICENSE.txt` are available at the same pinned revision. TIGRE uses BSD-3-Clause, copyright University of Bath and CERN (2015).
Production code uses the existing local PyTorch operators; TIGRE is not a
runtime dependency and no TIGRE CUDA kernels are incorporated.

| Update | TIGRE v2.6 | Local `sart2d_feasibility_backtracking_v1` |
| --- | --- | --- |
| Data sweep | SART, ASD forces blocksize=1 | ordered angle blocks, reciprocal row/column normalization, nonnegativity after each block; block_size>1 is a block-SART variant |
| Relaxation | beta from lmbda, multiplied by lmbda_red | relaxation=1, relaxation_decay=.99 per complete sweep |
| Initial TV step | alpha times first data displacement, alpha=.002 | tv_step=.2 times max(first displacement, tolerance times image norm); explicit feasible-start fallback |
| TV iterations | tviter=20, minimizeTV | tv_updates=20 normalized local TV subgradient steps, each projected nonnegative |
| Adaptation | reduce dtvg by alpha_red=.95 when TV displacement exceeds rmax=.95 times data displacement and residual exceeds epsilon | bounded 40-trial backtracking by tv_step_decay=.95; enforce displacement ratio update_ratio=.95, TV nonincrease, and preserve feasibility once the data point is feasible |
| Radius | explicit maxl2err or FDK-derived default | mandatory epsilon and epsilon_source in raw line-integral units; no FDK calibration |
| Stop | displacement angle, beta<.005, or niter | resolved SolveControl min/check/patience; feasible iterate stagnation is stalled, never global convergence; sweep and operator limits remain resource outcomes |

Local unit-grid TV uses forward differences with zero terminal differences,
with explicit isotropic/anisotropic modes. The implementation is a two-dimensional
SART-based ASD-POCS adaptation, not a numerically identical port of TIGRE or the
original ART algorithm. The variant identity and absence of a global optimality guarantee are preserved. Parameters above are engineering initializations, not quality-tuned defaults.

PDHG independently solves the convex problem using K=[A;D], residual-ball
conjugate proximal, TV dual-ball projection and nonnegative primal projection.
The Schur norm bound of A plus 8 bounds ||K||²; tau=sigma=.99/sqrt(bound)
satisfies tau*sigma*||K||²<1. Convergence requires feasible returned x and the
joint primal/dual fixed-point residual below resolved tolerance over the
requested checked patience window. This certificate is distinct from ASD's
feasible stagnation, and epsilon is distinct from optimization tolerance.

## TIGRE reference license

BSD 3-Clause License

Copyright (c) 2015, University of Bath and CERN- European Organization for  Nuclear Research
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
