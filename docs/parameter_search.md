# Parameter search conventions

[Algorithm pages](algorithms/README.md) read defaults and hard domains from existing YAML/CT registry and recommendations from inv_framework/algorithm_catalog.py. Regenerate with `python tools/render_algorithm_docs.py`; discovery reads those same definitions. Recommended search ranges are never enforced or clipped by execution. Validation rejects invalid mathematical or implemented-domain conditions rather than silently selecting another method.

Weight searches use default times {0.01,0.1,1,10,100,1000}, with 0.01 as the positive starting reference for zero/absent defaults. These are initial task-conditioned log grids, not universal optimal ranges. Fix data units, grid/spacing, regularizer definition, weights and noise calibration first; retain a valid zero-weight baseline separately. Do not jointly reduce fit by arbitrary noise scaling and regularizer rescaling. Choose candidates on development/held-out data, not evaluation truth. If a selected value is at an endpoint, the range may be insufficient.

ADMM rho and HQS/latent initial penalties use {0.1,1,10} times default while the objective stays fixed. Preserve final>=initial and growth>1; select by residuals and time to declared accuracy. Iteration budgets use half/default/double; tolerances, norm estimation and inner precision stay fixed for quality comparisons. Explicit step sizes require actual operator curvature; finite power iteration is an estimate, not a certified upper bound. PDHG steps are coupled; no independent unrestricted tau/sigma quality search is proposed.

Categorical choices describe model or method alternatives, only when data conditions permit. Subset partitions must match view counts; Haar depths must divide spatial sizes; calibration patches must fit ACS; bounds follow physical image units. Seeds and analytic scales are fixed/calibrated. Numerical floors prevent undefined operations and are not quality-search knobs. Parameters marked fixed have no recommended automatic quality search; they remain explicitly overridable within their legal domain.

FBP/FDK have no iterative optimization parameter; filter/integration/scale choices require geometry and acquisition justification. Zero-filled exposes only coil combination; RSS and complex SENSE have different targets. VarNet checkpoint/architecture are training conditions, not non-learning search parameters. Nonconvex blind and Hyper-Laplacian searches are local-method comparisons without a global convergence guarantee. No full parameter-search experiment is included here.

## Task-context candidate screening

Each domain's `list_capabilities(context=None)` accepts optional task facts. CT keeps its positional `data_root` argument and takes context by keyword. Calls without context retain static recommendations and explicitly list missing facts.

```python
from inv_framework.mri.api import list_capabilities
caps = list_capabilities(context={
    'model': {'spatial_shape': [4, 4], 'sampling': 'cartesian'},
    'params': {},
})
wavelet = next(r for r in caps['algorithms'] if r['name'] == 'cs_wavelet')
print(wavelet['parameter_schema']['levels']['recommended_search'])
# values=[1,2,3]; resolved_values=[1,2]; unresolved_conditions=[]
```

Context uses positive spatial-only `model.spatial_shape`, positive `num_views` for CT, spatial-only `calibration_shape` for MRI ACS, and canonical algorithm `params` to fix coupled settings. Model keys `noise_model`, `background`, `sigma_e` describe known observation facts for CT; `available_inputs` lists public field names such as sigma, weights, counts and I0. Array contents are not loaded by discovery. MRI sampling and deconvolution boundary facts are checked against the existing structured method declarations.

Each parameter's recommended_search keeps its original `values` and adds `resolved_values`, `resolution` and `unresolved_conditions`. Missing relevant facts yield resolution=pending, so the remaining values are provisional. Empty resolved_values means none of the existing suggestions pass the supplied screening; the library does not invent a new search grid. Resolution checks Haar divisibility (including the CT Haar prior choice), CT view/block/subset partition conditions, odd MRI ACS kernel widths fitting calibration_shape, HQS initial penalty against the configured final penalty, and CT public-data/noise prerequisites for data-term, weight-source and residual-metric suggestions. Coupled params use algorithm defaults when omitted.

The record-level recommendation_context separately reports unresolved acquisition fields and incompatible sampling/boundary declarations. A parameter marked resolved has passed only the listed candidate conditions. It does not certify backend availability, array values/shapes, convergence, full mathematical compatibility or that the whole task is executable. Use the existing public request validation/execution entry for those checks. Candidate screening never changes hard parameter domains, defaults, caller params or solver inputs; legal user values outside the suggested grid remain accepted. There is no search or combination enumeration in discovery.
