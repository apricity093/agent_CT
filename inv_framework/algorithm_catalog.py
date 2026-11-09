"""Public algorithm descriptions shared by discovery and documentation."""

DOCUMENTATION_ALIASES = {
    'tv_fista': 'fista', 'constrained_pdhg': 'pdhg',
    'tikhonov': 'cg', 'cg_sense': 'cg',
    'cs_tv': 'admm', 'cs_wavelet': 'admm', 'tv_admm': 'admm',
    'os_sart': 'sart', 'osem': 'mlem', 'rl_tv': 'richardson_lucy',
}


def algorithm_documentation(name):
    """One method page, with domain adapters and prior choices as sections."""
    return f"docs/algorithms/{DOCUMENTATION_ALIASES.get(name, name)}.md"

# These are execution contracts, not the later cross-domain adaptation plan.
OBJECTIVES = {
    'zero_filled': 'Centered orthonormal inverse FFT and SENSE combination or magnitude RSS; no optimization objective.',
    'cg_sense': '0.5 sum W|PFSx-y|^2 + 0.5 lam ||x||^2, complex Hermitian CG.',
    'cs_tv': '0.5 sum W|PFSx-y|^2 + lam sum ||Dx||_2; complex periodic isotropic TV, ADMM with inner CG.',
    'cs_wavelet': '0.5 sum W|PFSx-y|^2 + lam ||Haar(x)||_1; complex coefficients, all bands penalized, ADMM with inner CG.',
    'grappa': 'Local ACS ridge least squares, interpolate only missing k-space; magnitude RSS output.',
    'espirit': 'ACS patch SVD and image-domain calibration eigendecomposition; returns one sensitivity map set.',
    'varnet': 'Checkpoint-defined learned cascades with data consistency; no selectable analytic likelihood.',
    'wiener_hunt': '0.5 ||Hx-y||^2 + 0.5 balance ||Laplacian(x)||^2, periodic FFT direct solution.',
    'richardson_lucy': 'sum (gain Hx+b - y log(gain Hx+b)), x>=0; emission Poisson multiplicative update.',
    'tv_admm': 'D(Hx+b,y) + lam sum ||Dx||_2; periodic FFT ADMM; Poisson also enforces x>=0.',
    'rl_tv': 'Emission Poisson NLL + lam sum sqrt(||Dx||_2^2+tv_epsilon^2); Dey multiplicative update requires positive denominator.',
    'hyper_laplacian': '0.5 ||Hx-y||^2 + lam sum |Dx|^p; nonconvex finite-penalty HQS, not exact global minimization.',
    'blind_dcp': 'Alternating dark-channel/gradient sparse latent estimation and constrained kernel ridge fitting; final p=0.5 Hyper-Laplacian recovery.',
    'blind_l0': 'Alternating near-L0 gradient latent estimation and constrained kernel ridge fitting; final p=0.5 Hyper-Laplacian recovery.',
}

CT_OBJECTIVES = {
 'fbp': 'Ramp-filtered analytic backprojection, no iterative objective.',
 'fdk': 'Cone-beam FDK filtered backprojection on ASTRA CUDA; approximate analytic inversion.',
 'sirt': 'Diagonally normalized simultaneous residual iteration for Ax=y; optional box projection, not an arbitrary likelihood solver.',
 'sart': 'Block row/column normalized residual updates for Ax=y; ordered or random angle schedule, optional box projection.',
 'os_sart': 'Balanced ordered-subset SART residual updates; subset_count controls actual angle partition.',
 'landweber': 'Unconstrained augmented weighted quadratic gradient iteration; legacy box path is projected unweighted L2 gradient.',
 'cg': 'Independent Hermitian CG: 0.5 sum Q|Ax-y|^2 + 0.5 strength ||x||^2; requires strength>0.',
 'cgls': '0.5 sum W|Ax-y|^2 + 0.5 strength ||Kx||^2, K=I or gradient; augmented weighted path is unconstrained.',
 'lsqr': '0.5 sum W|Ax-y|^2 + 0.5 strength ||Kx||^2 or 0.5 damping^2 ||x||^2; no simultaneous explicit augmentation and damping.',
 'tikhonov': '0.5 sum W|Ax-y|^2 + 0.5 reg_strength ||Kx||^2, K=I or gradient; CG normal system, unconstrained.',
 'tv_fista': '0.5 ||Ax-y||^2 + reg_strength TV(x), optional box; specialized FISTA with inner TV prox.',
 'fista': 'D(Ax,y)+strength R(x), optional box; accelerated proximal gradient with smooth H1 assignment and joint box prox where required.',
 'admm': 'D(Ax,y)+strength R(x), optional box; fixed-block ADMM with data/regularizer/identity splits; TV+Haar uses strength and haar_strength.',
 'pdhg': 'D(Ax,y)+strength R(x), optional box; primal-dual data/regularizer updates and norm-based steps.',
 'mlem': 'Emission Poisson NLL sum(Ax-y log Ax); multiplicative update with sensitivity normalization, not transmission NLL.',
 'osem': 'Ordered-subset emission Poisson multiplicative updates; no full-data monotonicity guarantee.',
 'constrained_pdhg': 'min TV(x) subject to ||W(Ax-y)||<=epsilon, x>=0; raw or known-sigma-whitened residual ball.',
 'asd_pocs_tv': 'Nonnegative residual-ball feasibility and TV descent heuristic; feasible stagnation is not a global optimum certificate.',
}

DESCRIPTIONS = {
 'lam': ('Regularizer weight on the stated unnormalized sum', 'objective/regularizer'),
 'strength': ('Regularizer weight on the stated unnormalized sum', 'objective/regularizer'),
 'reg_strength': ('Regularizer weight; alias strength on composable CT solvers', 'objective/regularizer'),
 'balance': ('Squared Laplacian weight in Wiener-Hunt denominator', 'objective/regularizer'),
 'ridge': ('Relative ACS ridge weight scaled by ||S*S||_F/ncols', 'dimensionless'),
 'rho': ('Fixed ADMM quadratic splitting penalty, coupled to data and regularizer scaling', 'objective/split-variable^2'),
 'kernel_width': ('Odd ACS patch width within acquired calibration grid', 'samples'),
 'svd_threshold': ('Singular-value cutoff relative to largest ACS singular value; 0<value<1', 'dimensionless'),
 'crop': ('Retain dominant calibration eigenvectors above cutoff; 0<=value<=1', 'dimensionless'),
 'levels': ('Haar depth; each spatial size divisible by 2**levels', 'levels'),
 'level': ('Haar depth; each spatial size divisible by 2**level', 'levels'),
 'p': ('Anisotropic gradient exponent; 0<p<1', 'dimensionless'),
 'scale_factor': ('Image pyramid ratio; 0<value<1', 'dimensionless'),
 'coarse_kernel_size': ('Coarsest PSF support target, at least 3', 'pixels'),
 'kernel_ridge': ('Ridge penalty for compact nonnegative kernel fitting', 'objective/kernel^2'),
 'dark_weight': ('Dark-channel sparse latent penalty', 'objective/penalty'),
 'gradient_weight': ('Gradient sparse latent penalty', 'objective/penalty'),
 'patch_size': ('Odd dark-channel spatial patch width', 'pixels'),
 'epsilon_initial': ('Initial near-L0 latent gradient threshold', 'image intensity'),
 'epsilon_levels': ('Number of halved near-L0 thresholds; inner count grows as 2**level', 'levels'),
 'penalty_initial': ('Initial dark/gradient latent splitting penalty', 'objective/variable^2'),
 'penalty_factor': ('Latent penalty growth, strictly greater than 1', 'dimensionless'),
 'beta_initial': ('Initial HQS splitting penalty', 'objective/gradient^2'),
 'beta_max': ('Final finite HQS penalty, >=beta_initial', 'objective/gradient^2'),
 'beta_factor': ('HQS penalty growth, strictly greater than 1', 'dimensionless'),
 'final_lam': ('Final p=0.5 Hyper-Laplacian recovery weight', 'objective/regularizer'),
 'epsilon': ('Residual-ball radius for constrained CT; Poisson division floor for RL', 'residual norm or counts, method-specific'),
 'eps': ('Positive numerical division floor', 'denominator units'),
 'tv_epsilon': ('Smoothed TV gradient magnitude floor', 'image intensity/pixel'),
 'initial_value': ('Positive constant object initialization; ignored with explicit initial_estimate', 'image intensity'),
 'step_size': ('Gradient step; automatic norm estimate when absent, coupled to actual curvature', 'inverse curvature'),
 'tau': ('PDHG primal step; coupled to sigma and actual operator norm', 'primal step'),
 'sigma': ('PDHG dual step; coupled to tau and actual operator norm', 'dual step'),
 'delta': ('Huber transition in gradient magnitude', 'image intensity/pixel'),
 'damping': ('LSQR damping: contributes damping squared times ||x||^2/2', 'square root curvature'),
 'relaxation': ('Row-action relaxation; SART requires 0<value<=1', 'dimensionless'),
 'scale': ('FBP analytic scale; None uses geometry-derived scale', 'geometry normalization'),
 'min_value': ('Optional lower image constraint, coupled to upper bound', 'image intensity'),
 'max_value': ('Optional upper image constraint, coupled to lower bound', 'image intensity'),
 'block_size': ('Views per row-action block; coupled to subset_count and view count', 'views'),
 'subset_count': ('Balanced angle subsets; <=views, coupled to block_size', 'subsets'),
 'seed': ('Random view-order seed, fixed for paired comparisons', 'integer seed'),
 'power_iterations': ('Operator-norm estimation iterations; estimate is not a proven upper bound', 'iterations'),
 'tv_step': ('ASD-POCS TV descent scale', 'relative step'),
 'relaxation_decay': ('ASD-POCS multiplicative row-action relaxation decay', 'dimensionless'),
 'tv_step_decay': ('ASD-POCS multiplicative TV descent decay', 'dimensionless'),
 'update_ratio': ('ASD-POCS TV versus feasibility update norm ratio', 'dimensionless'),
 'voxel_supersampling': ('ASTRA FDK subvoxel integration factor', 'samples/axis'),
}

DESCRIPTIONS.update({
 'data_fidelity': ('Optimization data term; public sigma, weights or counts required by selected term', 'enum'),
 'weight_source': ('Origin of public WLS weights; provided, known variance or matched post-log count approximation', 'enum'),
 'regularizer': ('Prior selector; ADMM tv_haar is a joint objective with separate weights', 'enum'),
 'epsilon_source': ('Required explicit provenance of residual-ball radius, not automatic truth tuning', 'text'),
 'residual_metric': ('Raw masked residual or Gaussian-whitened residual with public sigma', 'enum'),
 'tv_mode': ('Gradient group norm: isotropic or anisotropic where registered', 'enum'),
 'order_strategy': ('Ordered or randomized angle traversal, coupled to seed', 'enum'),
 'filter_type': ('ASTRA FDK reconstruction filter', 'enum'),
 'short_scan': ('ASTRA Parker short-scan weighting; requires suitable acquisition coverage', 'boolean'),
 'coil_combination': ('Complex sensitivity-normalized SENSE or real magnitude RSS', 'enum'),
 'checkpoint': ('Explicit local trained weights matching declared VarNet architecture', 'path'),
 'num_cascades': ('VarNet cascades; must match checkpoint', 'cascades'),
 'chans': ('VarNet reconstruction U-Net width; must match checkpoint', 'channels'),
 'pools': ('VarNet reconstruction U-Net depth; must match checkpoint/grid', 'levels'),
 'sens_chans': ('Sensitivity U-Net width; must match checkpoint', 'channels'),
 'sens_pools': ('Sensitivity U-Net depth; must match checkpoint/grid', 'levels'),
 'num_low_frequencies': ('Acquired central ACS width used for sensitivity estimation', 'k-space columns'),
})


def parameter_description(name, default, rule):
    """Supplement existing type/domain rules with units and conditional search advice."""
    description, units = DESCRIPTIONS.get(name, (rule.get('description') or name.replace('_', ' '), 'dimensionless'))
    if 'iter' in name or name == 'tv_updates':
        description, units = rule.get('description') or 'Iteration budget; actual stop may be earlier', 'iterations'
    if 'tol' in name or name in {'atol','btol'}:
        description, units = rule.get('description') or 'Relative numerical stopping accuracy', 'relative residual'
    search = {'sampling': 'fixed', 'values': [default], 'condition': 'Fix acquisition, calibration, initialization and numerical accuracy before comparing model weights.'}
    if name in {'lam','strength','reg_strength','balance','final_lam','dark_weight','gradient_weight','kernel_ridge','ridge'}:
        base = default if default is not None and default > 0 else .01
        search = dict(sampling='log_grid', values=[base*v for v in (.01,.1,1,10,100,1000)], condition='Same data units, weights, discretization and regularizer; include zero only for an admissible unregularized baseline. Noise scale and regularizer weight are coupled.')
    elif name in {'rho','beta_initial','penalty_initial'}:
        search = dict(sampling='log_grid', values=[default*v for v in (.1,1,10)], condition='Fix objective; select by residual or time to accuracy, not raw fit; retain positive penalties and schedule coupling.')
    elif name in {'max_iter','num_iterations','final_max_iter'}:
        search = dict(sampling='integer_grid', values=sorted(set([max(1,default//2),default,default*2])), condition='Budget experiment; compare actual stopping and residuals, not an assumed iteration count.')
    elif rule.get('choices'):
        search = dict(sampling='categorical', values=list(rule['choices']), condition='Only compare physically and mathematically compatible choices; names do not imply cross-domain support.')
    elif name in {'levels','level'}:
        search = dict(sampling='integer_grid',values=[1,2,3],condition='Only depths dividing all spatial sizes by 2**depth; fixed low-frequency coefficient convention.')
    elif name=='p':
        search=dict(sampling='categorical',values=[.5,.8],condition='Changes nonconvex prior; compare at separately selected lam, no convex convergence claim.')
    elif name=='step_size':
        search=dict(sampling='categorical',values=[None],condition='Auto: 1 / augmented squared norm upper bound. Explicit alpha>0 requires alpha*bound<2; no clipping.')
    elif name=='relaxation':
        search=dict(sampling='linear_grid',values=[.25,.5,1.],condition='SART/OS-SART 0<relaxation<=1; fix subset partition.')
    return dict(description=description, units=units, recommended_search=search)


def fidelity_contract(domain, name, parameter_names=()):
    from inv_framework.experiments import COMPOSABLE_METHODS as METHODS
    if domain != 'ct' and name in METHODS:
        return dict(mode='selectable', default='l2', supported=['l2','gaussian_nll','wls'], reason='Quadratic 0.5 sum W |Ax-y|^2; real inverse-variance weights, identity if absent. WLS requires explicit weights. Shared MRI: 2D Cartesian SENSE; deconvolution: known PSF, gain=1, background=0.')
    if domain=='ct':
        if name in {'fbp','fdk','constrained_pdhg','asd_pocs_tv'}:
            return dict(mode='not_applicable',default=None,supported=[],reason='Analytic inversion or residual-ball feasibility, not an optimization likelihood selector.')
        if name in {'mlem','osem'}:
            return dict(mode='fixed',default='emission_poisson_nll',supported=['emission_poisson_nll'],reason='Emission multiplicative update; transmission counts/log projections are incompatible.')
        choices=['l2','gaussian_nll','wls'] if 'data_fidelity' in parameter_names else ['l2']
        if name in {'fista','admm','pdhg'}: choices+=['transmission_poisson_nll']
        return dict(mode='selectable' if len(choices)>1 else 'fixed',default='l2',supported=choices,reason='Only registered updates; WLS requires public weights, Gaussian NLL public sigma, transmission NLL public counts/I0 and nonnegative constraints.')
    if domain=='mri':
        choices=['l2','gaussian_nll','wls'] if name in {'cg_sense','cs_tv','cs_wavelet'} else []
        return dict(mode='selectable' if choices else 'not_applicable',default='gaussian_nll' if choices else None,supported=choices,reason='Complex weighted quadratic only; Gaussian uses supplied inverse-variance data_weights or identity; WLS requires explicit weights. Calibration, interpolation and learned inference have no selectable reconstruction data term.')
    choices=['l2','gaussian_nll','poisson_nll'] if name=='tv_admm' else ['poisson_nll'] if name in {'richardson_lucy','rl_tv'} else ['l2','gaussian_nll']
    return dict(mode='selectable' if name=='tv_admm' else 'fixed',default='poisson_nll' if name in {'richardson_lucy','rl_tv'} else 'l2',supported=choices,reason='RL multiplicative updates are Poisson only; Gaussian FFT/HQS/blind updates are quadratic only. L2 and unit-variance Gaussian are equivalent; arbitrary WLS not implemented.')


def describe_algorithm(domain, record):
    name=record['name']
    rules=record.get('parameter_schema',{})
    for key,rule in rules.items():
        rule.update(parameter_description(key,record['parameters'][key],rule))
    return dict(canonical_id=name, entry_point=f'inv_framework.{domain}.api.run_{domain if domain != "deconvolution" else "deconvolution"}_task',
        objective=record.get("objective", OBJECTIVES.get(name)),fidelity=fidelity_contract(domain,name),
        documentation=algorithm_documentation(name),
        verification='Implementation exists; validation is environment and input dependent. See tests and run results; implemented does not mean converged.',
        unsupported=record.get('unsupported', 'Only documented domain-specific combinations are implemented.'))


def resolve_recommendations(domain, record, context=None):
    """Screen existing candidates using declared task facts; never change execution rules."""
    from math import ceil
    context = context or {}
    model = context.get('model', {})
    params = context.get('params', {})
    shape = model.get('spatial_shape')
    views = context.get('num_views')
    available = context.get('available_inputs')
    defaults = record.get('default_parameters', record.get('parameters', {}))
    # CT discovery calls this before attaching default_parameters.
    if domain == 'ct':
        rules = record['parameter_specs']
        defaults = {p['name']: p['default'] for p in rules}
    else:
        rules = [dict(name=k, **v) for k, v in record['parameter_schema'].items()]
    effective = {**defaults, **params}
    for rule in rules:
        name = rule['name']
        search = rule['recommended_search']
        values = list(search['values'])
        pending = []
        if name in {'level', 'levels'} or (name == 'regularizer' and 'haar_l1' in values):
            if shape is None:
                pending.append('model.spatial_shape: Haar dimensions must be divisible by 2**level')
            elif name == 'regularizer':
                level = effective['level']
                values = [v for v in values if v not in {'haar_l1','tv_haar'} or all(n % 2**level == 0 for n in shape)]
            else:
                values = [v for v in values if all(n % 2**v == 0 for n in shape)]
        elif domain == 'ct' and name in {'block_size', 'subset_count'}:
            if views is None:
                pending.append('num_views: block/subset size and balanced partition')
            else:
                values = [v for v in values if v is None or v <= views]
                other = effective.get('subset_count' if name == 'block_size' else 'block_size')
                if other is not None and record['name'] in {'os_sart', 'osem'}:
                    values = [v for v in values if v is None or (ceil(views / v) == other if name == 'block_size' else ceil(views / other) == v)]
        elif domain == 'mri' and name == 'kernel_width':
            acs = context.get('calibration_shape')
            values = [v for v in values if v % 2 == 1]
            if acs is None:
                pending.append('calibration_shape: spatial ACS sizes must contain kernel_width')
            else:
                values = [v for v in values if v <= min(acs)]
        elif name == 'beta_initial':
            values = [v for v in values if v <= effective['beta_max']]
        elif domain == 'ct' and name in {'data_fidelity', 'weight_source', 'residual_metric'}:
            if available is None:
                pending.append('available_inputs: public sigma, weights, counts and I0 as required')
            else:
                requirements = {'gaussian_nll': {'sigma'}, 'wls': {'weights'},
                                'transmission_poisson_nll': {'counts', 'I0'},
                                'gaussian_whitened': {'sigma'}}
                values = [v for v in values if requirements.get(v, set()) <= set(available)]
            if name == 'data_fidelity' and 'transmission_poisson_nll' in values:
                if 'noise_model' not in model:
                    pending.append('model.noise_model: transmission NLL requires pure Poisson')
                elif model['noise_model'] != 'poisson':
                    values.remove('transmission_poisson_nll')
                if model.get('background', 0) != 0 or model.get('sigma_e', 0) != 0:
                    values = [v for v in values if v != 'transmission_poisson_nll']
            if name == 'weight_source':
                if available is not None and 'weights' not in available:
                    values = [v for v in values if v is None]
                if 'noise_model' not in model:
                    pending.append('model.noise_model: post-log weights must match the noise model')
                else:
                    matched = {'post_log_poisson': 'poisson', 'post_log_poisson_gaussian': 'poisson_gaussian'}
                    values = [v for v in values if v not in matched or matched[v] == model['noise_model']]
        if record['name']=='cg' and name=='regularizer':values=[v for v in values if v=='tikhonov']
        if record['name']=='cg' and name in ('strength','reg_strength'):values=[v for v in values if v is not None and v>0]
        search.update(resolved_values=values, unresolved_conditions=pending,
                      resolution='pending' if pending else 'resolved')
    # Acquisition prerequisites apply to the whole method, independently of a
    # parameter's candidate screening. They are not a full request validation.
    pending = []
    incompatible = []
    for field, allowed in (('sampling', record.get('sampling')), ('boundary', record.get('boundaries'))):
        if allowed:
            if field not in model:
                pending.append(f'model.{field}')
            elif model[field] not in allowed:
                incompatible.append(f'model.{field}={model[field]} requires one of {allowed}')
    record['recommendation_context'] = dict(unresolved_conditions=pending,
        incompatible_conditions=incompatible, scope='Candidate screening only; run the public request validator for full execution prerequisites.')
