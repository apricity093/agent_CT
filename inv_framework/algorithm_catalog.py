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


# Descriptions of existing termination tests, not a cross-domain policy mapper.
STOPPING_CRITERIA = {
    'cg_sense': '||normal(x)-rhs|| <= tolerance*||rhs|| (Hermitian CG)',
    'cs_tv': 'max(primal, dual, inner normal residual) <= tolerance*max(1,||x||)',
    'cs_wavelet': 'max(primal, dual, inner normal residual) <= tolerance*max(1,||x||)',
    'richardson_lucy': '||next_x-x||/max(||x||,epsilon) <= tolerance',
    'rl_tv': '||next_x-x||/max(||x||,epsilon) <= tolerance; nonpositive denominator fails',
    'tv_admm': 'max(primal,dual) <= tolerance*max(1,||x||)',
    'hyper_laplacian': 'beta >= beta_max and ||next_x-x||/max(1,||x||) <= tolerance',
    'blind_dcp': 'Fixed max_iter alternating updates per pyramid scale; final_max_iter is a separate final recovery budget',
    'blind_l0': 'Fixed max_iter alternating updates per pyramid scale; final_max_iter is a separate final recovery budget',
    'fista': 'Normalized proximal-gradient mapping and relative composite-objective change; consecutive stopping monitor',
    'tv_fista': 'Normalized proximal-gradient mapping and relative composite-objective change; legacy iterate plateau may stop as stalled',
    'admm': 'Joint primal block residual and adjoint dual residual; consecutive stopping monitor',
    'pdhg': 'Normalized primal-dual fixed-point residual; consecutive stopping monitor',
    'landweber': '||A*Q(Ax-y)+strength L*Lx||/max(1,||A*Qy||) <= tolerance on augmented quadratic path',
    'cg': '||A*Q(Ax-y)+strength x||/max(1,||A*Qy||) <= tolerance',
    'cgls': 'Augmented normalized normal residual <= tolerance; tol used only when tolerance is None; unconstrained',
    'lsqr': 'Augmented normalized normal residual <= tolerance; atol+btol used only when tolerance is None; unconstrained',
    'tikhonov': 'Normal-equation residual tolerance; weighted quadratic path is unconstrained',
    'sirt': 'Normalized data residual or relative iterate change; default control tolerance 1e-5',
    'sart': 'Normalized data residual or relative iterate change; default control tolerance 1e-5',
    'os_sart': 'Full-epoch normalized residual and relative change; default control tolerance 1e-5',
    'mlem': 'Normalized Poisson deviance change and relative iterate change',
    'osem': 'Full-epoch Poisson deviance change and relative change; subset cycle diagnostics',
    'constrained_pdhg': 'Residual-ball feasibility_tolerance and primal-dual fixed-point tolerance',
    'asd_pocs_tv': 'Residual-ball feasibility_tolerance and relative iterate tolerance with relaxation > 0.005',
}


def _input_specs(domain, record):
    """External task inputs; conditions describe alternatives to required fields."""
    name = record['name']
    from inv_framework.experiments import COMPOSABLE_METHODS
    shared = name in COMPOSABLE_METHODS
    if domain == 'ct':
        return [dict(name='observation', required=True, axes='batch,channel,view,detector (2D); batch,detector_v,view,detector_u (3D)', dtype='finite real'),
                dict(name='geometry', required=True, axes=None, dtype='mapping', description='type, domain_shape, range_shape, angles_rad; cone distances/spacing for cone_3d'),
                dict(name='measurement', required=True, axes=None, dtype='mapping', description='Declared observation domain/model and noise metadata'),
                dict(name='public_measurement', required=False, axes='matching observation', dtype='mapping of real arrays', description='counts/I0 for transmission NLL; sigma for Gaussian; weights for WLS'),
                dict(name='valid_measurement_mask', required=False, axes='observation', dtype='boolean'),
                dict(name='truth', required=False, axes='batch,*domain_shape', dtype='finite real', description='Evaluation only; requires positive data_range'),
                dict(name='data_range', required=False, axes=None, dtype='positive real', description='Required with truth')]
    if domain == 'mri':
        spatial = 'y,x' if shared or name in {'varnet','grappa','espirit'} else 'y,x or z,y,x'
        axes = f'batch,coil,{spatial}; noncartesian kspace: batch,coil,sample'
        specs = dict(
            kspace=(axes, 'finite complex', True, None),
            sampling_mask=(f'batch,coil,{spatial}', 'binary real/bool, same shape as kspace', shared or name in {'zero_filled','varnet','grappa'}, 'Required for Cartesian sampling'),
            sensitivity_maps=(f'batch,coil,{spatial}', 'finite complex', shared or name in {'cg_sense','cs_tv','cs_wavelet'}, 'Required for SENSE coil combination; not accepted by VarNet/calibration'),
            trajectory=('sample,spatial_dimension', 'finite real', False, 'Required for noncartesian sampling; declared trajectory_units, converted to [-pi,pi)'),
            calibration_data=('batch,coil,acs_y,acs_x', 'finite complex', True, 'Explicit ACS; odd kernel_width fits acquired grid'),
            data_weights=('same as kspace', 'finite nonnegative real', False, 'Required for WLS; optional Gaussian inverse variance'),
            initial_estimate=(f'batch,{spatial}', 'finite complex', False, 'Initial object'))
        if name == 'espirit': keys = ['calibration_data']
        elif name == 'grappa': keys = ['kspace','sampling_mask','calibration_data']
        elif name == 'varnet': keys = ['kspace','sampling_mask']
        elif name == 'zero_filled': keys = ['kspace','sampling_mask','sensitivity_maps']
        else: keys = ['kspace','sampling_mask','sensitivity_maps','data_weights','initial_estimate'] + ([] if shared else ['trajectory'])
        return [dict(name=k, axes=specs[k][0], dtype=specs[k][1], required=specs[k][2], condition=specs[k][3]) for k in keys]
    blind = name.startswith('blind_')
    axes = 'y,x with optional last channel' if blind else 'y,x' if shared else 'y,x or z,y,x with optional last channel'
    specs = [dict(name='observation', required=True, axes=axes, dtype='finite real; nonnegative for Poisson')]
    if blind:
        specs.append(dict(name='initial_psf', required=False, axes='kernel_y,kernel_x', dtype='finite nonnegative real, positive sum', condition='Matches odd model.kernel_shape; normalized on entry'))
    else:
        specs.extend([dict(name='psf', required=True, axes='spatial PSF axes', dtype='finite real; nonnegative positive sum for Poisson'),
                      dict(name='initial_estimate', required=False, axes=axes, dtype='finite real; nonnegative for Poisson')])
        if name == 'wiener_hunt': specs.pop()
        if shared: specs.append(dict(name='data_weights', required=False, axes='observation', dtype='finite nonnegative real', condition='Required for WLS'))
    return specs


def describe_execution(domain, record, context=None):
    """Attach declarative controls and cheap prerequisite screening to a record.

    Screening never validates array contents, loads a checkpoint or probes GPUs.
    A pending state is deliberate: installed dependencies do not establish that a
    particular acquisition and its numerical parameters satisfy the contract.
    """
    import importlib.util
    from pathlib import Path
    from inv_framework.experiments import COMPOSABLE_METHODS
    context = context or {}
    name = record['name']
    shared = domain != 'ct' and name in COMPOSABLE_METHODS
    defaults = record.get('default_parameters', record.get('parameters', {}))
    params = {**defaults, **context.get('params', {})}
    rules = record.get('parameter_schema', {})
    if domain != 'ct':
        for key, rule in rules.items():
            rule['default'] = defaults[key]
        record.setdefault('devices', ['cpu','cuda'] if name == 'varnet' else ['cpu'])
        record.setdefault('spatial_dimensions', [2] if shared or name in {'varnet','grappa','espirit','blind_dcp','blind_l0'} else [2,3])
        if name == 'varnet': record.setdefault('sampling', ['cartesian'])
        if name.startswith('blind_'): record.setdefault('boundaries', ['periodic'])
    record.update(modality=domain, input_specs=_input_specs(domain, record))
    record.setdefault('task_types', ['reconstruction'])
    record.setdefault('outputs', ['reconstruction'])
    record['input_source'] = 'external inputs or explicit case_id' if domain == 'ct' else 'external inputs'
    budget = 'num_iterations' if 'num_iterations' in defaults else 'max_iter' if 'max_iter' in defaults else None
    threshold_keys = [k for k in defaults if k in {'tolerance','tol','atol','btol','feasibility_tolerance'}]
    inner = [k for k in defaults if k in {'cg_max_iter','cg_max_iterations','cg_tolerance','tv_num_iterations','tv_tolerance','nll_max_iterations','nll_tolerance','kernel_max_iter','latent_iter','epsilon_levels','final_max_iter'}]
    constraints = []
    if domain == 'ct':
        constraints = list(record['metadata'].get('constraints', []))
        if not constraints and 'min_value' in defaults: constraints = ['box']
        if name == 'landweber': constraints = ['box on legacy unweighted L2 path; augmented quadratic path unconstrained']
        if name == 'tikhonov': constraints = ['unconstrained']
    elif domain == 'deconvolution':
        if shared and name in {'fista','admm','pdhg'}: constraints = ['min_value','max_value']
        elif name == 'tv_admm': constraints = ['nonnegative (params; defaults to Poisson selection)']
        elif name in {'richardson_lucy','rl_tv'}: constraints = ['nonnegative (intrinsic multiplicative update)']
        elif name.startswith('blind_'): constraints = ['PSF nonnegative, unit sum, centered support; latent image unconstrained']
    criterion = STOPPING_CRITERIA.get(name, 'Direct completion; no convergence threshold')
    if shared and name in {'cgls','lsqr'}:
        criterion = 'Augmented normalized normal residual <= tolerance; no real box constraints'
    from inv_framework.experiments import execution_controls
    mappings = execution_controls(domain, name, defaults)
    record['controls'] = dict(
        native=dict(iteration_parameter=budget, iteration_scope='per pyramid scale' if name.startswith('blind_') else 'outer iterations/epochs',
                    threshold_parameters=threshold_keys, stopping_criterion=criterion,
                    inner_parameters=inner, constraints=constraints,
                    fixed_inference_parameter='num_cascades' if name == 'varnet' else None),
        unified_entry=dict(params=True, stopping_policy=(budget is not None if domain == 'ct' else bool(mappings['stopping_policy'])),
                           problem_constraints=(any(c != 'unconstrained' for c in constraints) if domain == 'ct' else bool(mappings['problem_constraints'])),
                           field_mappings=mappings if domain != 'ct' else None,
                           note='CT policies retain algorithm-specific semantics; direct methods have no iterative stop.' if domain == 'ct' else
                           'Only field_mappings are supported; tolerance uses the native stopping_criterion. Outer budget leaves inner_parameters unchanged.'))
    model_fields = (['geometry','measurement'] if domain == 'ct' else
                    ['axes','sampling','spatial_shape'] if domain == 'mri' else
                    ['axes','boundary','kernel_shape'] if name.startswith('blind_') else ['axes','boundary','psf_origin'])
    record['model_requirements'] = model_fields
    requirements = list(record.get('requirements', [])) if domain == 'ct' else [record.get('applicability', 'See objective and domain input contract')]
    if domain == 'mri':
        requirements += ['Centered orthonormal FFT; explicit finite maps retain their supplied scale']
        if name in {'varnet','grappa'}: requirements += ['Multicoil Cartesian sampling with shared masks; VarNet requires acquired central ACS columns']
    record['runtime_requirements'] = requirements
    modules = ['numpy','yaml']
    if domain == 'ct' or shared or name == 'varnet': modules.append('torch')
    if domain == 'deconvolution' and name.startswith('blind_'): modules.append('scipy')
    if domain == 'ct':
        backend = context.get('backend', record['execution_backend'])
        if backend == 'astra': modules.append('astra')
        elif backend == 'leap': modules.append('leapctype')
    sampling = context.get('model', {}).get('sampling')
    conditional = []
    if domain == 'mri' and 'noncartesian' in record.get('sampling', []):
        conditional.append(dict(module='torchkbnufft', condition='model.sampling=noncartesian', installed=importlib.util.find_spec('torchkbnufft') is not None))
        if sampling == 'noncartesian': modules += ['torch','torchkbnufft']
    dependencies = {module: importlib.util.find_spec(module) is not None for module in modules}
    record['dependencies'] = dict(required=dependencies, conditional=conditional)
    record['weights'] = dict(required=name == 'varnet', parameter='checkpoint' if name == 'varnet' else None,
                             condition='Explicit local checkpoint matching architecture; no download' if name == 'varnet' else None)
    pending, reasons = [], []
    reasons.extend(f'Missing dependency: {m}' for m, installed in dependencies.items() if not installed)
    available = context.get('available_inputs')
    required = [s['name'] for s in record['input_specs'] if s['required']]
    if domain == 'mri' and name == 'zero_filled' and params['coil_combination'] == 'sense': required += ['sensitivity_maps']
    if domain == 'mri' and sampling == 'cartesian' and name != 'espirit': required += ['sampling_mask']
    if domain == 'mri' and sampling == 'noncartesian': required += ['trajectory']
    if available is None: pending.append('available_inputs: ' + ', '.join(dict.fromkeys(required)))
    else: reasons.extend(f'Missing input: {k}' for k in dict.fromkeys(required) if k not in available)
    model = context.get('model', {})
    pending.extend(f'model.{k}' for k in model_fields if k not in model and domain != 'ct')
    if domain == 'ct': pending.extend(k for k in model_fields if k not in model)
    reasons.extend(record['recommendation_context']['incompatible_conditions'])
    # Screen declared facts only. Array and coupled numerical validation remains
    # with the existing API boundary, rather than being duplicated in discovery.
    for field, allowed in [('sampling',record.get('sampling')), ('boundary',record.get('boundaries'))]:
        if allowed and field in model and model[field] not in allowed:
            reason = f'model.{field}={model[field]} requires one of {allowed}'
            if reason not in reasons: reasons.append(reason)
    if domain == 'ct':
        backend = context.get('backend', record['execution_backend'])
        if backend not in record['backends']: reasons.append(f'Unsupported backend: {backend}')
    if domain == 'mri' and model.get('spatial_shape') is not None:
        if len(model['spatial_shape']) not in record['spatial_dimensions']:
            reasons.append('Unsupported spatial dimension')
    if domain == 'mri' and (params.get('min_value') is not None or params.get('max_value') is not None):
        reasons.append('Complex MRI does not accept real box constraints')
    device = context.get('device')
    devices = ['cuda'] if name == 'fdk' else ['cpu','cuda'] if domain == 'ct' or name == 'varnet' else ['cpu']
    record.setdefault('devices', devices)
    if device is None: pending.append('device: ' + ', '.join(devices))
    elif device.split(':')[0] not in devices: reasons.append(f'Unsupported device: {device}; supported: {devices}')
    if name == 'fdk' or (device and device.startswith('cuda')):
        pending.append('CUDA runtime/device availability (not probed by discovery)' + (' and ASTRA CUDA support' if domain == 'ct' else ''))
    if name == 'varnet':
        checkpoint = params['checkpoint']
        if not checkpoint: pending.append('params.checkpoint: matching local weights required')
        elif not Path(checkpoint).is_file(): reasons.append('Checkpoint file does not exist')
        else: pending.append('Checkpoint architecture compatibility (not loaded by discovery)')
    pending.append('Input shapes/dtypes, model and coupled parameter validation at execution')
    record.update(implemented=record.get('implemented', record.get('implementation_status') == 'implemented'),
                  dependency_installed=all(dependencies.values()), runnable=False if reasons else None,
                  runtime_status=dict(state='unavailable' if reasons else 'pending', reasons=reasons,
                                      unresolved_conditions=pending, scope='Prerequisite screening; no solver, weight loading or hardware probe'))


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
