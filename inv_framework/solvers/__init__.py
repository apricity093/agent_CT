from .base import ConsecutiveStoppingMonitor, InverseProblemSolver, IterationRecord, IterationRecorder, SolveControl, SolveResult
from .classical import FBPSolver, SIRTSolver, LandweberSolver, fbp, sirt, landweber
from .dip import DIPSolver
from .inr import INRSolver

__all__ = [
    "InverseProblemSolver",
    "SolveControl",
    "ConsecutiveStoppingMonitor",
    "IterationRecord",
    "IterationRecorder",
    "SolveResult",
    "FBPSolver",
    "SIRTSolver",
    "LandweberSolver",
    "fbp",
    "sirt",
    "landweber",
    "DIPSolver",
    "INRSolver",
]

from .classical import CGLSSolver, LSQRSolver, FDKSolver, cgls, lsqr, fdk
from .subset import SARTSolver, OSSARTSolver, sart, ossart
from .statistical import MLEMSolver, OSEMSolver, mlem, osem

__all__ += [
    "CGLSSolver",
    "LSQRSolver",
    "FDKSolver",
    "cgls",
    "lsqr",
    "fdk",
    "SARTSolver",
    "OSSARTSolver",
    "sart",
    "ossart",
    "MLEMSolver",
    "OSEMSolver",
    "mlem",
    "osem",
]

from .regularized import TikhonovSolver, TVFISTASolver, tikhonov, tv_fista
from .specs import (
    ALGORITHM_ALIASES,
    ALGORITHM_ALIAS_MAP,
    CANONICAL_ALGORITHM_IDS,
    CANONICAL_CT_ALGORITHM_IDS,
    COMPATIBILITY_REASON_CODES,
    CT_COMPATIBILITY_REASON_CODES,
    CTAlgorithmSpec,
    CompatibilityRule,
    EMISSION_OBSERVATION_MODELS,
    NON_APPLICABLE_PARAMETER_CATEGORIES,
    NON_APPLICABLE_PARAMETERS,
    PARAMETER_APPLICABILITY,
    ParameterSpec,
    ParameterValidationResult,
    REGISTRY_SCHEMA_VERSION,
    REGULARIZER_SPECS,
    RegularizerSpec,
    SOLVER_SPECS,
    TRANSMISSION_OBSERVATION_MODELS,
    compatibility_diagnostics,
    registry_contract,
    registry_digest,
    regularizer_records,
    solver_records,
    validate_registry,
    validate_compatibility,
    validate_compatibility_details,
    validate_parameter_values,
)

__all__ += [
    "REGISTRY_SCHEMA_VERSION",
    "CANONICAL_ALGORITHM_IDS",
    "CANONICAL_CT_ALGORITHM_IDS",
    "ALGORITHM_ALIASES",
    "ALGORITHM_ALIAS_MAP",
    "COMPATIBILITY_REASON_CODES",
    "CT_COMPATIBILITY_REASON_CODES",
    "PARAMETER_APPLICABILITY",
    "NON_APPLICABLE_PARAMETER_CATEGORIES",
    "NON_APPLICABLE_PARAMETERS",
    "EMISSION_OBSERVATION_MODELS",
    "TRANSMISSION_OBSERVATION_MODELS",
    "TikhonovSolver",
    "TVFISTASolver",
    "tikhonov",
    "tv_fista",
    "ParameterSpec",
    "ParameterValidationResult",
    "RegularizerSpec",
    "CTAlgorithmSpec",
    "CompatibilityRule",
    "SOLVER_SPECS",
    "REGULARIZER_SPECS",
    "registry_contract",
    "registry_digest",
    "validate_registry",
    "solver_records",
    "regularizer_records",
    "compatibility_diagnostics",
    "validate_compatibility",
    "validate_compatibility_details",
    "validate_parameter_values",
]

# Optional detailed execution entry points.  Imports are kept at the end so
# the legacy solver modules can still import the base contract without a
# circular dependency during module initialization.
from .detailed import (
    solve_cgls_detailed,
    solve_fbp_detailed,
    solve_fdk_detailed,
    solve_landweber_detailed,
    solve_lsqr_detailed,
    solve_mlem_detailed,
    solve_osem_detailed,
    solve_os_sart_detailed,
    solve_sart_detailed,
    solve_sirt_detailed,
    solve_tikhonov_detailed,
    solve_tv_fista_detailed,
)

__all__ += [
    "solve_cgls_detailed",
    "solve_fbp_detailed",
    "solve_fdk_detailed",
    "solve_landweber_detailed",
    "solve_lsqr_detailed",
    "solve_mlem_detailed",
    "solve_osem_detailed",
    "solve_os_sart_detailed",
    "solve_sart_detailed",
    "solve_sirt_detailed",
    "solve_tikhonov_detailed",
    "solve_tv_fista_detailed",
]
