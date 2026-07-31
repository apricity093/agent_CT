"""Approved local calibration records and conservative regression thresholds."""

CALIBRATION_ENVIRONMENT = {
    "date": "2026-07-31",
    "python": "3.11.11",
    "torch": "2.5.1",
    "astra": "2.2.0",
    "cuda_runtime": "12.1",
    "gpu": "NVIDIA GeForce GTX 1650",
    "precision": "float32",
}


BASELINES_2D = {
    "clean": {
        "cgls": {"relative_error": 0.13993, "data_residual": 0.01738, "psnr": 25.45, "ssim": 0.918},
        "lsqr": {"relative_error": 0.12981, "data_residual": 0.04426, "psnr": 26.10, "ssim": 0.696},
        "sart": {"relative_error": 0.16672, "data_residual": 0.03082, "psnr": 23.92, "ssim": 0.822},
        "os_sart": {"relative_error": 0.30595, "data_residual": 0.10442, "psnr": 18.65, "ssim": 0.505},
        "tikhonov": {"relative_error": 0.09880, "data_residual": 0.01722, "psnr": 28.47, "ssim": 0.836},
        "tv_fista": {"relative_error": 0.14247, "data_residual": 0.01882, "psnr": 25.29, "ssim": 0.914},
    },
    "sparse_noisy": {
        "cgls": {"relative_error": 0.14688, "data_residual": 0.01945, "psnr": 25.02, "ssim": 0.900},
        "lsqr": {"relative_error": 0.15313, "data_residual": 0.05043, "psnr": 24.66, "ssim": 0.659},
        "sart": {"relative_error": 0.19598, "data_residual": 0.04277, "psnr": 22.52, "ssim": 0.775},
        "os_sart": {"relative_error": 0.36888, "data_residual": 0.15460, "psnr": 17.03, "ssim": 0.401},
        "tikhonov": {"relative_error": 0.13155, "data_residual": 0.02924, "psnr": 25.98, "ssim": 0.733},
        "tv_fista": {"relative_error": 0.14586, "data_residual": 0.01971, "psnr": 25.09, "ssim": 0.901},
    },
}


THRESHOLDS_2D = {
    "clean": {
        "cgls": {"max_relative_error": 0.20, "max_data_residual": 0.03, "min_psnr": 24.0, "min_ssim": 0.85},
        "lsqr": {"max_relative_error": 0.20, "max_data_residual": 0.07, "min_psnr": 24.0, "min_ssim": 0.60},
        "sart": {"max_relative_error": 0.25, "max_data_residual": 0.06, "min_psnr": 21.5, "min_ssim": 0.72},
        "os_sart": {"max_relative_error": 0.45, "max_data_residual": 0.18, "min_psnr": 16.0, "min_ssim": 0.35},
        "tikhonov": {"max_relative_error": 0.16, "max_data_residual": 0.03, "min_psnr": 26.0, "min_ssim": 0.75},
        "tv_fista": {"max_relative_error": 0.20, "max_data_residual": 0.04, "min_psnr": 23.5, "min_ssim": 0.85},
    },
    "sparse_noisy": {
        "cgls": {"max_relative_error": 0.22, "max_data_residual": 0.04, "min_psnr": 23.0, "min_ssim": 0.80},
        "lsqr": {"max_relative_error": 0.23, "max_data_residual": 0.08, "min_psnr": 22.5, "min_ssim": 0.55},
        "sart": {"max_relative_error": 0.30, "max_data_residual": 0.08, "min_psnr": 20.0, "min_ssim": 0.65},
        "os_sart": {"max_relative_error": 0.50, "max_data_residual": 0.25, "min_psnr": 14.0, "min_ssim": 0.25},
        "tikhonov": {"max_relative_error": 0.20, "max_data_residual": 0.06, "min_psnr": 23.0, "min_ssim": 0.60},
        "tv_fista": {"max_relative_error": 0.22, "max_data_residual": 0.04, "min_psnr": 23.0, "min_ssim": 0.80},
    },
}


BASELINE_FDK_ASTRA = {
    "relative_error": 0.16430,
    "raw_data_residual": 287.45557,
    "scale_aligned_data_residual": 0.70582,
    "projection_norm_ratio": 288.16309,
    "psnr": 30.94,
    "contrast_recovery": 0.9040,
    "cnr": 7.839,
}


THRESHOLD_FDK_ASTRA = {
    "max_relative_error": 0.30,
    "max_scale_aligned_data_residual": 0.85,
    "min_psnr": 25.0,
    "min_contrast_recovery": 0.65,
    "min_cnr": 4.0,
}
