"""Conservative regression thresholds for CT workflow tests."""


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


THRESHOLD_FDK_ASTRA = {
    "max_relative_error": 0.30,
    "max_scale_aligned_data_residual": 0.85,
    "min_psnr": 25.0,
    "min_contrast_recovery": 0.65,
    "min_cnr": 4.0,
}
