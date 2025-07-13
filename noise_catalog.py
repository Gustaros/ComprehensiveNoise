import noises

# Функция создания тестового изображения (копия из noises.py)
def create_sample_image(size=(256, 256), pattern="gradient"):
    import numpy as np
    if pattern == "gradient":
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        image = xx * yy * 255
    elif pattern == "constant":
        image = np.ones(size) * 128
    elif pattern == "circles":
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        image = (np.sin(r * 15) + 1) * 127.5
    elif pattern == "checkerboard":
        x = np.arange(size[0])
        y = np.arange(size[1])
        xx, yy = np.meshgrid(x, y)
        image = ((xx + yy) % 2) * 255
    else:
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        image = xx * yy * 255
    return image.astype(np.uint8)

# Полный словарь noise_functions (копия из main() noises.py)
noise_functions = {
    # 1. STATISTICAL DISTRIBUTION-BASED NOISE TYPES
    "1.1 Gaussian Noise": noises.add_gaussian_noise,
    "1.2 Salt and Pepper Noise": noises.add_salt_and_pepper_noise,
    "1.3 Poisson Noise": noises.add_poisson_noise,
    "1.4 Speckle Noise": noises.add_speckle_noise,
    "1.5 Uniform Noise": noises.add_uniform_noise,
    "1.6 Rayleigh Noise": noises.add_rayleigh_noise,
    "1.7 Gamma Noise": noises.add_gamma_noise,
    "1.8 Exponential Noise": noises.add_exponential_noise,
    "1.9 Laplacian Noise": noises.add_laplacian_noise,
    "1.10 Cauchy Noise": noises.add_cauchy_noise,
    "1.11 Chi-Square Noise": noises.add_chi2_noise,
    "1.12 Beta Noise": noises.add_beta_noise,
    "1.13 Weibull Noise": noises.add_weibull_noise,
    "1.14 Logistic Noise": noises.add_logistic_noise,
    "1.15 Student's t-Noise": noises.add_t_noise,
    "1.16 F-Distribution Noise": noises.add_f_noise,
    "1.17 Lognormal Noise": noises.add_lognormal_noise,
    "1.18 Binomial Noise": noises.add_binomial_noise,
    "1.19 Negative Binomial Noise": noises.add_nbinom_noise,
    "1.20 Hypergeometric Noise": noises.add_hypergeom_noise,
    "1.21 Pareto Noise": noises.add_pareto_noise,
    "1.22 Maxwell-Boltzmann Noise": noises.add_maxwell_noise,
    # 2. COLORED NOISE TYPES
    "2.1 White Noise": noises.add_white_noise,
    "2.2 Pink Noise": noises.add_pink_noise,
    "2.3 Brown Noise": noises.add_brown_noise,
    "2.4 Blue Noise": noises.add_blue_noise,
    "2.5 Violet Noise": noises.add_violet_noise,
    "2.6 Grey Noise": noises.add_grey_noise,
    "2.7 Red Noise": noises.add_red_noise,
    "2.8 Orange Noise": noises.add_orange_noise,
    "2.9 Green Noise": noises.add_green_noise,
    "2.10 Black Noise": noises.add_black_noise,
    # 3. PROCEDURAL AND SYNTHETIC NOISE TYPES
    "3.1 Perlin Noise": noises.add_perlin_noise if hasattr(noises, 'add_perlin_noise') else noises.add_simplex_noise,
    "3.2 Simplex Noise": noises.add_simplex_noise,
    "3.3 Worley Noise": noises.add_worley_noise,
    "3.4 Fractional Brownian Motion": noises.add_fbm_noise,
    "3.5 Value Noise": noises.add_value_noise,
    "3.6 Wavelet Noise": noises.add_wavelet_noise,
    "3.7 Diamond-Square Noise": noises.add_diamond_square_noise,
    "3.8 Gabor Noise": noises.add_gabor_noise,
    "3.9 Sparse Convolution Noise": noises.add_sparse_convolution_noise,
    "3.10 Turbulence Noise": noises.add_turbulence_noise,
    "3.11 Mosaic Noise": noises.add_mosaic_noise,
    "3.12 Ridge Noise": noises.add_ridge_noise,
    "3.13 Curl Noise": noises.add_curl_noise,
    "3.14 Caustic Noise": noises.add_caustic_noise,
    "3.15 Cracks Noise": noises.add_cracks_noise,
    "3.16 Flow Noise": noises.add_flow_noise,
    # 4. PHYSICAL AND DEVICE-BASED NOISE TYPES
    "4.1 Film Grain Noise": noises.add_film_grain,
    "4.2 CCD/CMOS Sensor Noise": noises.add_sensor_noise,
    "4.3 Thermal Noise": noises.add_thermal_noise,
    "4.4 Fixed Pattern Noise": noises.add_fixed_pattern_noise,
    "4.5 Row/Column Noise": noises.add_row_column_noise,
    "4.6 Banding Noise": noises.add_banding_noise,
    "4.7 JPEG Compression Noise": noises.add_jpeg_noise,
    "4.8 Quantization Noise": noises.add_quantization_noise,
    "4.9 Demosaicing Noise": noises.add_demosaicing_noise,
    "4.10 Lens Blur": noises.add_lens_blur,
    "4.11 Motion Blur": noises.add_motion_blur,
    "4.12 Vignetting": noises.add_vignetting,
    "4.13 Blooming": noises.add_blooming,
    "4.14 Chromatic Aberration": noises.add_chromatic_aberration,
    "4.15 Hot Pixels": noises.add_hot_pixels,
    "4.16 Dead Pixels": noises.add_dead_pixels,
    "4.17 Rolling Shutter Effect": noises.add_rolling_shutter,
    "4.18 Moiré Pattern": noises.add_moire_pattern,
    # 5. REAL-WORLD AND ENVIRONMENTAL NOISE TYPES
    "5.1 Rain Noise": noises.add_rain_noise,
    "5.2 Snow Noise": noises.add_snow_noise,
    "5.3 Fog/Haze": noises.add_fog,
    "5.4 Dust and Scratches": noises.add_dust_scratches,
    "5.5 Lens Flare": noises.add_lens_flare,
    "5.6 Water Droplets": noises.add_water_droplets,
    "5.7 Smoke Effect": noises.add_smoke,
    "5.8 Scratched Film": noises.add_scratched_film,
    "5.9 Glitch Effect": noises.add_glitch,
    "5.10 Pixelation": noises.add_pixelation,
    # 6. ADDITIONAL STATISTICAL DISTRIBUTIONS
    "6.1 Rice/Rician Noise": noises.add_rician_noise,
    "6.2 Von Mises Noise": noises.add_vonmises_noise,
    "6.3 Gumbel/Extreme Value Noise": noises.add_gumbel_noise,
    "6.4 Lévy Noise": noises.add_levy_noise,
    "6.5 Nakagami Noise": noises.add_nakagami_noise,
    # 7. PHYSICAL AND QUANTUM NOISE
    "7.1 Johnson-Nyquist Noise": noises.add_johnson_nyquist_noise,
    "7.2 Flicker Noise": noises.add_flicker_noise,
    "7.3 Quantum Shot Noise": noises.add_quantum_shot_noise,
    "7.4 Avalanche Noise": noises.add_avalanche_noise,
    "7.5 Generation-Recombination Noise": noises.add_gr_noise,
    # 8. SPECIALIZED FIELDS NOISE
    "8.1 K-Distribution Noise": noises.add_k_distribution_noise,
    "8.2 Photon-Limited Noise": noises.add_photon_limited_noise,
    "8.3 Phase Noise": noises.add_phase_noise,
    "8.4 Cosmic Noise": noises.add_cosmic_noise,
    "8.5 Jitter Noise": noises.add_jitter_noise,
    # 9. MATHEMATICAL AND THEORETICAL NOISE
    "9.1 Lévy Stable Distribution Noise": noises.add_levy_stable_noise,
    "9.2 Chaotic Noise": noises.add_chaotic_noise,
    "9.3 Brownian Bridge Noise": noises.add_brownian_bridge_noise,
    "9.4 Multifractal Noise": noises.add_multifractal_noise,
    # 10. NEW TECHNOLOGY NOISE
    "10.1 Advanced Compression Artifacts": noises.add_advanced_compression_artifacts,
    "10.2 HDR Artifacts Noise": noises.add_hdr_artifacts,
    "10.3 Deep Learning Artifacts": noises.add_dl_artifacts,
    # 11. COMBINATIONAL AND SPECIFIC NOISE
    "11.1 Non-Stationary Noise": noises.add_nonstationary_noise,
    "11.2 Heteroskedastic Noise": noises.add_heteroskedastic_noise,
    "11.3 Anisotropic Noise": noises.add_anisotropic_noise,
    "11.4 Structured Correlated Noise": noises.add_structured_correlated_noise,
    # 12. ENVIRONMENT AND SPECIAL EFFECTS NOISE
    "12.1 Atmospheric Turbulence": noises.add_atmospheric_turbulence,
    "12.2 Electrical Interference Patterns": noises.add_electrical_interference,
    "12.3 Scanner Artifacts": noises.add_scanner_artifacts,
    "12.4 Print Artifacts": noises.add_print_artifacts,
    "12.5 Polarization Noise": noises.add_polarization_noise,
    # 13. SPECIALIZED SCIENTIFIC FIELDS NOISE
    "13.1 Spectroscopic Noise": noises.add_spectroscopic_noise,
    "13.2 Gravitational Wave Noise": noises.add_gravitational_wave_noise,
    "13.3 Astronomical Noise": noises.add_astronomical_noise,
    "13.4 Neuroimaging Noise": noises.add_neuroimaging_noise,
    "13.5 Quantum Computing Noise": noises.add_quantum_computing_noise,
    # 14. NEWEST NOISE TYPES AND FUTURE DIRECTIONS
    "14.1 Holographic Noise": noises.add_holographic_noise,
    "14.2 Diffraction Noise": noises.add_diffraction_noise,
    "14.3 Mixed Reality Noise": noises.add_mixed_reality_noise,
    "14.4 Machine Vision Noise": noises.add_machine_vision_noise,
    "14.5 HDR-Specific Noise": noises.add_hdr_specific_noise
}

# Категории и соответствующие шумы
NOISE_CATEGORIES = {
    "1. Статистические распределения": [
        "1.1 Gaussian Noise", "1.2 Salt and Pepper Noise", "1.3 Poisson Noise", "1.4 Speckle Noise", "1.5 Uniform Noise", "1.6 Rayleigh Noise", "1.7 Gamma Noise", "1.8 Exponential Noise", "1.9 Laplacian Noise", "1.10 Cauchy Noise", "1.11 Chi-Square Noise", "1.12 Beta Noise", "1.13 Weibull Noise", "1.14 Logistic Noise", "1.15 Student's t-Noise", "1.16 F-Distribution Noise", "1.17 Lognormal Noise", "1.18 Binomial Noise", "1.19 Negative Binomial Noise", "1.20 Hypergeometric Noise", "1.21 Pareto Noise", "1.22 Maxwell-Boltzmann Noise"
    ],
    "2. Цветные шумы": [
        "2.1 White Noise", "2.2 Pink Noise", "2.3 Brown Noise", "2.4 Blue Noise", "2.5 Violet Noise", "2.6 Grey Noise", "2.7 Red Noise", "2.8 Orange Noise", "2.9 Green Noise", "2.10 Black Noise"
    ],
    "3. Процедурные и синтетические шумы": [
        "3.1 Perlin Noise", "3.2 Simplex Noise", "3.3 Worley Noise", "3.4 Fractional Brownian Motion", "3.5 Value Noise", "3.6 Wavelet Noise", "3.7 Diamond-Square Noise", "3.8 Gabor Noise", "3.9 Sparse Convolution Noise", "3.10 Turbulence Noise", "3.11 Mosaic Noise", "3.12 Ridge Noise", "3.13 Curl Noise", "3.14 Caustic Noise", "3.15 Cracks Noise", "3.16 Flow Noise"
    ],
    "4. Аппаратные и физические шумы": [
        "4.1 Film Grain Noise", "4.2 CCD/CMOS Sensor Noise", "4.3 Thermal Noise", "4.4 Fixed Pattern Noise", "4.5 Row/Column Noise", "4.6 Banding Noise", "4.7 JPEG Compression Noise", "4.8 Quantization Noise", "4.9 Demosaicing Noise", "4.10 Lens Blur", "4.11 Motion Blur", "4.12 Vignetting", "4.13 Blooming", "4.14 Chromatic Aberration", "4.15 Hot Pixels", "4.16 Dead Pixels", "4.17 Rolling Shutter Effect", "4.18 Moiré Pattern"
    ],
    "5. Окружающая среда и реальные эффекты": [
        "5.1 Rain Noise", "5.2 Snow Noise", "5.3 Fog/Haze", "5.4 Dust and Scratches", "5.5 Lens Flare", "5.6 Water Droplets", "5.7 Smoke Effect", "5.8 Scratched Film", "5.9 Glitch Effect", "5.10 Pixelation"
    ],
    "6. Дополнительные статистические распределения": [
        "6.1 Rice/Rician Noise", "6.2 Von Mises Noise", "6.3 Gumbel/Extreme Value Noise", "6.4 Lévy Noise", "6.5 Nakagami Noise"
    ],
    "7. Физические и квантовые шумы": [
        "7.1 Johnson-Nyquist Noise", "7.2 Flicker Noise", "7.3 Quantum Shot Noise", "7.4 Avalanche Noise", "7.5 Generation-Recombination Noise"
    ],
    "8. Специализированные шумы": [
        "8.1 K-Distribution Noise", "8.2 Photon-Limited Noise", "8.3 Phase Noise", "8.4 Cosmic Noise", "8.5 Jitter Noise"
    ],
    "9. Математические и теоретические шумы": [
        "9.1 Lévy Stable Distribution Noise", "9.2 Chaotic Noise", "9.3 Brownian Bridge Noise", "9.4 Multifractal Noise"
    ],
    "10. Шумы новых технологий": [
        "10.1 Advanced Compression Artifacts", "10.2 HDR Artifacts Noise", "10.3 Deep Learning Artifacts"
    ],
    "11. Комбинационные и специфические шумы": [
        "11.1 Non-Stationary Noise", "11.2 Heteroskedastic Noise", "11.3 Anisotropic Noise", "11.4 Structured Correlated Noise"
    ],
    "12. Шумы среды и спецэффекты": [
        "12.1 Atmospheric Turbulence", "12.2 Electrical Interference Patterns", "12.3 Scanner Artifacts", "12.4 Print Artifacts", "12.5 Polarization Noise"
    ],
    "13. Научные специализированные шумы": [
        "13.1 Spectroscopic Noise", "13.2 Gravitational Wave Noise", "13.3 Astronomical Noise", "13.4 Neuroimaging Noise", "13.5 Quantum Computing Noise"
    ],
    "14. Новейшие типы шумов": [
        "14.1 Holographic Noise", "14.2 Diffraction Noise", "14.3 Mixed Reality Noise", "14.4 Machine Vision Noise", "14.5 HDR-Specific Noise"
    ]
}
