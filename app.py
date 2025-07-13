import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from noise_catalog import noise_functions, create_sample_image
import inspect

st.title("Визуализация шумов (ComprehensiveNoise)")

# Категории и шумы (структурируем по категориям)
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

# Сайдбар: выбор категории и шума
category = st.sidebar.selectbox("Категория шума", list(NOISE_CATEGORIES.keys()))
noise_names = NOISE_CATEGORIES[category]
selected_noise = st.sidebar.selectbox("Тип шума", noise_names)

# Параметры изображения
img_size = st.slider("Размер изображения", 64, 512, 256, step=32)

# Генерируем тестовое изображение
image = create_sample_image(size=(img_size, img_size), pattern="gradient")

# Получаем функцию и docstring
noise_func = noise_functions[selected_noise]
doc = inspect.getdoc(noise_func)

st.markdown(f"**Описание:**\n{doc}")

# Применяем шум
noisy_img, noise = noise_func(image)

# Визуализация изображений
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Оригинал')
axs[0].axis('off')
axs[1].imshow(noisy_img, cmap='gray')
axs[1].set_title('С шумом')
axs[1].axis('off')
axs[2].imshow(noise, cmap='viridis')
axs[2].set_title('Шум')
axs[2].axis('off')
st.pyplot(fig)

# Визуализация гистограмм
fig_hist, axs_hist = plt.subplots(1, 3, figsize=(12, 3))
axs_hist[0].hist(image.ravel(), bins=64, color='gray')
axs_hist[0].set_title('Гистограмма оригинала')
axs_hist[1].hist(noisy_img.ravel(), bins=64, color='blue')
axs_hist[1].set_title('Гистограмма с шумом')
axs_hist[2].hist(noise.ravel(), bins=64, color='purple')
axs_hist[2].set_title('Гистограмма шума')
for ax in axs_hist:
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
st.pyplot(fig_hist)

st.caption("© ComprehensiveNoise, 2025")
