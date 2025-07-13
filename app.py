import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from noises import create_sample_image
import noises
import inspect

# Получаем словарь функций шумов из main()
def get_noise_functions():
    # Словарь из main() (скопировано вручную, т.к. main не возвращает его)
    noise_functions = {
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
        # ...existing code... (добавить остальные шумы по аналогии)
    }
    return noise_functions

st.title("Визуализация шумов (ComprehensiveNoise)")

noise_functions = get_noise_functions()
noise_names = list(noise_functions.keys())

selected_noise = st.selectbox("Выберите тип шума:", noise_names)

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

# Визуализация
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

st.caption("© ComprehensiveNoise, 2025")
