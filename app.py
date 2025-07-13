import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from noise_catalog import noise_functions, create_sample_image, NOISE_CATEGORIES
import inspect

st.title("Визуализация шумов (ComprehensiveNoise)")

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
