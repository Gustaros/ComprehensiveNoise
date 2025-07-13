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

noise_func = noise_functions[selected_noise]
doc = inspect.getdoc(noise_func)
sig = inspect.signature(noise_func)
params = sig.parameters
param_values = {}

st.markdown(f"### {selected_noise}")
if doc:
    st.markdown(f"**Описание и формула:**\n```{doc}```", unsafe_allow_html=True)

# Параметры изображения
img_size = st.sidebar.slider("Размер изображения", 64, 512, 256, step=32)
image = create_sample_image(size=(img_size, img_size), pattern="gradient")

# Генерируем интерактивные виджеты для параметров шума (кроме image)
with st.sidebar.expander("Параметры шума", expanded=True):
    for name, param in params.items():
        if name == "image":
            continue
        ann = param.annotation
        default = param.default
        if ann is int or (isinstance(default, int) and not isinstance(default, bool)):
            param_values[name] = st.number_input(name, value=default, step=1)
        elif ann is float or isinstance(default, float):
            param_values[name] = st.number_input(name, value=default, step=0.01, format="%.3f")
        elif ann is bool or isinstance(default, bool):
            param_values[name] = st.checkbox(name, value=default)
        elif ann is tuple or isinstance(default, tuple):
            param_str = st.text_input(name, value=str(default))
            try:
                param_values[name] = eval(param_str)
            except Exception:
                param_values[name] = default
        else:
            param_values[name] = st.text_input(name, value=str(default))

# Применяем шум с выбранными параметрами
try:
    noisy_img, noise = noise_func(image, **param_values)
except Exception as e:
    st.error(f"Ошибка при применении шума: {e}")
    noisy_img = image.copy()
    noise = np.zeros_like(image)

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
