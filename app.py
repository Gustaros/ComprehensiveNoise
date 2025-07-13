import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from noise_catalog import noise_functions, create_sample_image, NOISE_CATEGORIES
import inspect
import cv2
from PIL import Image
import io

st.title("Визуализация шумов (ComprehensiveNoise)")

# --- Режим: один шум или галерея ---
mode = st.sidebar.radio("Режим", ["Один шум", "Галерея"], index=0)
category_names = list(NOISE_CATEGORIES.keys())

if mode == "Галерея":
    selected_categories = st.sidebar.multiselect("Фильтрация по категориям", category_names, default=category_names)
    user_file = st.sidebar.file_uploader("Загрузить изображение (PNG/JPG)", type=["png", "jpg", "jpeg"], key="gallery_upload")
    if user_file is not None:
        img = Image.open(user_file).convert("L")
        image = np.array(img).astype(np.float32)
    else:
        img_size = st.sidebar.slider("Размер тестового изображения", 64, 512, 128, step=32, key="gallery_imgsize")
        image = create_sample_image(size=(img_size, img_size), pattern="gradient")
    st.header("Галерея шумов")
    for cat in selected_categories:
        st.subheader(cat)
        noise_names = NOISE_CATEGORIES[cat]
        cols = st.columns(3)
        for idx, noise_name in enumerate(noise_names):
            noise_func = noise_functions[noise_name]
            doc = inspect.getdoc(noise_func) or ""
            doc_lines = doc.splitlines()
            short_desc = doc_lines[0] if doc_lines else ""
            formula = ""
            for l in doc_lines[1:4]:
                if "=" in l or "Formula" in l or "PDF" in l or "g(x,y)" in l:
                    formula = l.strip()
                    break
            # Миниатюра
            try:
                noisy_img, _ = noise_func(image, **{k: v.default for k, v in inspect.signature(noise_func).parameters.items() if k != "image"})
                thumb = np.clip(noisy_img, 0, 255).astype(np.uint8)
            except Exception:
                thumb = np.zeros_like(image)
            with cols[idx % 3]:
                st.image(thumb, caption=noise_name, use_column_width=True, clamp=True)
                st.markdown(f"**Описание:** {short_desc}")
                if formula:
                    st.markdown(f"<span style='font-size:0.9em;color:#666;'>Формула: <code>{formula}</code></span>", unsafe_allow_html=True)
    st.caption("Выберите 'Один шум' для подробной настройки параметров.")
    st.stop()

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
user_file = st.sidebar.file_uploader("Загрузить изображение (PNG/JPG)", type=["png", "jpg", "jpeg"])

if user_file is not None:
    img = Image.open(user_file).convert("L")
    image = np.array(img).astype(np.float32)
    st.info(f"Используется пользовательское изображение: {img.size[0]}x{img.size[1]}")
else:
    img_size = st.sidebar.slider("Размер тестового изображения", 64, 512, 256, step=32)
    image = create_sample_image(size=(img_size, img_size), pattern="gradient")

# Генерируем интерактивные виджеты для параметров шума (кроме image)
with st.sidebar.expander("Параметры шума", expanded=True):
    minmax_cache = {}
    for name, param in params.items():
        if name == "image":
            continue
        ann = param.annotation
        default = param.default
        # Слайдер для вероятностей
        if name in {"p", "prob", "probability", "salt_prob", "pepper_prob"}:
            param_values[name] = st.slider(name, 0.0, 1.0, float(default), step=0.01)
        # Слайдер для mean, sigma, intensity, scale, loc, var (>=0)
        elif name in {"mean", "sigma", "intensity", "scale", "loc", "var", "std", "stddev"}:
            minv = 0.0
            maxv = float(default)*4 if float(default) > 0 else 100.0
            if isinstance(default, float):
                param_values[name] = st.slider(name, float(minv), float(maxv), float(default), step=0.01)
            else:
                param_values[name] = st.slider(name, int(minv), int(maxv), int(default), step=1)
        # min/max как слайдеры с кэшированием
        elif name in {"min", "low"}:
            minv = -255.0
            maxv = float(param.default) if isinstance(param.default, (int, float)) else 0.0
            minmax_cache['min'] = st.slider(name, minv, maxv if maxv > minv else minv+1, float(default), step=1.0)
            param_values[name] = minmax_cache['min']
        elif name in {"max", "high"}:
            minv = minmax_cache.get('min', float(default)-1)
            maxv = 255.0
            param_values[name] = st.slider(name, minv+1, maxv, float(default), step=1.0)
        # Для int с ограничениями
        elif ann is int or (isinstance(default, int) and not isinstance(default, bool)):
            minv = 0 if name in {"n", "M", "N", "df", "dfn", "dfd"} else -1000
            maxv = 1000
            param_values[name] = st.slider(name, minv, maxv, int(default), step=1)
        # Для float с ограничениями
        elif ann is float or isinstance(default, float):
            minv = 0.0 if name in {"alpha", "beta", "a", "b", "kappa"} else -1000.0
            maxv = 1000.0
            param_values[name] = st.number_input(name, value=float(default), step=0.01, format="%.3f", min_value=minv, max_value=maxv)
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
