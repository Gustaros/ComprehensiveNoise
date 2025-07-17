import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from noise_catalog import noise_functions, create_sample_image, NOISE_CATEGORIES
import inspect
import cv2
from PIL import Image
import io
from localization import LANGUAGES, TRANSLATIONS, NOISE_LABELS

# Выбор языка
lang = st.sidebar.selectbox('Language / Язык', list(LANGUAGES.keys()), format_func=lambda k: LANGUAGES[k], key='lang', index=1)
T = lambda key: TRANSLATIONS.get(key, {}).get(lang, key)

st.title(T('title'))

# --- Режим: один шум или галерея ---
mode = st.sidebar.radio(T('mode'), [T('mode_one'), T('mode_gallery')], index=0)
category_names = list(NOISE_CATEGORIES.keys())
category_labels = [NOISE_LABELS.get(cat, {}).get(lang, cat) for cat in category_names]

if mode == T('mode_gallery'):
    selected_categories = st.sidebar.multiselect(T('gallery_filter'), category_names, default=category_names, format_func=lambda k: NOISE_LABELS.get(k, {}).get(lang, k))
    user_file = st.sidebar.file_uploader(T('upload'), type=["png", "jpg", "jpeg"], key="gallery_upload")
    if user_file is not None:
        img = Image.open(user_file).convert("L")
        image = np.array(img).astype(np.float32)
    else:
        img_size = st.sidebar.slider(T('imgsize'), 64, 512, 128, step=32, key="gallery_imgsize")
        image = create_sample_image(size=(img_size, img_size), pattern="gradient")
    st.header(T('gallery_header'))
    for cat in selected_categories:
        st.subheader(NOISE_LABELS.get(cat, {}).get(lang, cat))
        noise_names = NOISE_CATEGORIES[cat]
        noise_labels = [NOISE_LABELS.get(n, {}).get(lang, n) for n in noise_names]
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
            try:
                noisy_img, _ = noise_func(image, **{k: v.default for k, v in inspect.signature(noise_func).parameters.items() if k != "image"})
                thumb = np.clip(noisy_img, 0, 255).astype(np.uint8)
            except Exception:
                thumb = np.zeros_like(image)
            with cols[idx % 3]:
                st.image(thumb, caption=NOISE_LABELS.get(noise_name, {}).get(lang, noise_name), use_container_width=True, clamp=True)
                st.markdown(f"**{T('desc')}:** {short_desc}")
                if formula:
                    st.markdown(f"<span style='font-size:0.9em;color:#666;'>{T('formula')}: <code>{formula}</code></span>", unsafe_allow_html=True)
    st.caption(T('gallery_hint'))
    st.stop()

# Сайдбар: выбор категории и шума
category = st.sidebar.selectbox(T('category'), category_names, format_func=lambda k: NOISE_LABELS.get(k, {}).get(lang, k))
noise_names = NOISE_CATEGORIES[category]
selected_noise = st.sidebar.selectbox(T('type'), noise_names, format_func=lambda k: NOISE_LABELS.get(k, {}).get(lang, k))

noise_func = noise_functions[selected_noise]
doc = inspect.getdoc(noise_func)
sig = inspect.signature(noise_func)
params = sig.parameters
param_values = {}

st.markdown(f"### {NOISE_LABELS.get(selected_noise, {}).get(lang, selected_noise)}")
# if doc:
#     st.markdown(f"**{T('desc_formula')}:**\n```{doc}```", unsafe_allow_html=True)

# Параметры изображения
user_file = st.sidebar.file_uploader(T('upload'), type=["png", "jpg", "jpeg"])

if user_file is not None:
    img = Image.open(user_file).convert("L")
    image = np.array(img).astype(np.float32)
    st.info(f"{T('upload')}: {img.size[0]}x{img.size[1]}")
else:
    img_size = st.sidebar.slider(T('imgsize'), 64, 512, 256, step=32)
    image = create_sample_image(size=(img_size, img_size), pattern="gradient")

# Генерируем интерактивные виджеты для параметров шума (кроме image)
with st.sidebar.expander(T('params'), expanded=True):
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

# Визуализация изображений через st.image
col1, col2, col3 = st.columns(3)
with col1:
    st.image(np.clip(image, 0, 255).astype(np.uint8), caption=T('original'), use_container_width=True, clamp=True)
with col2:
    st.image(np.clip(noisy_img, 0, 255).astype(np.uint8), caption=T('with_noise'), use_container_width=True, clamp=True)
with col3:
    # Для шума используем viridis, преобразуем к RGB для наглядности
    import matplotlib.cm as cm
    norm_noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    noise_rgb = (cm.viridis(norm_noise)[:, :, :3] * 255).astype(np.uint8)
    st.image(noise_rgb, caption=T('noise') + ' (viridis)', use_container_width=True, clamp=True)

# Визуализация гистограмм (каждая отдельно через st.image)
def plot_hist(arr, color, title):
    import io
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(arr.ravel(), bins=64, color=color)
    ax.set_title(title)
    ax.set_xlabel(T('xlabel'))
    ax.set_ylabel(T('ylabel'))
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

colh1, colh2, colh3 = st.columns(3)
with colh1:
    st.image(plot_hist(image, 'gray', T('hist_orig')), caption=T('hist_orig'), use_container_width=True)
with colh2:
    st.image(plot_hist(noisy_img, 'blue', T('hist_noisy')), caption=T('hist_noisy'), use_container_width=True)
with colh3:
    st.image(plot_hist(noise, 'purple', T('hist_noise')), caption=T('hist_noise'), use_container_width=True)

# --- Генерация кода для выбранного шума ---
st.markdown(f"#### {T('source_code')}")
try:
    code_str = inspect.getsource(noise_func)
    st.code(code_str, language="python")
except Exception:
    st.warning(T('source_code_error'))

st.caption(T('caption'))
