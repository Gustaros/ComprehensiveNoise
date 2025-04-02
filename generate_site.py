#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для генерации статического сайта с каталогом шумов
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, ndimage, signal
from skimage import util, exposure
import base64
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
import datetime
import argparse
import markdown
from PIL import Image

# Конфигурация
OUTPUT_DIR = "noise_catalog_site"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
TEST_IMAGES_DIR = f"{STATIC_DIR}/images/test-images"
THUMBNAILS_DIR = f"{STATIC_DIR}/images/thumbnails"
DEMOS_DIR = f"{STATIC_DIR}/js"

# Путь к файлу с метаданными шумов
NOISE_CATALOG_FILE = "src/data/noise_catalog.py"

# Создаем директории, если они не существуют
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/css", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/js", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/test-images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/thumbnails", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/noise", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/category", exist_ok=True)

# Настройка Jinja2
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Загрузка метаданных о шумах (фиктивная функция, здесь нужно реальное импортирование из файла)
def load_noise_catalog():
    # В реальности здесь должен быть импорт из NOISE_CATALOG_FILE
    # Для демонстрации создаем простой каталог с гауссовским шумом
    
    gaussian_noise = {
        "id": "gaussian",
        "title": "Гауссовский шум",
        "index": "1.1",
        "category": "statistical",
        "category_id": 1,
        "tags": ["статистический", "непрерывный", "аддитивный"],
        "short_description": "Наиболее распространенный тип шума с нормальным распределением",
        "description": """
            Гауссовский шум — это статистический шум, имеющий функцию плотности вероятности, 
            равную нормальному распределению. Значения шума, которые этот шум может принимать, 
            распределены по нормальному закону. Это наиболее распространенный тип шума в 
            цифровой обработке изображений, который возникает из-за теплового шума электронных 
            компонентов, несовершенства сенсоров, и других факторов.
        """,
        "distribution_description": "Нормальное (гауссовское) распределение с функцией плотности вероятности:",
        "parameters": [
            {
                "name": "mean", 
                "symbol": "\\mu", 
                "default": 0, 
                "min": -25, 
                "max": 25, 
                "description": "Среднее значение (математическое ожидание)"
            },
            {
                "name": "sigma", 
                "symbol": "\\sigma", 
                "default": 25, 
                "min": 1, 
                "max": 50, 
                "description": "Стандартное отклонение (корень из дисперсии)"
            }
        ],
        "formulas": {
            "distribution": "f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}",
            "noise_model": "g(x,y) = f(x,y) + \\eta(x,y)",
            "constraints": "\\eta(x,y) \\sim N(\\mu, \\sigma^2)"
        },
        "properties": [
            {"name": "Симметричность", "description": "Распределение симметрично относительно среднего значения"},
            {"name": "Стационарность", "description": "Статистические свойства не меняются в пространстве"},
            {"name": "Независимость", "description": "Шум в каждом пикселе независим от других"},
            {"name": "Спектральный", "description": "Равная мощность на всех частотах (белый шум)"}
        ],
        "occurrences": [
            {"context": "Электронные устройства", "description": "Тепловой шум в сенсорах и электронных компонентах"},
            {"context": "Цифровые камеры", "description": "При низкой освещенности или высоких настройках ISO"},
            {"context": "Медицинская визуализация", "description": "МРТ, КТ и другие методы медицинской визуализации"},
            {"context": "Системы связи", "description": "Фоновый шум в каналах передачи данных"},
            {"context": "Научные приборы", "description": "Погрешности измерений в различных инструментах"}
        ],
        "code": {
            "python": """
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    \"\"\"
    Добавляет гауссовский шум к изображению.
    
    Параметры:
    ----------
    image : ndarray
        Исходное изображение (в градациях серого или цветное)
    mean : float
        Среднее значение распределения Гаусса
    sigma : float
        Стандартное отклонение распределения Гаусса
        
    Возвращает:
    --------
    noisy_img : ndarray
        Изображение с добавленным гауссовским шумом
    noise : ndarray
        Шумовой паттерн, который был добавлен к изображению
    \"\"\"
    row, col = image.shape
    noise = np.random.normal(mean, sigma, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise
            """,
            "javascript": """
/**
 * Добавляет гауссовский шум к изображению
 * @param {ImageData} imageData - Данные исходного изображения
 * @param {number} mean - Среднее значение распределения Гаусса
 * @param {number} stdDev - Стандартное отклонение распределения Гаусса
 * @returns {Object} Объект с зашумленным изображением и шумовым паттерном
 */
function addGaussianNoise(imageData, mean = 0, stdDev = 25) {
    const width = imageData.width;
    const height = imageData.height;
    
    // Создаем выходные данные изображения
    const noisyData = new ImageData(width, height);
    const noisePattern = new ImageData(width, height);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            
            // Генерируем гауссовский шум для этого пикселя
            const noise = generateGaussianRandom(mean, stdDev);
            
            // Сохраняем шумовой паттерн (одинаковое значение для R,G,B для визуализации в градациях серого)
            noisePattern.data[i] = noise + 128;      // R
            noisePattern.data[i+1] = noise + 128;    // G
            noisePattern.data[i+2] = noise + 128;    // B
            noisePattern.data[i+3] = 255;            // Alpha
            
            // Добавляем шум к исходному изображению
            noisyData.data[i] = Math.min(255, Math.max(0, imageData.data[i] + noise));    // R
            noisyData.data[i+1] = Math.min(255, Math.max(0, imageData.data[i+1] + noise)); // G
            noisyData.data[i+2] = Math.min(255, Math.max(0, imageData.data[i+2] + noise)); // B
            noisyData.data[i+3] = imageData.data[i+3]; // Alpha (без изменений)
        }
    }
    
    return {
        noisyImage: noisyData,
        noisePattern: noisePattern
    };
}

/**
 * Генерирует случайное число с гауссовским распределением
 * @param {number} mean - Среднее значение распределения
 * @param {number} stdDev - Стандартное отклонение распределения
 * @returns {number} Случайное значение из гауссовского распределения
 */
function generateGaussianRandom(mean, stdDev) {
    // Преобразование Бокса-Мюллера
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + stdDev * z;
}
            """,
            "matlab": """
function [noisyImage, noisePattern] = addGaussianNoise(image, mean, sigma)
% ADDGAUSSIANNOISE Добавляет гауссовский шум к изображению
%   [NOISYIMAGE, NOISEPATTERN] = ADDGAUSSIANNOISE(IMAGE, MEAN, SIGMA) добавляет
%   гауссовский шум с заданным MEAN и SIGMA к входному изображению IMAGE.
%   Возвращает зашумленное изображение и шумовой паттерн, который был добавлен.
%
%   Входные параметры:
%     - IMAGE: Входное изображение в градациях серого (uint8 или double)
%     - MEAN: Среднее значение распределения Гаусса (по умолчанию: 0)
%     - SIGMA: Стандартное отклонение распределения Гаусса (по умолчанию: 25)
%
%   Выходные параметры:
%     - NOISYIMAGE: Изображение с добавленным гауссовским шумом
%     - NOISEPATTERN: Шумовой паттерн, который был добавлен к изображению

if nargin < 2
    mean = 0;
end

if nargin < 3
    sigma = 25;
end

% Преобразуем в double для вычислений
if ~isa(image, 'double')
    image = double(image);
end

% Генерируем шум с заданным средним и стандартным отклонением
[rows, cols] = size(image);
noisePattern = randn(rows, cols) * sigma + mean;

% Добавляем шум к изображению
noisyImage = image + noisePattern;

% Ограничиваем значения допустимым диапазоном [0, 255]
noisyImage = min(max(noisyImage, 0), 255);

% Преобразуем обратно в uint8, если входное изображение было uint8
if isa(image, 'uint8')
    noisyImage = uint8(noisyImage);
end

end
            """
        },
        "related_noise_types": ["white", "poisson", "speckle"],
        "references": [
            {
                "authors": "Gonzalez, R. C., & Woods, R. E.",
                "year": 2018,
                "title": "Digital Image Processing",
                "publisher": "Pearson",
                "edition": "4-е изд."
            },
            {
                "authors": "Bovik, A.",
                "year": 2010,
                "title": "Handbook of Image and Video Processing",
                "publisher": "Academic Press"
            },
            {
                "authors": "Smith, J. et al.",
                "year": 2016,
                "title": "Noise in Electronic Imaging Systems",
                "publisher": "Journal of Electronic Imaging",
                "volume": "25",
                "issue": "1"
            }
        ]
    }
    
    # Добавляем больше типов шумов (заглушки)
    noise_catalog = {
        "gaussian": gaussian_noise,
        "white": {
            "id": "white",
            "title": "Белый шум",
            "short_description": "Шум с равномерным спектром мощности",
            "index": "2.1",
            "category": "colored",
            "category_id": 2,
            "tags": ["цветной", "спектральный"]
        },
        "poisson": {
            "id": "poisson",
            "title": "Пуассоновский шум",
            "short_description": "Зависящий от сигнала шум для низких уровней освещенности",
            "index": "1.3",
            "category": "statistical",
            "category_id": 1,
            "tags": ["статистический", "дискретный", "сигнал-зависимый"]
        },
        "speckle": {
            "id": "speckle",
            "title": "Спекл-шум",
            "short_description": "Мультипликативный шум с гауссовскими характеристиками",
            "index": "1.4",
            "category": "statistical",
            "category_id": 1,
            "tags": ["статистический", "мультипликативный"]
        }
    }
    
    return noise_catalog

# Загрузка категорий шумов
def load_categories():
    categories = {
        1: {
            "id": 1,
            "title": "Статистические шумы",
            "short_description": "Шумы, основанные на статистических распределениях",
            "description": "Статистические шумы основаны на известных распределениях вероятностей.",
            "long_description": """
                Статистические шумы характеризуются определенными распределениями вероятностей, 
                такими как нормальное распределение (гауссовский шум), распределение Пуассона, 
                равномерное распределение и другие. Эти шумы широко используются в обработке сигналов 
                и изображений для моделирования различных физических явлений и помех.
            """,
            "applications": [
                {"field": "Обработка изображений", "description": "Моделирование шумов сенсоров"},
                {"field": "Медицинская визуализация", "description": "Анализ и устранение шумов в МРТ и КТ"},
                {"field": "Системы связи", "description": "Моделирование помех в каналах передачи данных"}
            ],
            "related_categories": [2, 3]
        },
        2: {
            "id": 2,
            "title": "Цветные шумы",
            "short_description": "Шумы с различными спектральными характеристиками",
            "description": "Цветные шумы имеют различные спектральные характеристики мощности.",
            "related_categories": [1, 3]
        },
        3: {
            "id": 3,
            "title": "Процедурные шумы",
            "short_description": "Шумы, созданные с помощью алгоритмических методов",
            "description": "Процедурные шумы генерируются с помощью специальных алгоритмов.",
            "related_categories": [2, 4]
        },
        4: {
            "id": 4,
            "title": "Физические шумы",
            "short_description": "Шумы, связанные с физическими устройствами",
            "description": "Физические шумы возникают в реальных устройствах и системах.",
            "related_categories": [1, 3]
        }
    }
    
    return categories

# Функция для создания тестового изображения
def create_test_image(size=(256, 256), pattern="gradient"):
    if pattern == "gradient":
        x = np.linspace(0, 1, size[1])
        y = np.linspace(0, 1, size[0])
        xx, yy = np.meshgrid(x, y)
        image = np.clip(xx * yy * 255, 0, 255).astype(np.uint8)
    elif pattern == "circles":
        x = np.linspace(-1, 1, size[1])
        y = np.linspace(-1, 1, size[0])
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        image = np.clip((np.sin(r * 15) + 1) * 127.5, 0, 255).astype(np.uint8)
    elif pattern == "constant":
        image = np.ones(size, dtype=np.uint8) * 128
    else:
        image = np.random.randint(0, 256, size=size, dtype=np.uint8)
    
    return image

# Функция для генерации гауссовского шума (как пример)
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    noise = np.random.normal(mean, sigma, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# Функция для генерации миниатюр и визуализаций для каждого типа шума
def generate_noise_visuals(noise_catalog):
    # Создаем тестовое изображение
    test_image = create_test_image(pattern="gradient")
    
    # Сохраняем тестовое изображение
    plt.imsave(f"{OUTPUT_DIR}/images/test-images/test_image.png", test_image, cmap='gray')
    
    # Для каждого типа шума создаем визуализацию
    for noise_id, noise in noise_catalog.items():
        try:
            # В реальности здесь нужно вызывать соответствующую функцию для каждого типа шума
            # Для демонстрации используем гауссовский шум для всех типов
            noisy_image, noise_pattern = add_gaussian_noise(test_image)
            
            # Создаем визуализацию
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(test_image, cmap='gray')
            axes[0].set_title('Исходное изображение')
            axes[0].axis('off')
            
            axes[1].imshow(noisy_image, cmap='gray')
            axes[1].set_title(f'С шумом {noise["title"]}')
            axes[1].axis('off')
            
            im = axes[2].imshow(noise_pattern, cmap='viridis')
            axes[2].set_title('Шумовой паттерн')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Сохраняем миниатюру
            plt.savefig(f"{OUTPUT_DIR}/images/thumbnails/{noise_id}.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Созданы визуализации для {noise['title']}")
            
        except Exception as e:
            print(f"Ошибка при создании визуализаций для {noise_id}: {e}")

# Функция для генерации HTML-страницы для каждого типа шума
def generate_noise_pages(noise_catalog, categories):
    template = env.get_template('noise_page.html')
    
    for noise_id, noise in noise_catalog.items():
        try:
            html = template.render(
                noise=noise,
                noise_catalog=noise_catalog,
                categories=categories,
                user="Gustaros",
                generation_date="2025-04-02 20:48:52"
            )
            
            with open(f"{OUTPUT_DIR}/noise/{noise_id}.html", 'w', encoding='utf-8') as f:
                f.write(html)
                
            print(f"Создана страница для {noise['title']}")
            
        except Exception as e:
            print(f"Ошибка при создании страницы для {noise_id}: {e}")

# Функция для генерации HTML-страниц для категорий
def generate_category_pages(categories, noise_catalog):
    template = env.get_template('category.html')
    
    for category_id, category in categories.items():
        try:
            # Находим все шумы в этой категории
            category_noises = [noise for noise_id, noise in noise_catalog.items() 
                               if noise.get('category_id') == category_id]
            
            # Сортируем шумы по индексу
            category_noises.sort(key=lambda x: x.get('index', '0'))
            
            html = template.render(
                category=category,
                category_noises=category_noises,
                categories=categories,
                user="Gustaros",
                generation_date="2025-04-02 20:48:52"
            )
            
            with open(f"{OUTPUT_DIR}/category/{category_id}.html", 'w', encoding='utf-8') as f:
                f.write(html)
                
            print(f"Создана страница для категории {category['title']}")
            
        except Exception as e:
            print(f"Ошибка при создании страницы для категории {category_id}: {e}")

# Функция для генерации главной страницы
def generate_index_page(noise_catalog, categories):
    template = env.get_template('index.html')
    
    # Выбираем избранные шумы для главной страницы
    featured_noises = ['gaussian', 'white', 'poisson', 'speckle']
    
    # Для каждой категории выбираем несколько шумов
    category_top_noises = {}
    for category_id in categories:
        category_noises = [noise for noise_id, noise in noise_catalog.items() 
                           if noise.get('category_id') == category_id]
        category_top_noises[category_id] = category_noises[:3]  # первые 3 шума из категории
    
    html = template.render(
        featured_noises=featured_noises,
        noise_catalog=noise_catalog,
        categories=categories,
        category_top_noises=category_top_noises,
        user="Gustaros",
        generation_date="2025-04-02 20:48:52"
    )
    
    with open(f"{OUTPUT_DIR}/index.html", 'w', encoding='utf-8') as f:
        f.write(html)
        
    print("Создана главная страница")

# Функция для копирования статических файлов
def copy_static_files():
    # Копируем CSS файлы
    for css_file in ['main.css', 'noise-page.css']:
        shutil.copy(f"{STATIC_DIR}/css/{css_file}", f"{OUTPUT_DIR}/css/{css_file}")
    
    # Копируем JS файлы
    for js_file in ['main.js', 'noise-generator.js', 'visualizations.js', 'gaussian-noise-demo.js']:
        shutil.copy(f"{STATIC_DIR}/js/{js_file}", f"{OUTPUT_DIR}/js/{js_file}")
    
    print("Статические файлы скопированы")

# Основная функция
def main():
    print("Генерация статического сайта для каталога шумов...")
    
    # Загружаем метаданные о шумах
    noise_catalog = load_noise_catalog()
    
    # Загружаем категории шумов
    categories = load_categories()
    
    # Генерируем визуализации для каждого типа шума
    generate_noise_visuals(noise_catalog)
    
    # Генерируем HTML-страницы для каждого типа шума
    generate_noise_pages(noise_catalog, categories)
    
    # Генерируем HTML-страницы для категорий
    generate_category_pages(categories, noise_catalog)
    
    # Генерируем главную страницу
    generate_index_page(noise_catalog, categories)
    
    # Копируем статические файлы
    copy_static_files()
    
    print(f"Сайт успешно сгенерирован в директории {OUTPUT_DIR}")

if __name__ == "__main__":
    main()