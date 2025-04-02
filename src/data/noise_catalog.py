"""
Полный каталог всех типов шумов с метаданными
"""

# Каталог с полными метаданными для каждого типа шума
NOISE_CATALOG = {
    # 1. СТАТИСТИЧЕСКИЕ ШУМЫ, ОСНОВАННЫЕ НА РАСПРЕДЕЛЕНИЯХ
    "gaussian": {
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
    },
    
    # Здесь должны быть определены все остальные 116 типов шумов
    # Для краткости приведен только один полный пример

    "salt_pepper": {
        "id": "salt_pepper",
        "title": "Шум соль и перец",
        "index": "1.2",
        "category": "statistical",
        "category_id": 1,
        "tags": ["статистический", "импульсный", "дискретный"],
        "short_description": "Случайные черные и белые пиксели на изображении",
        "parameters": [
            {
                "name": "salt_prob", 
                "symbol": "p_s", 
                "default": 0.05, 
                "min": 0, 
                "max": 0.5, 
                "description": "Вероятность белых пикселей (соль)"
            },
            {
                "name": "pepper_prob", 
                "symbol": "p_p", 
                "default": 0.05, 
                "min": 0, 
                "max": 0.5, 
                "description": "Вероятность черных пикселей (перец)"
            }
        ],
        "formulas": {
            "distribution": "P(g(x,y)) = \\begin{cases} p_s, & g(x,y) = 255 \\\\ p_p, & g(x,y) = 0 \\\\ 1 - p_s - p_p, & g(x,y) = f(x,y) \\end{cases}",
            "noise_model": "g(x,y) = \\begin{cases} 255, & \\text{с вероятностью } p_s \\\\ 0, & \\text{с вероятностью } p_p \\\\ f(x,y), & \\text{с вероятностью } 1-p_s-p_p \\end{cases}"
        },
        "related_noise_types": ["gaussian", "impulse", "speckle"]
    },
    
    "poisson": {
        "id": "poisson",
        "title": "Пуассоновский шум",
        "index": "1.3",
        "category": "statistical",
        "category_id": 1,
        "tags": ["статистический", "сигнал-зависимый", "дискретный"],
        "short_description": "Шум, возникающий при низкой интенсивности освещения",
        "parameters": [
            {
                "name": "lambda", 
                "symbol": "\\lambda", 
                "default": 10, 
                "min": 1, 
                "max": 50, 
                "description": "Параметр интенсивности (среднее количество фотонов)"
            }
        ],
        "related_noise_types": ["gaussian", "shot", "quantum"]
    },
    
    # Добавляем определения для других типов шумов...
    # (для полной реализации здесь должны быть все 117 типов шумов)
}

# Категории шумов
NOISE_CATEGORIES = {
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
        "long_description": """
            Цветные шумы определяются своими спектральными свойствами. Название происходит от 
            аналогии с видимым спектром света. Например, белый шум имеет равномерный спектр, 
            подобно белому свету. Розовый шум имеет спектральную плотность, пропорциональную 1/f, 
            коричневый шум - 1/f², и так далее.
        """,
        "related_categories": [1, 3]
    },
    3: {
        "id": 3,
        "title": "Процедурные шумы",
        "short_description": "Шумы, созданные с помощью алгоритмических методов",
        "description": "Процедурные шумы генерируются с помощью специальных алгоритмов.",
        "long_description": """
            Процедурные шумы создаются с помощью алгоритмов и используются для 
            процедурной генерации контента, особенно в компьютерной графике и играх. 
            Они позволяют создавать естественно выглядящие текстуры, ландшафты, облака 
            и другие элементы с контролируемой случайностью.
        """,
        "related_categories": [2, 4]
    },
    4: {
        "id": 4,
        "title": "Физические шумы",
        "short_description": "Шумы, связанные с физическими устройствами",
        "description": "Физические шумы возникают в реальных устройствах и системах.",
        "long_description": """
            Физические шумы связаны с конкретными физическими процессами и устройствами. 
            Они включают шумы сенсоров (CMOS, CCD), тепловой шум, шум квантования, фиксированный 
            шаблонный шум, и другие. Эти шумы важны для моделирования и коррекции в реальных 
            системах обработки изображений и сигналов.
        """,
        "related_categories": [1, 5]
    },
    
    # И так далее для всех 14 категорий...
    # (для полной реализации нужно добавить все категории)
}