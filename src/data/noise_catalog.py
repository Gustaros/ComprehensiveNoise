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
    
    "salt_pepper": {
        "id": "salt_pepper",
        "title": "Шум соль и перец",
        "index": "1.2",
        "category": "statistical",
        "category_id": 1,
        "tags": ["статистический", "импульсный", "дискретный", "бинарный", "точечный"],
        "short_description": "Случайные черные и белые пиксели на изображении",
        "description": """
            Шум "соль и перец" (salt and pepper noise), также известный как импульсный шум,
            характеризуется случайным появлением белых и черных пикселей на изображении.
            Этот тип шума типично возникает из-за ошибок в передаче данных, неисправных элементов
            в сенсорах камер, или проблем с памятью устройств.
            
            В отличие от гауссовского шума, который влияет на все пиксели, шум "соль и перец"
            затрагивает только небольшой процент пикселей, оставляя остальные неизменными.
            Пиксели случайно заменяются либо на минимальное значение (перец, черный),
            либо на максимальное (соль, белый).
        """,
        "distribution_description": "Бинарное распределение с крайними значениями (0 и 255 для 8-битных изображений)",
        "parameters": [
            {
                "name": "amount", 
                "symbol": "p", 
                "default": 0.05, 
                "min": 0, 
                "max": 1, 
                "description": "Общая вероятность шума (доля затронутых пикселей)"
            },
            {
                "name": "salt_vs_pepper", 
                "symbol": "s", 
                "default": 0.5, 
                "min": 0, 
                "max": 1, 
                "description": "Соотношение соли к перцу (0.5 означает равное количество)"
            }
        ],
        "formulas": {
            "distribution": "P(g(x,y) = 255) = p \\cdot s, \\quad P(g(x,y) = 0) = p \\cdot (1-s)",
            "noise_model": "g(x,y) = \\begin{cases} 255, & \\text{с вероятностью } p \\cdot s \\\\ 0, & \\text{с вероятностью } p \\cdot (1-s) \\\\ f(x,y), & \\text{с вероятностью } 1-p \\end{cases}",
            "constraints": "0 \\leq p \\leq 1, \\quad 0 \\leq s \\leq 1"
        },
        "properties": [
            {"name": "Дискретность", "description": "Шум принимает только экстремальные значения (0 или 255)"},
            {"name": "Разреженность", "description": "Затрагивает только малую часть пикселей изображения (обычно <10%)"},
            {"name": "Импульсность", "description": "Локализованные точечные выбросы, не связанные друг с другом"},
            {"name": "Нестационарность", "description": "Случайное распределение в пространстве без корреляции"}
        ],
        "occurrences": [
            {"context": "Цифровая фотография", "description": "Мертвые пиксели в сенсорах камер"},
            {"context": "Передача данных", "description": "Ошибки в битах при передаче изображений"},
            {"context": "Старые изображения", "description": "Повреждения на старых фотографиях или отсканированных документах"},
            {"context": "Аппаратные ошибки", "description": "Сбои в электронике или памяти устройств"},
            {"context": "Тестирование алгоритмов", "description": "Используется для проверки алгоритмов фильтрации и восстановления изображений"}
        ],
        "code": {
            "python": """
import numpy as np

def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5, seed=None):
    \"\"\"
    Добавляет шум 'соль и перец' к изображению.
    
    Параметры:
    ----------
    image : ndarray
        Исходное изображение (в градациях серого или цветное)
    amount : float
        Вероятность шума (доля затронутых пикселей), от 0 до 1
    salt_vs_pepper : float
        Соотношение соли к перцу (0.5 означает равное количество)
    seed : int, optional
        Зерно для генератора случайных чисел
        
    Возвращает:
    -----------
    noisy_img : ndarray
        Изображение с добавленным шумом 'соль и перец'
    noise_pattern : ndarray
        Шумовой паттерн, который был добавлен к изображению
    \"\"\"
    if seed is not None:
        np.random.seed(seed)
        
    noisy_img = np.copy(image)
    noise_pattern = np.zeros_like(image)
    
    # Общее количество пикселей для изменения
    num_pixels = int(amount * image.size)
    
    # Количество пикселей для соли (белых)
    num_salt = int(num_pixels * salt_vs_pepper)
    
    # Координаты для соли (белых пикселей)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_img[tuple(coords)] = 255
    noise_pattern[tuple(coords)] = 255
    
    # Координаты для перца (черных пикселей)
    num_pepper = num_pixels - num_salt
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_img[tuple(coords)] = 0
    noise_pattern[tuple(coords)] = -255
    
    return noisy_img, noise_pattern
            """,
            "javascript": """
/**
 * Добавляет шум 'соль и перец' к изображению
 * @param {ImageData} imageData - Данные исходного изображения
 * @param {number} amount - Вероятность шума (доля затронутых пикселей), от 0 до 1
 * @param {number} saltVsPepper - Соотношение соли к перцу (0.5 означает равное количество)
 * @returns {Object} Объект с зашумленным изображением и шумовым паттерном
 */
function addSaltPepperNoise(imageData, amount = 0.05, saltVsPepper = 0.5) {
    const width = imageData.width;
    const height = imageData.height;
    
    // Создаем выходные данные изображения
    const noisyData = new ImageData(width, height);
    const noisePattern = new ImageData(width, height);
    
    // Копируем исходное изображение
    for (let i = 0; i < imageData.data.length; i++) {
        noisyData.data[i] = imageData.data[i];
        noisePattern.data[i] = 0;
    }
    
    // Общее количество пикселей
    const totalPixels = width * height;
    // Количество пикселей, которые нужно изменить
    const pixelsToChange = Math.floor(totalPixels * amount);
    // Количество 'соли' (белых пикселей)
    const saltPixels = Math.floor(pixelsToChange * saltVsPepper);
    // Количество 'перца' (черных пикселей)
    const pepperPixels = pixelsToChange - saltPixels;
    
    // Добавляем 'соль' (белые пиксели)
    for (let i = 0; i < saltPixels; i++) {
        const x = Math.floor(Math.random() * width);
        const y = Math.floor(Math.random() * height);
        const idx = (y * width + x) * 4;
        
        // Устанавливаем белый цвет (255) для всех каналов
        noisyData.data[idx] = 255;      // R
        noisyData.data[idx+1] = 255;    // G
        noisyData.data[idx+2] = 255;    // B
        
        // Для шумового паттерна
        noisePattern.data[idx] = 255;   // R
        noisePattern.data[idx+1] = 255; // G
        noisePattern.data[idx+2] = 255; // B
        noisePattern.data[idx+3] = 255; // Alpha
    }
    
    // Добавляем 'перец' (черные пиксели)
    for (let i = 0; i < pepperPixels; i++) {
        const x = Math.floor(Math.random() * width);
        const y = Math.floor(Math.random() * height);
        const idx = (y * width + x) * 4;
        
        // Устанавливаем черный цвет (0) для всех каналов
        noisyData.data[idx] = 0;      // R
        noisyData.data[idx+1] = 0;    // G
        noisyData.data[idx+2] = 0;    // B
        
        // Для шумового паттерна
        noisePattern.data[idx] = 0;    // R
        noisePattern.data[idx+1] = 0;  // G
        noisePattern.data[idx+2] = 0;  // B
        noisePattern.data[idx+3] = 255; // Alpha
    }
    
    return {
        noisyImage: noisyData,
        noisePattern: noisePattern
    };
}
            """,
            "matlab": """
function [noisyImage, noisePattern] = addSaltPepperNoise(image, amount, saltVsPepper)
% ADDSALTPEPPERNOISE Добавляет шум 'соль и перец' к изображению
%   [NOISYIMAGE, NOISEPATTERN] = ADDSALTPEPPERNOISE(IMAGE, AMOUNT, SALTVPEPPER) 
%   добавляет шум 'соль и перец' с заданной вероятностью AMOUNT и соотношением
%   SALTVPEPPER к входному изображению IMAGE.
%
%   Входные параметры:
%     - IMAGE: Входное изображение в градациях серого (uint8 или double)
%     - AMOUNT: Вероятность шума (доля затронутых пикселей), от 0 до 1 (по умолчанию: 0.05)
%     - SALTVPEPPER: Соотношение соли к перцу, от 0 до 1 (по умолчанию: 0.5)
%
%   Выходные параметры:
%     - NOISYIMAGE: Изображение с добавленным шумом 'соль и перец'
%     - NOISEPATTERN: Шумовой паттерн, который был добавлен к изображению

if nargin < 2
    amount = 0.05;
end

if nargin < 3
    saltVsPepper = 0.5;
end

% Создаем копию входного изображения
noisyImage = image;
[rows, cols] = size(image);

% Создаем шумовой паттерн
noisePattern = zeros(rows, cols, 'like', image);

% Вычисляем количество пикселей для шума
totalPixels = rows * cols;
numNoise = round(amount * totalPixels);
numSalt = round(saltVsPepper * numNoise);
numPepper = numNoise - numSalt;

% Генерируем случайные координаты для соли (белых пикселей)
saltCoords = randi([1, totalPixels], numSalt, 1);
[saltRows, saltCols] = ind2sub([rows, cols], saltCoords);

% Генерируем случайные координаты для перца (черных пикселей)
pepperCoords = randi([1, totalPixels], numPepper, 1);
[pepperRows, pepperCols] = ind2sub([rows, cols], pepperCoords);

% Устанавливаем значения для соли (255 для uint8)
for i = 1:numSalt
    noisyImage(saltRows(i), saltCols(i)) = 255;
    noisePattern(saltRows(i), saltCols(i)) = 255;
end

% Устанавливаем значения для перца (0)
for i = 1:numPepper
    noisyImage(pepperRows(i), pepperCols(i)) = 0;
    noisePattern(pepperRows(i), pepperCols(i)) = -255;
end

end
            """
        },
        "related_noise_types": ["gaussian", "impulse", "bit_error", "uniform", "binary"],
        "references": [
            {
                "authors": "Gonzalez, R. C., & Woods, R. E.",
                "year": 2018,
                "title": "Digital Image Processing",
                "publisher": "Pearson",
                "edition": "4-е изд."
            },
            {
                "authors": "Chan, R. H., Ho, C. W., & Nikolova, M.",
                "year": 2005,
                "title": "Salt-and-pepper noise removal by median-type noise detectors and detail-preserving regularization",
                "publisher": "IEEE Transactions on Image Processing",
                "volume": "14",
                "issue": "10",
                "pages": "1479-1485"
            },
            {
                "authors": "Hwang, H., & Haddad, R. A.",
                "year": 1995,
                "title": "Adaptive median filters: new algorithms and results",
                "publisher": "IEEE Transactions on Image Processing",
                "volume": "4",
                "issue": "4",
                "pages": "499-502"
            }
        ]
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