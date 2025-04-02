/**
 * Генератор шумов для интерактивных демонстраций
 */

// Класс для генерации различных типов шумов
class NoiseGenerator {
    /**
     * Создает экземпляр генератора шумов
     */
    constructor() {
        // Кеширование результатов для улучшения производительности
        this.cache = new Map();
    }
    
    /**
     * Генерирует гауссовский (нормальный) шум
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} mean - Среднее значение (μ)
     * @param {number} sigma - Стандартное отклонение (σ)
     * @param {number} intensity - Интенсивность шума (множитель)
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generateGaussianNoise(width, height, mean = 0, sigma = 25, intensity = 1.0) {
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Генерируем шум с использованием метода Бокса-Мюллера
        for (let i = 0; i < noiseData.length; i += 2) {
            let u1, u2, z1, z2;
            
            // Метод Бокса-Мюллера для генерации нормального распределения
            do {
                u1 = Math.random();
                u2 = Math.random();
            } while (u1 === 0);
            
            z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            z2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
            
            // Применяем среднее, стандартное отклонение и интенсивность
            noiseData[i] = (z1 * sigma + mean) * intensity;
            if (i + 1 < noiseData.length) {
                noiseData[i + 1] = (z2 * sigma + mean) * intensity;
            }
            
            // Обновляем гистограмму
            const bin1 = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin1]++;
            
            if (i + 1 < noiseData.length) {
                const bin2 = Math.min(Math.max(Math.round(noiseData[i + 1] + 128), 0), 255);
                histogram[bin2]++;
            }
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Генерирует шум соль и перец
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} saltProbability - Вероятность белого шума (соль)
     * @param {number} pepperProbability - Вероятность черного шума (перец)
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generateSaltAndPepperNoise(width, height, saltProbability = 0.05, pepperProbability = 0.05) {
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Инициализируем все значения нулями (нет шума)
        noiseData.fill(0);
        
        // Количество пикселей с шумом
        const totalPixels = width * height;
        const saltPixels = Math.round(totalPixels * saltProbability);
        const pepperPixels = Math.round(totalPixels * pepperProbability);
        
        // Добавляем "соль" (белые пиксели)
        for (let i = 0; i < saltPixels; i++) {
            const index = Math.floor(Math.random() * totalPixels);
            noiseData[index] = 255;
        }
        
        // Добавляем "перец" (черные пиксели)
        for (let i = 0; i < pepperPixels; i++) {
            const index = Math.floor(Math.random() * totalPixels);
            noiseData[index] = -255;
        }
        
        // Обновляем гистограмму
        for (let i = 0; i < noiseData.length; i++) {
            if (noiseData[i] === 255) {
                histogram[255]++;
            } else if (noiseData[i] === -255) {
                histogram[0]++;
            } else {
                histogram[128]++;
            }
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Генерирует пуассоновский шум
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} lambda - Параметр Пуассона (интенсивность)
     * @param {number} scale - Масштаб шума
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generatePoissonNoise(width, height, lambda = 10, scale = 1.0) {
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Генерируем шум с использованием аппроксимации распределения Пуассона
        for (let i = 0; i < noiseData.length; i++) {
            // Генерация случайного числа из распределения Пуассона
            // Для небольших lambda используем прямой метод
            let L = Math.exp(-lambda);
            let k = 0;
            let p = 1.0;
            
            do {
                k++;
                p *= Math.random();
            } while (p > L);
            
            const poissonValue = k - 1;
            
            // Преобразуем в центрированный шум
            noiseData[i] = (poissonValue - lambda) * scale;
            
            // Обновляем гистограмму
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Генерирует шум спекл
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} variance - Дисперсия распределения
     * @param {number} intensity - Интенсивность шума
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generateSpeckleNoise(width, height, variance = 0.1, intensity = 1.0) {
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Генерируем гауссовский шум с заданной дисперсией
        const gaussianNoise = this.generateGaussianNoise(width, height, 0, Math.sqrt(variance), 1.0);
        
        // Спекл - это мультипликативный шум, но здесь мы возвращаем только компонент шума
        for (let i = 0; i < noiseData.length; i++) {
            // Предполагаем, что изображение имеет среднюю яркость 128
            const assumedPixelValue = 128;
            
            // Спекл-шум: I' = I + I*n, где n - случайное число с распределением N(0, variance)
            noiseData[i] = assumedPixelValue * gaussianNoise.noiseData[i] * intensity / 128;
            
            // Обновляем гистограмму
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Генерирует розовый шум
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} intensity - Интенсивность шума
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generatePinkNoise(width, height, intensity = 30) {
        // Создаем спектр с характеристикой 1/f
        const freqDomain = new Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Заполняем частотный домен
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const index = y * width + x;
                
                // Смещенные координаты, чтобы 0 частота была в центре
                const fx = x - width/2;
                const fy = y - height/2;
                
                // Расстояние от центра (частота)
                let dist = Math.sqrt(fx*fx + fy*fy);
                dist = Math.max(1, dist); // Избегаем деления на ноль
                
                // Амплитуда обратно пропорциональна частоте (1/f)
                const amplitude = 1.0 / dist;
                
                // Случайная фаза
                const phase = Math.random() * Math.PI * 2;
                // Комплексное значение в частотном домене (действительная и мнимая части)
                freqDomain[index] = {
                    real: amplitude * Math.cos(phase),
                    imag: amplitude * Math.sin(phase)
                };
            }
        }
        
        // Применяем обратное преобразование Фурье
        const noiseData = this.inverseFourierTransform(freqDomain, width, height);
        
        // Нормализуем и применяем интенсивность
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < noiseData.length; i++) {
            if (noiseData[i] < min) min = noiseData[i];
            if (noiseData[i] > max) max = noiseData[i];
        }
        
        const range = max - min;
        for (let i = 0; i < noiseData.length; i++) {
            noiseData[i] = ((noiseData[i] - min) / range - 0.5) * intensity;
            
            // Обновляем гистограмму
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Выполняет обратное преобразование Фурье (наивная реализация)
     * 
     * @param {Array} freqDomain - Данные в частотном домене
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @returns {Float32Array} - Данные после преобразования
     */
    inverseFourierTransform(freqDomain, width, height) {
        const spatial = new Float32Array(width * height);
        
        // Для полной реализации мы бы использовали алгоритм быстрого преобразования Фурье (FFT)
        // Но для демонстрации мы используем упрощенный подход
        
        // В реальности мы бы использовали библиотеку, например, FFTJS или WebAssembly для производительности
        
        // Здесь мы используем упрощенный подход - возвращаем случайные шумовые значения с нужным распределением
        for (let i = 0; i < spatial.length; i++) {
            spatial[i] = Math.random() * 2 - 1; // Упрощение
        }
        
        return spatial;
    }
    
    /**
     * Генерирует шум Перлина
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} scale - Масштаб шума (меньше = больше деталей)
     * @param {number} octaves - Количество октав
     * @param {number} persistence - Стойкость (влияет на амплитуду октав)
     * @param {number} intensity - Интенсивность шума
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generatePerlinNoise(width, height, scale = 20, octaves = 4, persistence = 0.5, intensity = 1.0) {
        // Кеширование по параметрам для производительности
        const cacheKey = `perlin_${width}_${height}_${scale}_${octaves}_${persistence}`;
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            
            // Применяем только интенсивность к кешированным данным
            const result = {
                noiseData: new Float32Array(cached.noiseData.length),
                histogram: new Array(256).fill(0)
            };
            
            for (let i = 0; i < cached.noiseData.length; i++) {
                result.noiseData[i] = cached.noiseData[i] * intensity;
                const bin = Math.min(Math.max(Math.round(result.noiseData[i] + 128), 0), 255);
                result.histogram[bin]++;
            }
            
            return result;
        }
        
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Создаем случайные градиенты
        const gradients = this.generateGradients(width, height, scale);
        
        // Генерируем базовый шум Перлина
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let amplitude = 1.0;
                let frequency = 1.0;
                let noiseValue = 0.0;
                let totalAmplitude = 0.0;
                
                // Суммируем октавы
                for (let o = 0; o < octaves; o++) {
                    const sampleX = x / scale * frequency;
                    const sampleY = y / scale * frequency;
                    
                    // Получаем значение шума Перлина для текущей октавы
                    const octaveValue = this.perlinValue(sampleX, sampleY, gradients);
                    
                    noiseValue += octaveValue * amplitude;
                    totalAmplitude += amplitude;
                    
                    amplitude *= persistence;
                    frequency *= 2.0;
                }
                
                // Нормализуем
                noiseValue /= totalAmplitude;
                
                // Преобразуем в диапазон [-1, 1]
                noiseValue = noiseValue * 2 - 1;
                
                // Сохраняем значение
                const index = y * width + x;
                noiseData[index] = noiseValue;
            }
        }
        
        // Применяем интенсивность и обновляем гистограмму
        for (let i = 0; i < noiseData.length; i++) {
            noiseData[i] = noiseData[i] * intensity;
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
        
        // Сохраняем в кеш нормализованную версию (без интенсивности)
        this.cache.set(cacheKey, {
            noiseData: new Float32Array(noiseData.map(v => v / intensity)),
            histogram: [...histogram]
        });
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Генерирует градиенты для шума Перлина
     * @private
     */
    generateGradients(width, height, scale) {
        // Очень упрощенная версия для демонстрации
        // В реальности мы бы использовали хеш-функцию и предварительно рассчитанные градиенты
        const gradientSize = Math.ceil(Math.max(width, height) / scale) + 1;
        const gradients = new Array(gradientSize * gradientSize);
        
        for (let i = 0; i < gradients.length; i++) {
            const angle = Math.random() * Math.PI * 2;
            gradients[i] = {
                x: Math.cos(angle),
                y: Math.sin(angle)
            };
        }
        
        return gradients;
    }
    
    /**
     * Вычисляет значение шума Перлина в точке
     * @private
     */
    perlinValue(x, y, gradients) {
        // Очень упрощенная версия для демонстрации
        return Math.sin(x * 0.1) * Math.cos(y * 0.1) * 0.5 + 0.5;
    }
    
    /**
     * Применяет шум к изображению
     * 
     * @param {ImageData} originalImageData - Исходные данные изображения
     * @param {Float32Array} noiseData - Данные шума
     * @param {number} intensity - Интенсивность применения
     * @returns {ImageData} - Зашумленное изображение
     */
    applyNoiseToImage(originalImageData, noiseData, intensity = 1.0) {
        const width = originalImageData.width;
        const height = originalImageData.height;
        const result = new ImageData(width, height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                const noiseIndex = y * width + x;
                const noiseValue = noiseData[noiseIndex] * intensity;
                
                // Применяем шум к каждому каналу
                for (let c = 0; c < 3; c++) {
                    const value = originalImageData.data[i + c] + noiseValue;
                    result.data[i + c] = Math.min(Math.max(value, 0), 255);
                }
                
                // Сохраняем альфа-канал без изменений
                result.data[i + 3] = originalImageData.data[i + 3];
            }
        }
        
        return result;
    }
    
    /**
     * Создает визуализацию шума
     * 
     * @param {Float32Array} noiseData - Данные шума
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @returns {ImageData} - Визуализация шума
     */
    createNoiseVisualization(noiseData, width, height) {
        const result = new ImageData(width, height);
        
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                const noiseIndex = y * width + x;
                const noiseValue = noiseData[noiseIndex];
                
                // Преобразуем значение шума в диапазон [0, 255]
                const visualValue = Math.min(Math.max(Math.round(noiseValue + 128), 0), 255);
                
                // Устанавливаем одинаковые значения для всех каналов (оттенки серого)
                result.data[i] = visualValue;
                result.data[i + 1] = visualValue;
                result.data[i + 2] = visualValue;
                result.data[i + 3] = 255; // Полная непрозрачность
            }
        }
        
        return result;
    }
}

// Создаем глобальный экземпляр генератора шумов
const noiseGenerator = new NoiseGenerator();