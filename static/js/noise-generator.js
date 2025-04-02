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
        // Флаг для веб-воркеров если они поддерживаются
        this.workersSupported = typeof Worker !== 'undefined';
        // Очередь задач 
        this.processingQueue = Promise.resolve();
    }
    
    /**
     * Генерирует гауссовский (нормальный) шум
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} mean - Среднее значение (μ)
     * @param {number} sigma - Стандартное отклонение (σ)
     * @param {number} intensity - Интенсивность шума (множитель)
     * @returns {Promise<Object>} - Объект с данными шума и гистограммой
     */
    generateGaussianNoise(width, height, mean = 0, sigma = 25, intensity = 1.0) {
        // Добавляем в очередь
        this.processingQueue = this.processingQueue.then(() => {
            return new Promise((resolve) => {
                // Проверяем кеш
                const cacheKey = `gaussian_${width}_${height}_${mean}_${sigma}`;
                if (this.cache.has(cacheKey)) {
                    const cached = this.cache.get(cacheKey);
                    const result = this.applyIntensity(cached, intensity);
                    resolve(result);
                    return;
                }

                // Генерируем шум
                try {
                    const noiseData = new Float32Array(width * height);
                    const histogram = new Array(256).fill(0);
                    
                    // Используем оптимизированный метод Бокса-Мюллера
                    for (let i = 0; i < noiseData.length; i += 2) {
                        let u1, u2, z1, z2;
                        
                        do {
                            u1 = Math.random();
                            u2 = Math.random();
                        } while (u1 <= 1e-7); // Предотвращаем log(0)
                        
                        z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                        z2 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
                        
                        // Применяем среднее и стандартное отклонение
                        noiseData[i] = z1 * sigma + mean;
                        if (i + 1 < noiseData.length) {
                            noiseData[i + 1] = z2 * sigma + mean;
                        }
                    }
                    
                    // Обновляем гистограмму и применяем интенсивность
                    this.updateHistogram(noiseData, histogram);
                    
                    // Кешируем результат без интенсивности
                    const result = { 
                        noiseData: noiseData,
                        histogram: histogram
                    };
                    
                    this.cache.set(cacheKey, {
                        noiseData: new Float32Array(noiseData),
                        histogram: [...histogram]
                    });
                    
                    // Применяем интенсивность к итоговому результату
                    const finalResult = this.applyIntensity(result, intensity);
                    resolve(finalResult);
                } catch (error) {
                    console.error("Error generating Gaussian noise:", error);
                    resolve(this.generateFallbackNoise(width, height, intensity));
                }
            });
        });
        
        return this.processingQueue;
    }
    
    /**
     * Применяет интенсивность к данным шума
     * 
     * @param {Object} noiseData - Объект с данными шума
     * @param {number} intensity - Интенсивность
     * @returns {Object} - Объект с модифицированными данными
     */
    applyIntensity(noiseData, intensity) {
        if (intensity === 1.0) return noiseData;
        
        const result = {
            noiseData: new Float32Array(noiseData.noiseData.length),
            histogram: new Array(256).fill(0)
        };
        
        for (let i = 0; i < noiseData.noiseData.length; i++) {
            result.noiseData[i] = noiseData.noiseData[i] * intensity;
        }
        
        this.updateHistogram(result.noiseData, result.histogram);
        return result;
    }
    
    /**
     * Обновляет гистограмму на основе данных шума
     * 
     * @param {Float32Array} noiseData - Данные шума
     * @param {Array} histogram - Массив для гистограммы
     */
    updateHistogram(noiseData, histogram) {
        // Сбрасываем гистограмму
        histogram.fill(0);
        
        for (let i = 0; i < noiseData.length; i++) {
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
    }
    
    /**
     * Генерирует резервный шум в случае ошибки
     * 
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @param {number} intensity - Интенсивность
     * @returns {Object} - Объект с данными шума и гистограммой
     */
    generateFallbackNoise(width, height, intensity) {
        const noiseData = new Float32Array(width * height);
        const histogram = new Array(256).fill(0);
        
        // Генерируем случайный шум
        for (let i = 0; i < noiseData.length; i++) {
            noiseData[i] = (Math.random() * 256 - 128) * intensity;
            const bin = Math.min(Math.max(Math.round(noiseData[i] + 128), 0), 255);
            histogram[bin]++;
        }
        
        return {
            noiseData: noiseData,
            histogram: histogram
        };
    }
    
    /**
     * Применяет шум к изображению с оптимизацией
     * 
     * @param {ImageData} originalImageData - Исходные данные изображения
     * @param {Float32Array} noiseData - Данные шума
     * @param {number} intensity - Интенсивность применения
     * @returns {Promise<ImageData>} - Зашумленное изображение
     */
    applyNoiseToImage(originalImageData, noiseData, intensity = 1.0) {
        return new Promise((resolve) => {
            try {
                const width = originalImageData.width;
                const height = originalImageData.height;
                const result = new ImageData(width, height);
                
                // Копируем данные для обработки
                const origData = originalImageData.data;
                const resultData = result.data;
                
                // Проверяем размеры данных шума и изображения
                if (noiseData.length !== width * height) {
                    console.warn("Noise data dimensions don't match image dimensions. Resizing noise.");
                    // Здесь можно добавить логику изменения размера, но для простоты используем циклическое обращение
                }
                
                // Применяем шум с оптимизацией
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const i = (y * width + x) * 4;
                        const noiseIndex = y * width + x % noiseData.length; // Используем остаток для безопасности
                        const noiseValue = noiseData[noiseIndex] * intensity;
                        
                        // Применяем шум к каждому каналу с проверкой границ
                        for (let c = 0; c < 3; c++) {
                            const value = Math.max(0, Math.min(255, origData[i + c] + noiseValue));
                            resultData[i + c] = value;
                        }
                        
                        // Сохраняем альфа-канал без изменений
                        resultData[i + 3] = origData[i + 3];
                    }
                }
                
                resolve(result);
            } catch (error) {
                console.error("Error applying noise to image:", error);
                // Возвращаем оригинальное изображение в случае ошибки
                resolve(originalImageData);
            }
        });
    }
    
    /**
     * Создает визуализацию шума с оптимизацией
     * 
     * @param {Float32Array} noiseData - Данные шума
     * @param {number} width - Ширина изображения
     * @param {number} height - Высота изображения
     * @returns {Promise<ImageData>} - Визуализация шума
     */
    createNoiseVisualization(noiseData, width, height) {
        return new Promise((resolve) => {
            try {
                const result = new ImageData(width, height);
                const resultData = result.data;
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        const i = (y * width + x) * 4;
                        const noiseIndex = y * width + x;
                        
                        // Проверяем границы noiseData
                        if (noiseIndex >= noiseData.length) {
                            // Используем повторение данных шума, если не хватает
                            const fallbackIndex = noiseIndex % noiseData.length;
                            const visualValue = Math.min(Math.max(Math.round(noiseData[fallbackIndex] + 128), 0), 255);
                            
                            resultData[i] = visualValue;
                            resultData[i + 1] = visualValue;
                            resultData[i + 2] = visualValue;
                            resultData[i + 3] = 255;
                        } else {
                            // Преобразуем значение шума в диапазон [0, 255]
                            const visualValue = Math.min(Math.max(Math.round(noiseData[noiseIndex] + 128), 0), 255);
                            
                            // Устанавливаем одинаковые значения для всех каналов (оттенки серого)
                            resultData[i] = visualValue;
                            resultData[i + 1] = visualValue;
                            resultData[i + 2] = visualValue;
                            resultData[i + 3] = 255; // Полная непрозрачность
                        }
                    }
                }
                
                resolve(result);
            } catch (error) {
                console.error("Error creating noise visualization:", error);
                // Создаем простую заглушку в случае ошибки
                const fallbackResult = new ImageData(width, height);
                fallbackResult.data.fill(128); // Заполняем серым цветом
                for (let i = 3; i < fallbackResult.data.length; i += 4) {
                    fallbackResult.data[i] = 255; // Устанавливаем альфа-канал
                }
                resolve(fallbackResult);
            }
        });
    }
}

// Создаем глобальный экземпляр генератора шумов
const noiseGenerator = new NoiseGenerator();