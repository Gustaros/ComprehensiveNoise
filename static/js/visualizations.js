/**
 * Функции для визуализации шума и его характеристик
 */

/**
 * Создает гистограмму распределения шума
 * 
 * @param {string} elementId - ID элемента для размещения гистограммы
 * @param {Array} histogram - Данные гистограммы
 * @param {Object} options - Дополнительные параметры
 */
function createHistogram(elementId, histogram, options = {}) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Настройки по умолчанию
    const defaultOptions = {
        title: 'Распределение шума',
        xAxisLabel: 'Значение шума',
        yAxisLabel: 'Частота',
        barColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        showTheoretical: false,
        theoreticalType: 'gaussian',
        theoreticalParams: { mean: 0, sigma: 25 }
    };
    
    // Объединяем с переданными опциями
    const config = { ...defaultOptions, ...options };
    
    // Подготавливаем данные
    const labels = Array.from({ length: histogram.length }, (_, i) => i - 128);
    
    // Создаем наборы данных
    const datasets = [{
        label: 'Фактическое распределение',
        data: histogram,
        backgroundColor: config.barColor,
        borderColor: config.borderColor,
        borderWidth: 1
    }];
    
    // Добавляем теоретическое распределение, если нужно
    if (config.showTheoretical) {
        const theoreticalData = generateTheoreticalDistribution(
            config.theoreticalType, 
            labels, 
            config.theoreticalParams
        );
        
        // Нормализуем теоретическое распределение к тому же масштабу
        const maxHistogram = Math.max(...histogram);
        const maxTheoretical = Math.max(...theoreticalData);
        const scaleFactor = maxHistogram / maxTheoretical;
        
        const scaledTheoretical = theoreticalData.map(v => v * scaleFactor);
        
        datasets.push({
            label: 'Теоретическое распределение',
            data: scaledTheoretical,
            type: 'line',
            fill: false,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            pointRadius: 0
        });
    }
    
    // Настраиваем конфигурацию графика
    const chartConfig = {
        type: 'bar',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: config.title,
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: config.xAxisLabel
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: config.yAxisLabel
                    },
                    beginAtZero: true
                }
            }
        }
    };
    
    // Создаем график
    new Chart(element, chartConfig);
}

/**
 * Генерирует теоретическое распределение по заданным параметрам
 * 
 * @param {string} type - Тип распределения
 * @param {Array} xValues - Значения по оси X
 * @param {Object} params - Параметры распределения
 * @returns {Array} - Значения плотности вероятности
 */
function generateTheoreticalDistribution(type, xValues, params) {
    switch (type) {
        case 'gaussian':
            return xValues.map(x => {
                const { mean, sigma } = params;
                return (1 / (sigma * Math.sqrt(2 * Math.PI))) * 
                       Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(sigma, 2)));
            });
            
        case 'poisson':
            return xValues.map(x => {
                const { lambda } = params;
                const k = Math.round(x + 128); // Преобразуем обратно в [0, 255]
                if (k < 0) return 0;
                return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
            });
            
        case 'uniform':
            return xValues.map(x => {
                const { a, b } = params;
                return (x >= a && x <= b) ? 1 / (b - a) : 0;
            });
            
        default:
            return xValues.map(() => 0);
    }
}

/**
 * Факториал числа (вспомогательная функция)
 * 
 * @param {number} n - Целое число
 * @returns {number} - Факториал
 */
function factorial(n) {
    if (n === 0 || n === 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

/**
 * Создает спектральный анализ шума
 * 
 * @param {string} elementId - ID элемента для размещения спектра
 * @param {Float32Array} noiseData - Данные шума
 * @param {number} width - Ширина изображения
 * @param {number} height - Высота изображения
 * @param {Object} options - Дополнительные параметры
 */
function createSpectrumVisualization(elementId, noiseData, width, height, options = {}) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Настройки по умолчанию
    const defaultOptions = {
        title: 'Частотный спектр шума',
        colorscale: 'Viridis'
    };
    
    // Объединяем с переданными опциями
    const config = { ...defaultOptions, ...options };
    
    // Создаем 2D матрицу из данных шума
    const noiseMatrix = [];
    for (let y = 0; y < height; y++) {
        const row = [];
        for (let x = 0; x < width; x++) {
            row.push(noiseData[y * width + x]);
        }
        noiseMatrix.push(row);
    }
    
    // Применяем 2D FFT (используем упрощенный спектр для демонстрации)
    const spectrum = computeSimplifiedSpectrum(noiseMatrix, width, height);
    
    // Настраиваем данные для тепловой карты
    const plotData = [
        {
            z: spectrum,
            type: 'heatmap',
            colorscale: config.colorscale,
            showscale: true
        }
    ];
    
    // Настраиваем макет
    const layout = {
        title: config.title,
        xaxis: {
            title: 'Частота X',
            showticklabels: false
        },
        yaxis: {
            title: 'Частота Y',
            showticklabels: false
        },
        margin: { t: 50, r: 50, b: 50, l: 50 }
    };
    
    // Создаем визуализацию
    Plotly.newPlot(element, plotData, layout, { responsive: true });
}

/**
 * Упрощенный расчет спектра для демонстрации
 * 
 * @param {Array} matrix - 2D матрица с данными шума
 * @param {number} width - Ширина
 * @param {number} height - Высота
 * @returns {Array} - 2D матрица спектра
 */
function computeSimplifiedSpectrum(matrix, width, height) {
    // Для реального применения мы бы использовали библиотеку FFT
    // Это упрощенный пример для демонстрации
    
    const spectrum = [];
    const centerX = Math.floor(width / 2);
    const centerY = Math.floor(height / 2);
    
    for (let y = 0; y < height; y++) {
        const row = [];
        for (let x = 0; x < width; x++) {
            // Расстояние от центра (симуляция радиального профиля спектра)
            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Разные шумы имеют разные спектральные характеристики
            let value;
            
            // Пример для белого шума (равномерный спектр)
            value = Math.random() * 0.5 + 0.5;
            
            // Применяем логарифмический масштаб для лучшей визуализации
            value = Math.log(1 + value);
            
            row.push(value);
        }
        spectrum.push(row);
    }
    
    return spectrum;
}