/**
 * Демонстрация гауссовского шума
 */

document.addEventListener('DOMContentLoaded', function() {
    // Получаем элементы канвас
    const noisyCanvas = document.getElementById('noisy-canvas');
    const noiseCanvas = document.getElementById('noise-canvas');
    const noisyCtx = noisyCanvas.getContext('2d');
    const noiseCtx = noiseCanvas.getContext('2d');
    
    // Получаем элементы слайдеров
    const meanSlider = document.getElementById('mean-slider');
    const sigmaSlider = document.getElementById('sigma-slider');
    const intensitySlider = document.getElementById('intensity-slider');
    
    // Получаем элементы отображения значений
    const meanValue = document.getElementById('mean-value');
    const sigmaValue = document.getElementById('sigma-value');
    const intensityValue = document.getElementById('intensity-value');
    
    // Получаем элементы кнопок
    const regenerateBtn = document.getElementById('regenerate-btn');
    const downloadBtn = document.getElementById('download-btn');
    
    // Исходное изображение
    const originalImg = document.getElementById('original');
    let originalImageData;
    
    // Флаг для отслеживания статуса генерации
    let isGenerating = false;
    
    // Функция для загрузки исходного изображения
    function loadOriginalImage() {
        // Показываем индикатор загрузки
        showLoading();
        
        try {
            // Создаем оффскрин канвас для загрузки изображения
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = noisyCanvas.width;
            offscreenCanvas.height = noisyCanvas.height;
            const offscreenCtx = offscreenCanvas.getContext('2d');
            
            // Рисуем изображение на канвасе
            offscreenCtx.drawImage(originalImg, 0, 0, noisyCanvas.width, noisyCanvas.height);
            originalImageData = offscreenCtx.getImageData(0, 0, noisyCanvas.width, noisyCanvas.height);
            
            // Генерируем и применяем начальный шум
            generateAndApplyNoise();
        } catch (error) {
            console.error("Error loading original image:", error);
            hideLoading();
            showError("Не удалось загрузить исходное изображение.");
        }
    }
    
    // Загружаем исходное изображение, когда оно готово
    if (originalImg.complete) {
        loadOriginalImage();
    } else {
        originalImg.onload = loadOriginalImage;
    }
    
    // Обработчики событий для слайдеров с дебаунсом
    let debounceTimer;
    
    function addSliderListener(slider, valueElement, callback) {
        slider.addEventListener('input', function() {
            valueElement.textContent = this.value;
            
            // Обновляем только отображение значения при перетаскивании слайдера
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                if (callback) callback();
            }, 50); // 50мс дебаунс для генерации шума
        });
        
        // Немедленное обновление при отпускании слайдера
        slider.addEventListener('change', function() {
            clearTimeout(debounceTimer);
            if (callback) callback();
        });
    }
    
    addSliderListener(meanSlider, meanValue, generateAndApplyNoise);
    addSliderListener(sigmaSlider, sigmaValue, generateAndApplyNoise);
    addSliderListener(intensitySlider, intensityValue, generateAndApplyNoise);
    
    // Обработчик для кнопки регенерации шума
    regenerateBtn.addEventListener('click', function() {
        if (!isGenerating) {
            // Добавляем класс для визуальной обратной связи
            this.classList.add('loading');
            this.textContent = 'Генерация...';
            generateAndApplyNoise(true);
        }
    });
    
    // Обработчик для кнопки скачивания
    downloadBtn.addEventListener('click', function() {
        try {
            const link = document.createElement('a');
            link.download = 'gaussian_noise_demo.png';
            link.href = noisyCanvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            console.error("Error downloading image:", error);
            showError("Не удалось скачать изображение.");
        }
    });
    
    // Функции для показа и скрытия индикатора загрузки
    function showLoading() {
        isGenerating = true;
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
        }
    }
    
    function hideLoading() {
        isGenerating = false;
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('active');
        }
        
        // Сбрасываем состояние кнопки
        if (regenerateBtn) {
            regenerateBtn.classList.remove('loading');
            regenerateBtn.textContent = 'Сгенерировать заново';
        }
    }
    
    // Функция для показа ошибки
    function showError(message) {
        alert(message);
    }
    
    // Основная функция для генерации и применения шума
    async function generateAndApplyNoise(forceRegen = false) {
        if (!originalImageData || isGenerating) return;
        
        showLoading();
        
        try {
            // Получаем значения параметров
            const mean = parseInt(meanSlider.value);
            const sigma = parseInt(sigmaSlider.value);
            const intensity = parseInt(intensitySlider.value) / 100;
            
            // Размеры изображения
            const width = originalImageData.width;
            const height = originalImageData.height;
            
            // Генерируем гауссовский шум
            const { noiseData, histogram } = await noiseGenerator.generateGaussianNoise(
                width, height, mean, sigma, intensity
            );
            
            // Применяем шум к изображению
            const noisyImageData = await noiseGenerator.applyNoiseToImage(
                originalImageData, noiseData, 1.0
            );
            
            // Создаем визуализацию шума
            const noiseVisualization = await noiseGenerator.createNoiseVisualization(
                noiseData, width, height
            );
            
            // Отображаем зашумленное изображение
            noisyCtx.putImageData(noisyImageData, 0, 0);
            
            // Отображаем визуализацию шума
            noiseCtx.putImageData(noiseVisualization, 0, 0);
            
            // Создаем гистограмму с обновленными параметрами
            createHistogram('histogram-plot', histogram, {
                title: 'Распределение гауссовского шума',
                showTheoretical: true,
                theoreticalType: 'gaussian',
                theoreticalParams: { mean: mean, sigma: sigma }
            });
            
            // Создаем визуализацию спектра
            createSpectrumVisualization('spectrum-plot', noiseData, width, height, {
                title: 'Спектр гауссовского шума'
            });
            
            // Скрываем заглушки для графиков
            document.querySelectorAll('.plot-placeholder').forEach(placeholder => {
                placeholder.classList.add('hidden');
            });
            
        } catch (error) {
            console.error("Error generating or applying noise:", error);
            showError("Произошла ошибка при генерации шума.");
        } finally {
            hideLoading();
        }
    }
});