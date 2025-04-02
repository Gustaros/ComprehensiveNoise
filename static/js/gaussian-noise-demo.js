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
    
    // Функция для загрузки исходного изображения
    function loadOriginalImage() {
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
    }
    
    // Загружаем исходное изображение, когда оно готово
    if (originalImg.complete) {
        loadOriginalImage();
    } else {
        originalImg.onload = loadOriginalImage;
    }
    
    // Обработчики событий для слайдеров
    meanSlider.addEventListener('input', function() {
        meanValue.textContent = this.value;
        generateAndApplyNoise();
    });
    
    sigmaSlider.addEventListener('input', function() {
        sigmaValue.textContent = this.value;
        generateAndApplyNoise();
    });
    
    intensitySlider.addEventListener('input', function() {
        intensityValue.textContent = this.value;
        generateAndApplyNoise();
    });
    
    // Обработчик для кнопки регенерации шума
    regenerateBtn.addEventListener('click', generateAndApplyNoise);
    
    // Обработчик для кнопки скачивания
    downloadBtn.addEventListener('click', function() {
        const link = document.createElement('a');
        link.download = 'gaussian_noise_demo.png';
        link.href = noisyCanvas.toDataURL('image/png');
        link.click();
    });
    
    // Основная функция для генерации и применения шума
    function generateAndApplyNoise() {
        if (!originalImageData) return;
        
        // Получаем значения параметров
        const mean = parseInt(meanSlider.value);
        const sigma = parseInt(sigmaSlider.value);
        const intensity = parseInt(intensitySlider.value) / 100;
        
        // Размеры изображения
        const width = originalImageData.width;
        const height = originalImageData.height;
        
        // Генерируем гауссовский шум
        const { noiseData, histogram } = noiseGenerator.generateGaussianNoise(
            width, height, mean, sigma, intensity
        );
        
        // Применяем шум к изображению
        const noisyImageData = noiseGenerator.applyNoiseToImage(
            originalImageData, noiseData, 1.0
        );
        
        // Создаем визуализацию шума
        const noiseVisualization = noiseGenerator.createNoiseVisualization(
            noiseData, width, height
        );
        
        // Отображаем зашумленное изображение
        noisyCtx.putImageData(noisyImageData, 0, 0);
        
        // Отображаем визуализацию шума
        noiseCtx.putImageData(noiseVisualization, 0, 0);
        
        // Создаем гистограмму
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
    }
});