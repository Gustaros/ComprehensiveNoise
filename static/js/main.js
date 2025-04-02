/**
 * Основной JavaScript файл для сайта каталога шумов
 */

// Функция для настройки вкладок на странице
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Деактивация всех кнопок и панелей
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelectorAll('.tab-pane').forEach(pane => {
                pane.classList.remove('active');
            });
            
            // Активация текущей кнопки и соответствующей панели
            this.classList.add('active');
            const tabId = this.getAttribute('data-tab') + '-code';
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Функция для настройки поиска
function setupSearch() {
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-btn');
    
    if (searchInput && searchButton) {
        searchButton.addEventListener('click', function() {
            performSearch(searchInput.value);
        });
        
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                performSearch(searchInput.value);
            }
        });
    }
}

// Выполнение поиска
function performSearch(query) {
    if (query.trim() !== '') {
        window.location.href = '/search?q=' + encodeURIComponent(query.trim());
    }
}

// Функция для добавления эффекта "прилипания" к верхнему меню
function setupStickyHeader() {
    const header = document.querySelector('header');
    const headerHeight = header.offsetHeight;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            header.classList.add('sticky');
            document.body.style.paddingTop = headerHeight + 'px';
        } else {
            header.classList.remove('sticky');
            document.body.style.paddingTop = 0;
        }
    });
}

// Функция для анимации статистики на главной странице
function animateStatistics() {
    const statValues = document.querySelectorAll('.stat-value');
    
    if (statValues.length === 0) return;
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const valueElement = entry.target;
                const finalValue = parseInt(valueElement.textContent);
                let currentValue = 0;
                
                // Получаем суффикс, если есть
                const suffix = valueElement.textContent.match(/\D+$/)?.[0] || '';
                
                // Анимация подсчета
                const duration = 2000; // 2 секунды
                const interval = 20; // Интервал обновления в мс
                const steps = duration / interval;
                const increment = finalValue / steps;
                
                const counter = setInterval(() => {
                    currentValue += increment;
                    
                    if (currentValue >= finalValue) {
                        valueElement.textContent = finalValue + suffix;
                        clearInterval(counter);
                    } else {
                        valueElement.textContent = Math.floor(currentValue) + suffix;
                    }
                }, interval);
                
                // Отключаем наблюдатель после срабатывания
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    statValues.forEach(value => {
        observer.observe(value);
    });
}

// Функция для установки текущего года в футере
function setCurrentYear() {
    const yearElement = document.querySelector('.copyright span.year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
}

// Функция для обработки мобильного меню
function setupMobileMenu() {
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
        
        // Закрыть меню при клике на ссылку в мобильной версии
        const navLinksA = navLinks.querySelectorAll('a');
        navLinksA.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    navLinks.classList.remove('active');
                    menuToggle.classList.remove('active');
                }
            });
        });
        
        // Закрыть меню при изменении размера окна
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768) {
                navLinks.classList.remove('active');
                menuToggle.classList.remove('active');
            }
        });
    }
}

// Функция для управления состоянием загрузки
function showLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.classList.add('active');
    }
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.classList.remove('active');
    }
}

// Функция для улучшения интерактивных элементов
function enhanceInteractiveElements() {
    // Анимация для слайдеров
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        // Обновляем градиент для отображения заполнения
        const updateRangeGradient = (input) => {
            const min = input.min ? parseFloat(input.min) : 0;
            const max = input.max ? parseFloat(input.max) : 100;
            const value = parseFloat(input.value);
            const percentage = ((value - min) / (max - min)) * 100;
            
            input.style.background = `linear-gradient(to right, 
                var(--primary-color) 0%, 
                var(--primary-color) ${percentage}%, 
                #d3d3d3 ${percentage}%, 
                #d3d3d3 100%)`;
        };
        
        updateRangeGradient(input);
        input.addEventListener('input', () => updateRangeGradient(input));
    });
    
    // Добавляем заглушки для графиков перед их загрузкой
    const plotElements = document.querySelectorAll('#histogram-plot, #spectrum-plot');
    plotElements.forEach(element => {
        const placeholder = document.createElement('div');
        placeholder.className = 'plot-placeholder';
        placeholder.innerHTML = '<div class="spinner"></div><p>Загрузка графика...</p>';
        element.parentNode.insertBefore(placeholder, element.nextSibling);
        
        // Добавим наблюдателя для удаления заглушки, когда график загрузится
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && element.children.length > 0) {
                    placeholder.classList.add('hidden');
                    observer.disconnect();
                }
            });
        });
        
        observer.observe(element, { childList: true, subtree: true });
    });
}

// Инициализация всех функций при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Показываем индикатор загрузки
    showLoading();
    
    // Инициализируем все компоненты
    setupTabs();
    setupSearch();
    setupStickyHeader();
    animateStatistics();
    setCurrentYear();
    setupMobileMenu();
    enhanceInteractiveElements();
    
    // Инициализация всплывающих подсказок, если библиотека подключена
    if (typeof tippy !== 'undefined') {
        tippy('[data-tippy-content]');
    }
    
    // Инициализация рендеринга математических формул, если библиотека подключена
    if (typeof renderMathInElement !== 'undefined') {
        renderMathInElement(document.body, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ]
        });
    }
    
    // Инициализация подсветки синтаксиса, если библиотека подключена
    if (typeof hljs !== 'undefined') {
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
    }
    
    // Скрываем индикатор загрузки, когда все критические ресурсы загружены
    window.addEventListener('load', function() {
        hideLoading();
    });
    
    // На случай, если событие load уже сработало
    if (document.readyState === 'complete') {
        hideLoading();
    }
});