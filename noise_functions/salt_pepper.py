import numpy as np
from typing import Tuple, Optional

def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Добавляет шум 'соль и перец' к изображению.
    
    Параметры:
    ----------
    image : np.ndarray
        Исходное изображение (в градациях серого или цветное)
    amount : float
        Вероятность шума (доля затронутых пикселей), от 0 до 1
    salt_vs_pepper : float
        Соотношение соли к перцу (0.5 означает равное количество)
    seed : int, optional
        Зерно для генератора случайных чисел
        
    Возвращает:
    -----------
    noisy_img : np.ndarray
        Изображение с добавленным шумом 'соль и перец'
    noise_pattern : np.ndarray
        Шумовой паттерн, который был добавлен к изображению
    
    Примеры:
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> img = np.ones((100, 100)) * 128  # Серое изображение
    >>> noisy, pattern = add_salt_pepper_noise(img, amount=0.1)
    >>> plt.imshow(noisy, cmap='gray')
    """
    # Проверка параметров
    if not (0 <= amount <= 1):
        raise ValueError("Параметр amount должен быть в диапазоне [0, 1]")
    if not (0 <= salt_vs_pepper <= 1):
        raise ValueError("Параметр salt_vs_pepper должен быть в диапазоне [0, 1]")
    
    # Устанавливаем seed для воспроизводимости
    if seed is not None:
        np.random.seed(seed)
    
    # Создаем копию исходного изображения
    noisy_img = np.copy(image)
    
    # Создаем шумовой паттерн
    noise_pattern = np.zeros_like(image)
    
    # Рассчитываем количество пикселей для изменения
    total_pixels = image.size
    num_noise_pixels = int(amount * total_pixels)
    
    # Количество "соли" (белых пикселей)
    num_salt = int(num_noise_pixels * salt_vs_pepper)
    
    # Количество "перца" (черных пикселей)
    num_pepper = num_noise_pixels - num_salt
    
    # Определяем максимальное и минимальное значение для изображения
    # Обычно 0 и 255 для uint8, или 0 и 1 для float
    max_value = 255 if image.dtype == np.uint8 else 1.0
    min_value = 0
    
    # Добавляем "соль" (белые пиксели)
    if num_salt > 0:
        # Получаем случайные индексы для "соли"
        salt_idx = np.random.choice(total_pixels, num_salt, replace=False)
        # Конвертируем линейные индексы в многомерные
        salt_coords = np.unravel_index(salt_idx, image.shape)
        
        # Устанавливаем белые пиксели
        noisy_img[salt_coords] = max_value
        noise_pattern[salt_coords] = max_value - image[salt_coords]
    
    # Добавляем "перец" (черные пиксели)
    if num_pepper > 0:
        # Получаем случайные индексы для "перца"
        pepper_idx = np.random.choice(total_pixels, num_pepper, replace=False)
        # Конвертируем линейные индексы в многомерные
        pepper_coords = np.unravel_index(pepper_idx, image.shape)
        
        # Устанавливаем черные пиксели
        noisy_img[pepper_coords] = min_value
        noise_pattern[pepper_coords] = min_value - image[pepper_coords]
    
    return noisy_img, noise_pattern
