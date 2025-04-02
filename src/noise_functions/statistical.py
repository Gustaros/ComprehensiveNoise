"""
Функции для генерации статистических типов шумов
"""

import numpy as np
from scipy import stats, ndimage

def add_gaussian_noise(image, mean=0, sigma=25):
    """
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
    """
    row, col = image.shape
    noise = np.random.normal(mean, sigma, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """
    Добавляет шум соль и перец к изображению.
    
    Параметры:
    ----------
    image : ndarray
        Исходное изображение (в градациях серого или цветное)
    salt_prob : float
        Вероятность белых пикселей (соль)
    pepper_prob : float
        Вероятность черных пикселей (перец)
        
    Возвращает:
    --------
    noisy_img : ndarray
        Изображение с добавленным шумом соль и перец
    noise : ndarray
        Шумовой паттерн, который был добавлен к изображению
    """
    noisy_img = np.copy(image)
    noise = np.zeros_like(image, dtype=float)
    
    # Добавляем соль (белые пиксели)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_img[salt_mask] = 255
    noise[salt_mask] = 255 - image[salt_mask]
    
    # Добавляем перец (черные пиксели)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_img[pepper_mask] = 0
    noise[pepper_mask] = -image[pepper_mask]
    
    return noisy_img, noise

def add_poisson_noise(image, lambda_factor=1.0):
    """
    Добавляет пуассоновский шум к изображению.
    
    Параметры:
    ----------
    image : ndarray
        Исходное изображение (в градациях серого или цветное)
    lambda_factor : float
        Фактор масштабирования для параметра интенсивности
        
    Возвращает:
    --------
    noisy_img : ndarray
        Изображение с добавленным пуассоновским шумом
    noise : ndarray
        Шумовой паттерн, который был добавлен к изображению
    """
    # Масштабируем изображение для генерации шума
    vals = lambda_factor * image
    
    # Защита от отрицательных значений
    vals = np.maximum(vals, 0)
    
    # Генерируем шум с распределением Пуассона
    noisy = np.random.poisson(vals)
    
    # Вычисляем шумовой паттерн
    noise = noisy - vals
    
    # Масштабируем обратно
    noisy_img = noisy / lambda_factor
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

def add_speckle_noise(image, var=0.1):
    """
    Добавляет спекл-шум к изображению.
    
    Параметры:
    ----------
    image : ndarray
        Исходное изображение (в градациях серого или цветное)
    var : float
        Дисперсия мультипликативного шума
        
    Возвращает:
    --------
    noisy_img : ndarray
        Изображение с добавленным спекл-шумом
    noise : ndarray
        Шумовой паттерн, который был добавлен к изображению
    """
    row, col = image.shape
    
    # Генерируем гауссовский шум с заданной дисперсией и нулевым средним
    gauss = np.random.normal(0, np.sqrt(var), (row, col))
    
    # Спекл-шум - это мультипликативный шум
    noise = image * gauss
    noisy_img = image + noise
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# Другие функции для других статистических типов шумов...