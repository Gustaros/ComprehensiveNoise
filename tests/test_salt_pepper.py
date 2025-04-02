import numpy as np
import pytest
from noise_functions.salt_pepper import add_salt_pepper_noise

def test_salt_pepper_basic():
    """Тест базовой функциональности шума соль и перец."""
    # Создаем тестовое изображение (серое)
    img = np.ones((100, 100)) * 128
    
    # Добавляем шум соль и перец
    noisy, pattern = add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=0.5, seed=42)
    
    # Проверяем, что форма не изменилась
    assert noisy.shape == img.shape
    assert pattern.shape == img.shape
    
    # Подсчитываем количество белых (соль) и черных (перец) пикселей
    salt_count = np.sum(noisy == 255)
    pepper_count = np.sum(noisy == 0)
    
    # Проверяем соответствие количества шумовых пикселей параметрам
    assert 450 < salt_count < 550  # Примерно 5% от 10000 пикселей (с учетом salt_vs_pepper=0.5)
    assert 450 < pepper_count < 550  # Примерно 5% от 10000 пикселей (с учетом salt_vs_pepper=0.5)
    
    # Проверяем, что шумовой паттерн корректный
    # Значения паттерна должны быть либо 0 (неизменный пиксель), 
    # либо 255-128=127 (соль), либо 0-128=-128 (перец)
    unique_values = np.unique(pattern)
    assert set(unique_values).issubset({0, 127, -128})

def test_salt_pepper_parameters():
    """Тест различных значений параметров шума соль и перец."""
    # Создаем тестовое изображение
    img = np.ones((100, 100)) * 128
    
    # Тест с только солью (без перца)
    noisy, _ = add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=1.0, seed=42)
    salt_count = np.sum(noisy == 255)
    pepper_count = np.sum(noisy == 0)
    assert 950 < salt_count < 1050  # ~10% соли
    assert pepper_count == 0  # Без перца
    
    # Тест с только перцем (без соли)
    noisy, _ = add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=0.0, seed=42)
    salt_count = np.sum(noisy == 255)
    pepper_count = np.sum(noisy == 0)
    assert salt_count == 0  # Без соли
    assert 950 < pepper_count < 1050  # ~10% перца
    
    # Тест с разным количеством шума
    noisy_low, _ = add_salt_pepper_noise(img, amount=0.01, salt_vs_pepper=0.5, seed=42)
    noisy_high, _ = add_salt_pepper_noise(img, amount=0.2, salt_vs_pepper=0.5, seed=42)
    
    low_noise_count = np.sum((noisy_low == 0) | (noisy_low == 255))
    high_noise_count = np.sum((noisy_high == 0) | (noisy_high == 255))
    
    assert low_noise_count < high_noise_count
    assert 50 < low_noise_count < 150  # ~1% шума
    assert 1900 < high_noise_count < 2100  # ~20% шума

def test_salt_pepper_float_images():
    """Тест шума соль и перец на изображениях с плавающей точкой."""
    # Создаем тестовое изображение в диапазоне [0, 1]
    img = np.ones((100, 100)) * 0.5
    
    # Добавляем шум
    noisy, pattern = add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=0.5, seed=42)
    
    # Проверяем, что значения соответствуют диапазону [0, 1]
    assert np.max(noisy) <= 1.0
    assert np.min(noisy) >= 0.0
    
    # Подсчитываем количество белых (соль) и черных (перец) пикселей
    salt_count = np.sum(np.isclose(noisy, 1.0))
    pepper_count = np.sum(np.isclose(noisy, 0.0))
    
    assert 450 < salt_count < 550  # ~5% соли
    assert 450 < pepper_count < 550  # ~5% перца

def test_salt_pepper_colored_images():
    """Тест шума соль и перец на цветных изображениях."""
    # Создаем тестовое цветное изображение (RGB)
    img = np.ones((50, 50, 3)) * 128
    
    # Добавляем шум
    noisy, pattern = add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=0.5, seed=42)
    
    # Проверяем, что форма не изменилась
    assert noisy.shape == img.shape
    assert pattern.shape == img.shape
    
    # В цветных изображениях шум применяется одинаково ко всем каналам,
    # поэтому белые пиксели будут [255, 255, 255], а черные [0, 0, 0]
    # Подсчитываем их, проверяя все три канала одновременно
    salt_pixels = np.sum(np.all(noisy == 255, axis=2))
    pepper_pixels = np.sum(np.all(noisy == 0, axis=2))
    
    assert 100 < salt_pixels < 150  # ~5% от 2500 пикселей
    assert 100 < pepper_pixels < 150  # ~5% от 2500 пикселей

def test_salt_pepper_error_handling():
    """Тест проверки граничных условий и обработки ошибок."""
    img = np.ones((10, 10)) * 128
    
    # Проверка на некорректные значения amount
    with pytest.raises(ValueError):
        add_salt_pepper_noise(img, amount=1.5)
    
    with pytest.raises(ValueError):
        add_salt_pepper_noise(img, amount=-0.1)
    
    # Проверка на некорректные значения salt_vs_pepper
    with pytest.raises(ValueError):
        add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=1.5)
    
    with pytest.raises(ValueError):
        add_salt_pepper_noise(img, amount=0.1, salt_vs_pepper=-0.1)
    
    # Проверка граничных случаев
    # amount=0 не должен добавлять шума
    noisy, pattern = add_salt_pepper_noise(img, amount=0, salt_vs_pepper=0.5)
    assert np.array_equal(noisy, img)
    assert np.all(pattern == 0)
    
    # amount=1 должен заменить все пиксели
    noisy, _ = add_salt_pepper_noise(img, amount=1, salt_vs_pepper=0.5, seed=42)
    assert np.sum(noisy == 128) == 0  # Не должно остаться исходных пикселей
