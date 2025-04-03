import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm, kurtosis, skew
import io
import base64
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import re
import importlib.util

# Add parent directory to path to import noise library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noises

# Configuration
SITE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SITE_DIR, "output")
TEMPLATES_DIR = os.path.join(SITE_DIR, "templates")
ASSETS_DIR = os.path.join(SITE_DIR, "assets")
NOISE_PAGES_DIR = os.path.join(OUTPUT_DIR, "noise_types")
CATEGORY_PAGES_DIR = os.path.join(OUTPUT_DIR, "categories")

# Make sure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NOISE_PAGES_DIR, exist_ok=True)
os.makedirs(CATEGORY_PAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "assets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "assets", "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "assets", "js"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "assets", "css"), exist_ok=True)

# Load metadata
metadata_file = os.path.join(SITE_DIR, 'noise_metadata.json')
if os.path.exists(metadata_file):
    with open(metadata_file, 'r', encoding='utf-8') as f:
        NOISE_METADATA = json.load(f)
else:
    # Fall back to basic metadata
    NOISE_METADATA = {
        "gaussian_noise": {
            "title": "Gaussian (Normal) Noise",
            "category": "Statistical Distribution-Based",
            "description": "Gaussian noise is a type of statistical noise with a probability density function equal to the normal distribution. It is also known as normal noise.",
            "formula": r"g(x,y) = f(x,y) + \eta(x,y), \text{ where } \eta \sim \mathcal{N}(\mu, \sigma^2)",
            "formula_explanation": "The noisy pixel g(x,y) is the sum of the original pixel f(x,y) and Gaussian distributed random variable η with mean μ and variance σ².",
            "parameters": {
                "mean": {"default": 0, "min": -50, "max": 50, "step": 1, "description": "Mean of the Gaussian distribution"},
                "sigma": {"default": 25, "min": 0, "max": 100, "step": 1, "description": "Standard deviation of the Gaussian distribution"}
            },
            "code_snippet": """def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    noise = np.random.normal(mean, sigma, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise"""
        }
    }

# Function to get noise function by name - rest of the file is the same as build_site.py
# ...existing code...
