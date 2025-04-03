import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal, stats
from skimage import util, color, exposure
import cv2
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import pywt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.util import random_noise
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal, stats, special
from skimage import util, color, exposure, transform
import cv2
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
import pywt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import rice, vonmises, gumbel_r, levy, nakagami

try:
    from perlin_noise import PerlinNoise
    has_perlin = True
except ImportError:
    has_perlin = False
    print("perlin_noise package not found. Perlin noise will not be available.")

# Set random seed for reproducibility
np.random.seed(42)

# Create a directory for the output images
OUTPUT_DIR = "noise_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to create a sample image
def create_sample_image(size=(256, 256), pattern="gradient"):
    """Create different types of sample images"""
    if pattern == "gradient":
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        image = xx * yy * 255
    elif pattern == "constant":
        image = np.ones(size) * 128
    elif pattern == "circles":
        x = np.linspace(-1, 1, size[0])
        y = np.linspace(-1, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        image = (np.sin(r * 15) + 1) * 127.5
    elif pattern == "checkerboard":
        x = np.arange(size[0])
        y = np.arange(size[1])
        xx, yy = np.meshgrid(x, y)
        image = ((xx + yy) % 2) * 255
    else:  # default to gradient
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        xx, yy = np.meshgrid(x, y)
        image = xx * yy * 255
    
    return image.astype(np.uint8)

# Function to display and save noise examples
def display_noise_example(title, image, noisy_image, noise, index, total):
    """Display and save a noise example"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Noisy image
    axes[1].imshow(np.clip(noisy_image, 0, 255), cmap='gray')
    axes[1].set_title(f'Image with {title}')
    axes[1].axis('off')
    
    # Noise pattern
    im = axes[2].imshow(noise, cmap='viridis')
    axes[2].set_title(f'{title} Pattern')
    axes[2].axis('off')
    
    # Add colorbar
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.suptitle(f"{index}/{total}: {title}", fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    filename = f"{OUTPUT_DIR}/{index:02d}_{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# -----------------------------------------------
# 1. STATISTICAL DISTRIBUTION-BASED NOISE TYPES
# -----------------------------------------------

# 1.1 Gaussian (Normal) Noise
def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows N(mean, sigma²)
    η(x,y) ~ (1/(sigma*sqrt(2π))) * e^(-(z-mean)²/(2*sigma²))
    """
    row, col = image.shape
    noise = np.random.normal(mean, sigma, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.2 Salt and Pepper Noise
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """
    Formula: For each pixel (x,y):
    g(x,y) = 255 with probability p_salt
    g(x,y) = 0 with probability p_pepper
    g(x,y) = f(x,y) with probability 1 - p_salt - p_pepper
    """
    noisy_img = np.copy(image)
    noise = np.zeros_like(image, dtype=float)
    
    # Salt noise
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_img[salt_mask] = 255
    noise[salt_mask] = 1
    
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_img[pepper_mask] = 0
    noise[pepper_mask] = -1
    
    return noisy_img, noise * 127

# 1.3 Poisson Noise (Shot Noise)
def add_poisson_noise(image, scale=1.0):
    """
    Formula: g(x,y) = Poisson(f(x,y)/scale) * scale
    P(X=k) = (λ^k * e^(-λ)) / k!, where λ = f(x,y)/scale
    """
    # Ensure image values are positive
    img_data = np.maximum(image, 0) / scale
    noise = np.random.poisson(img_data) * scale - img_data * scale
    noisy_img = img_data * scale + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.4 Speckle Noise
def add_speckle_noise(image, var=0.1):
    """
    Formula: g(x,y) = f(x,y) + f(x,y)*η(x,y), where η follows N(0, var)
    """
    row, col = image.shape
    noise = np.random.normal(0, np.sqrt(var), (row, col))
    noisy_img = image + image * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise * 50  # Scale for visualization

# 1.5 Uniform Noise
def add_uniform_noise(image, low=-50, high=50):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows U(low, high)
    p(η) = 1/(high-low) for low ≤ η ≤ high
    """
    row, col = image.shape
    noise = np.random.uniform(low, high, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.6 Rayleigh Noise
def add_rayleigh_noise(image, scale=35):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Rayleigh distribution
    p(z) = (z/σ²)*e^(-z²/(2σ²)) for z ≥ 0
    """
    row, col = image.shape
    noise = stats.rayleigh.rvs(scale=scale, size=(row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.7 Gamma Noise
def add_gamma_noise(image, shape=1.0, scale=40.0):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Gamma(k, θ)
    p(z) = (z^(k-1) * e^(-z/θ)) / (θ^k * Γ(k)) for z > 0
    """
    row, col = image.shape
    noise = stats.gamma.rvs(shape, scale=scale, size=(row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.8 Exponential Noise
def add_exponential_noise(image, scale=25):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Exp(λ)
    p(z) = λ*e^(-λz) for z ≥ 0
    λ = 1/scale
    """
    row, col = image.shape
    noise = stats.expon.rvs(scale=scale, size=(row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.9 Laplacian Noise
def add_laplacian_noise(image, loc=0, scale=20):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Laplace(μ,b)
    p(z) = (1/(2b))*e^(-|z-μ|/b)
    """
    row, col = image.shape
    noise = stats.laplace.rvs(loc=loc, scale=scale, size=(row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.10 Cauchy Noise
def add_cauchy_noise(image, loc=0, scale=5):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Cauchy(x₀,γ)
    p(z) = 1/(π*γ*(1+((z-x₀)/γ)²))
    """
    row, col = image.shape
    noise = stats.cauchy.rvs(loc=loc, scale=scale, size=(row, col))
    # Clip extreme values (Cauchy has heavy tails)
    noise = np.clip(noise, -100, 100)
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.11 Chi-Square Noise
def add_chi2_noise(image, df=1, scale=10):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows χ²(k)
    p(z) = (z^(k/2-1)*e^(-z/2))/(2^(k/2)*Γ(k/2)) for z > 0
    """
    row, col = image.shape
    noise = stats.chi2.rvs(df, size=(row, col)) * scale - df * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.12 Beta Noise
def add_beta_noise(image, a=2, b=2, scale=100):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Beta(α,β)
    p(z) = (z^(α-1)*(1-z)^(β-1))/B(α,β) for 0 < z < 1
    """
    row, col = image.shape
    noise = (stats.beta.rvs(a, b, size=(row, col)) - 0.5) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.13 Weibull Noise
def add_weibull_noise(image, a=1.5, scale=20):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Weibull(λ,k)
    p(z) = (k/λ)*(z/λ)^(k-1)*e^(-(z/λ)^k) for z ≥ 0
    """
    row, col = image.shape
    noise = stats.weibull_min.rvs(a, scale=scale, size=(row, col))
    noise -= np.mean(noise)  # Center the noise around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.14 Logistic Noise
def add_logistic_noise(image, loc=0, scale=15):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Logistic(μ,s)
    p(z) = e^(-(z-μ)/s)/(s*(1+e^(-(z-μ)/s))²)
    """
    row, col = image.shape
    noise = stats.logistic.rvs(loc=loc, scale=scale, size=(row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.15 Student's t-Noise
def add_t_noise(image, df=3, scale=20):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Student's t(ν)
    p(z) = Γ((ν+1)/2)/(Γ(ν/2)*sqrt(νπ)*(1+z²/ν)^((ν+1)/2))
    """
    row, col = image.shape
    noise = stats.t.rvs(df, size=(row, col)) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.16 F-Distribution Noise
def add_f_noise(image, dfn=5, dfd=2, scale=10):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows F(d1,d2)
    Complex PDF formula based on beta function
    """
    row, col = image.shape
    noise = stats.f.rvs(dfn, dfd, size=(row, col)) * scale
    noise -= np.mean(noise)  # Center the noise around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.17 Lognormal Noise
def add_lognormal_noise(image, mean=0, sigma=1, scale=10):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows LogN(μ,σ)
    p(z) = (1/(zσ√(2π)))*e^(-(ln(z)-μ)²/(2σ²)) for z > 0
    """
    row, col = image.shape
    noise = stats.lognorm.rvs(s=sigma, scale=np.exp(mean), size=(row, col)) * scale
    noise -= np.mean(noise)  # Center the noise around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.18 Binomial Noise
def add_binomial_noise(image, n=20, p=0.5, scale=10):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Bin(n,p)
    p(k) = (n choose k)*p^k*(1-p)^(n-k) for k ∈ {0,1,...,n}
    """
    row, col = image.shape
    noise = (stats.binom.rvs(n, p, size=(row, col)) - n*p) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.19 Negative Binomial Noise
def add_nbinom_noise(image, n=10, p=0.5, scale=5):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows NB(r,p)
    p(k) = (k+r-1 choose k)*(1-p)^r*p^k for k ≥ 0
    """
    row, col = image.shape
    noise = (stats.nbinom.rvs(n, p, size=(row, col)) - n*(1-p)/p) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.20 Hypergeometric Noise
def add_hypergeom_noise(image, M=100, n=20, N=10, scale=5):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Hypergeometric(M,n,N)
    p(k) = (n choose k)*(M-n choose N-k)/(M choose N) for max(0,N+n-M) ≤ k ≤ min(n,N)
    """
    row, col = image.shape
    noise = (stats.hypergeom.rvs(M, n, N, size=(row, col)) - n*N/M) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.21 Pareto Noise
def add_pareto_noise(image, b=1.5, scale=5):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Pareto(α,xm)
    p(z) = (α*xm^α)/z^(α+1) for z ≥ xm
    """
    row, col = image.shape
    noise = (stats.pareto.rvs(b, size=(row, col)) - b/(b-1)) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 1.22 Maxwell-Boltzmann Noise
def add_maxwell_noise(image, scale=25):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows Maxwell-Boltzmann distribution
    p(z) = sqrt(2/π)*(z²/a³)*e^(-z²/(2a²)) for z ≥ 0
    """
    row, col = image.shape
    noise = stats.maxwell.rvs(scale=scale, size=(row, col))
    noise -= np.mean(noise)  # Center the noise around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 2. COLORED NOISE TYPES
# -----------------------------------------------

# 2.1 White Noise
def add_white_noise(image, intensity=25):
    """
    Formula: g(x,y) = f(x,y) + η(x,y), where η follows N(0,σ²)
    Spectral density: S(f) = constant
    """
    row, col = image.shape
    noise = np.random.normal(0, intensity, (row, col))
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.2 Pink Noise (1/f Noise)
def add_pink_noise(image, intensity=25):
    """
    Formula: Power spectrum: S(f) ∝ 1/f
    Generated by filtering white noise
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center (avoid division by zero)
    r = np.sqrt(xx**2 + yy**2)
    r[r == 0] = r[r > 0].min()
    
    # 1/f filter in frequency domain
    noise_fft = np.fft.fft2(white_noise)
    filt = 1 / r
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.3 Brown Noise (Random Walk Noise)
def add_brown_noise(image, intensity=1.0):
    """
    Formula: B(t) = B(t-1) + W(t), where W(t) is white noise
    Power spectrum: S(f) ∝ 1/f²
    """
    row, col = image.shape
    
    # Generate 1D Brown noise
    brown_1d = np.zeros(row * col)
    white_noise = np.random.normal(0, 1, row * col)
    
    for i in range(1, row * col):
        brown_1d[i] = brown_1d[i-1] + white_noise[i] * intensity
    
    # Normalize and reshape
    brown_1d = (brown_1d - brown_1d.min()) / (brown_1d.max() - brown_1d.min())
    noise = (brown_1d.reshape(row, col) * 100) - 50
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.4 Blue Noise
def add_blue_noise(image, intensity=20):
    """
    Formula: Power spectrum: S(f) ∝ f
    Generated by high-pass filtering white noise
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center
    r = np.sqrt(xx**2 + yy**2)
    
    # Blue noise filter (emphasize high frequencies)
    noise_fft = np.fft.fft2(white_noise)
    filt = r
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.5 Violet Noise (Purple Noise)
def add_violet_noise(image, intensity=10):
    """
    Formula: Power spectrum: S(f) ∝ f²
    Generated by double-differentiation of white noise
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center
    r = np.sqrt(xx**2 + yy**2)
    
    # Violet noise filter (emphasize very high frequencies)
    noise_fft = np.fft.fft2(white_noise)
    filt = r**2
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.6 Grey Noise
def add_grey_noise(image, intensity=25):
    """
    Formula: White noise filtered to match human auditory perception
    In images, can be simulated with perceptual weighting
    """
    row, col = image.shape
    
    # Generate base white noise
    noise = np.random.normal(0, intensity, (row, col))
    
    # Apply perceptual weighting (simple approximation)
    y = np.linspace(0, 1, row)
    perceptual_weight = np.sqrt(y).reshape(-1, 1)
    weighted_noise = noise * perceptual_weight
    
    noisy_img = image + weighted_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), weighted_noise

# 2.7 Red Noise (Brownian Noise variant)
def add_red_noise(image, intensity=20, alpha=2.0):
    """
    Formula: Power spectrum: S(f) ∝ 1/f^α where α ≈ 2
    Similar to Brown noise but with adjustable exponent
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center (avoid division by zero)
    r = np.sqrt(xx**2 + yy**2)
    r[r == 0] = r[r > 0].min()
    
    # 1/f^alpha filter
    noise_fft = np.fft.fft2(white_noise)
    filt = 1 / (r**alpha)
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.8 Orange Noise
def add_orange_noise(image, intensity=20, alpha=1.0):
    """
    Formula: Power spectrum: S(f) ∝ 1/f^α where α ≈ 1
    Between red and pink noise, less bass-heavy than red noise
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center (avoid division by zero)
    r = np.sqrt(xx**2 + yy**2)
    r[r == 0] = r[r > 0].min()
    
    # 1/f^alpha filter
    noise_fft = np.fft.fft2(white_noise)
    filt = 1 / (r**alpha)
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.9 Green Noise
def add_green_noise(image, intensity=25, band_center=0.5, band_width=0.1):
    """
    Formula: Band-limited noise with mid-range frequencies only
    """
    row, col = image.shape
    
    # Generate 2D white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Create frequency domain filter
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    xx = xx - 0.5
    yy = yy - 0.5
    
    # Distance from center
    r = np.sqrt(xx**2 + yy**2)
    
    # Band-pass filter centered on mid-range frequencies
    filt = np.exp(-((r - band_center)**2) / (2 * band_width**2))
    
    # Apply filter in frequency domain
    noise_fft = np.fft.fft2(white_noise)
    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt
    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)
    
    # Inverse FFT
    noise = np.real(np.fft.ifft2(noise_fft_filtered))
    
    # Normalize and apply intensity
    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 2.10 Black Noise
def add_black_noise(image, intensity=10, sparsity=0.99):
    """
    Formula: Silence with occasional spikes
    S(f) = 0 for most f, with rare, unpredictable spikes
    """
    row, col = image.shape
    
    # Generate base noise
    base_noise = np.random.normal(0, 1, (row, col))
    
    # Create sparse mask
    mask = np.random.random((row, col)) > sparsity
    
    # Apply mask to create sparse noise
    noise = np.zeros((row, col))
    noise[mask] = base_noise[mask] * intensity * 5  # Amplify the sparse points
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 3. PROCEDURAL AND SYNTHETIC NOISE TYPES
# -----------------------------------------------

# 3.1 Perlin Noise
def add_perlin_noise(image, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, intensity=50):
    """
    Formula: Complex procedural noise function using gradient interpolation
    P(x,y) = Σᵢ₌₁ᵗᵒ ₙ (persistence^i) * noise(x*lacunarity^i, y*lacunarity^i)
    """
    if not has_perlin:
        # Generate alternative to Perlin noise if package not available
        return add_simplex_noise(image, scale, octaves, persistence, lacunarity, intensity)
        
    row, col = image.shape
    
    # Initialize Perlin noise generator
    noise_gen = PerlinNoise(octaves=octaves, seed=42)
    
    # Generate noise
    noise = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            noise[i][j] = noise_gen([i/scale, j/scale])
    
    # Scale noise to desired intensity
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.2 Simplex Noise (alternative to Perlin)
def add_simplex_noise(image, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, intensity=50):
    """
    Formula: Improved version of Perlin noise with better computational properties
    Similar formula but different grid structure
    """
    row, col = image.shape
    
    # Generate approximation using filtered noise
    freq = 1.0 / scale
    noise = np.zeros((row, col))
    amplitude = 1.0
    
    for _ in range(octaves):
        # Generate base noise at current frequency
        base = gaussian_filter(np.random.randn(row, col), sigma=1.0/freq)
        
        # Add to accumulated noise
        noise += amplitude * base
        
        # Update parameters for next octave
        amplitude *= persistence
        freq *= lacunarity
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.3 Worley Noise (Cellular Noise)
def add_worley_noise(image, n_points=20, intensity=50):
    """
    Formula: Based on distance to nearest feature point
    W(x,y) = min(dist((x,y), (x_i,y_i))) for all feature points i
    """
    row, col = image.shape
    
    # Generate random feature points
    points = np.random.rand(n_points, 2)
    points[:, 0] *= row
    points[:, 1] *= col
    
    # Calculate distance to nearest feature point for each pixel
    xx, yy = np.meshgrid(np.arange(col), np.arange(row))
    noise = np.ones((row, col)) * np.inf
    
    for p in points:
        dist = np.sqrt((yy - p[0])**2 + (xx - p[1])**2)
        noise = np.minimum(noise, dist)
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * (1 - noise) - intensity/2  # Invert so cells are darker
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.4 Fractional Brownian Motion (fBm)
def add_fbm_noise(image, H=0.7, octaves=8, intensity=40):
    """
    Formula: Sum of octaves of noise with frequency and amplitude scaling
    fBm(x,y) = Σᵢ₌₀ᵗᵒ ₙ₋₁ amplitude_i * noise(frequency_i * (x,y))
    where amplitude_i = persistence^(i*H) and frequency_i = lacunarity^i
    """
    row, col = image.shape
    
    # Parameters
    persistence = 0.5
    lacunarity = 2.0
    
    noise = np.zeros((row, col))
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        # Generate noise at current frequency
        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)
        
        # Add to accumulated noise
        noise += amplitude * noise_layer
        
        # Update parameters for next octave
        amplitude *= persistence ** H
        frequency *= lacunarity
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.5 Value Noise
def add_value_noise(image, octaves=4, persistence=0.5, scale=20, intensity=40):
    """
    Formula: Interpolation between random values at grid vertices
    Similar to Perlin but using value interpolation instead of gradient interpolation
    """
    row, col = image.shape
    
    noise = np.zeros((row, col))
    amplitude = 1.0
    frequency = 1.0
    
    for _ in range(octaves):
        # Generate random values at grid points
        grid_size = int(scale / frequency)
        if grid_size < 2:
            grid_size = 2
            
        # Create grid of random values
        grid = np.random.rand(grid_size, grid_size)
        
        # Resize to image dimensions using bilinear interpolation
        layer = cv2.resize(grid, (col, row), interpolation=cv2.INTER_LINEAR)
        
        # Add to noise with current amplitude
        noise += layer * amplitude
        
        # Update for next octave
        amplitude *= persistence
        frequency *= 2.0
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.6 Wavelet Noise
def add_wavelet_noise(image, intensity=30):
    """
    Formula: Based on wavelet decomposition and reconstruction
    Uses wavelet transforms to generate band-limited noise
    """
    row, col = image.shape
    
    # Generate white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Perform wavelet transform (using 'db4' wavelet, level 3)
    coeffs = pywt.wavedec2(white_noise, 'db4', level=3)
    
    # Modify coefficients (without directly modifying the tuple)
    # Create a new list of coefficients
    new_coeffs = [coeffs[0]]  # Keep approximation coefficients
    
    # Modify detail coefficients by creating new tuples
    for i in range(1, len(coeffs)):
        detail_coeffs = coeffs[i]
        new_detail = (detail_coeffs[0] * 0.8, 
                     detail_coeffs[1] * 0.8, 
                     detail_coeffs[2] * 0.8)
        new_coeffs.append(new_detail)
    
    # Reconstruct signal
    noise = pywt.waverec2(new_coeffs, 'db4')
    
    # Crop to original size if dimensions changed
    noise = noise[:row, :col]
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.7 Diamond-Square Noise
def add_diamond_square_noise(image, roughness=0.5, intensity=40):
    """
    Formula: Recursive subdivision algorithm with random displacement
    Also known as plasma noise or cloud noise
    """
    # Get closest power of 2 plus 1
    row, col = image.shape
    size = max(row, col)
    power = int(np.ceil(np.log2(size - 1)))
    n = 2**power + 1
    
    # Create grid for diamond-square algorithm
    grid = np.zeros((n, n))
    
    # Set four corners to random values
    grid[0, 0] = np.random.random()
    grid[0, n-1] = np.random.random()
    grid[n-1, 0] = np.random.random()
    grid[n-1, n-1] = np.random.random()
    
    # Run diamond-square algorithm
    step = n - 1
    while step > 1:
        half_step = step // 2
        
        # Diamond step
        for i in range(half_step, n, step):
            for j in range(half_step, n, step):
                avg = (grid[i-half_step, j-half_step] + 
                       grid[i-half_step, j+half_step] + 
                       grid[i+half_step, j-half_step] + 
                       grid[i+half_step, j+half_step]) / 4.0
                grid[i, j] = avg + (np.random.random() - 0.5) * roughness * step
        
        # Square step
        for i in range(0, n, half_step):
            j_start = half_step if i % step == 0 else 0
            for j in range(j_start, n, step):
                # Count valid neighbors
                count = 0
                avg = 0
                
                if i - half_step >= 0:
                    avg += grid[i - half_step, j]
                    count += 1
                if i + half_step < n:
                    avg += grid[i + half_step, j]
                    count += 1
                if j - half_step >= 0:
                    avg += grid[i, j - half_step]
                    count += 1
                if j + half_step < n:
                    avg += grid[i, j + half_step]
                    count += 1
                
                avg /= count
                grid[i, j] = avg + (np.random.random() - 0.5) * roughness * step
        
        step = half_step
        roughness *= 0.5
    
    # Crop to original size
    noise = grid[:row, :col]
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.8 Gabor Noise
def add_gabor_noise(image, k_scale=5.0, intensity=20, orient=0, freq=0.1):
    """
    Formula: Based on summation of randomly positioned Gabor kernels
    """
    row, col = image.shape
    
    # Generate random positions
    n_kernels = 300
    positions = np.random.rand(n_kernels, 2)
    positions[:, 0] *= row
    positions[:, 1] *= col
    
    # Create grid
    yy, xx = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    
    # Initialize noise
    noise = np.zeros((row, col))
    
    # Parameters for Gabor kernel
    sigma = 1.0 / freq
    theta = orient * np.pi / 180  # Convert to radians
    lambda_val = 1.0 / freq
    gamma = 1.0  # Aspect ratio
    
    # Calculate rotated coordinates once
    x_theta = xx * np.cos(theta) + yy * np.sin(theta)
    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
    
    # Create Gabor kernel
    gabor = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lambda_val)
    
    # Generate noise by summing shifted Gabor kernels
    for pos in positions:
        y, x = int(pos[0]), int(pos[1])
        if 0 <= y < row and 0 <= x < col:
            # Random weight
            weight = np.random.normal(0, 1)
            
            # Calculate bounds
            kernel_size = int(k_scale * sigma)
            y1 = max(0, y - kernel_size)
            y2 = min(row, y + kernel_size + 1)
            x1 = max(0, x - kernel_size)
            x2 = min(col, x + kernel_size + 1)
            
            # Calculate kernel bounds
            ky1 = kernel_size - (y - y1)
            ky2 = kernel_size + (y2 - y)
            kx1 = kernel_size - (x - x1)
            kx2 = kernel_size + (x2 - x)
            
            # Ensure bounds are valid
            if y2 > y1 and x2 > x1 and ky2 > ky1 and kx2 > kx1:
                gabor_small = gabor[ky1:ky2, kx1:kx2]
                try:
                    noise[y1:y2, x1:x2] += weight * gabor_small
                except ValueError:
                    # Handle potential dimension mismatch
                    continue
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.9 Sparse Convolution Noise
def add_sparse_convolution_noise(image, intensity=30, density=0.05, kernel_size=15):
    """
    Formula: Convolution of sparse impulses with a kernel function
    """
    row, col = image.shape
    
    # Create sparse impulse image
    impulses = np.zeros((row, col))
    mask = np.random.random((row, col)) < density
    impulses[mask] = np.random.normal(0, 1, size=np.sum(mask))
    
    # Create kernel
    x = np.linspace(-1, 1, kernel_size)
    y = np.linspace(-1, 1, kernel_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    kernel = np.exp(-4 * r**2)
    kernel /= kernel.sum()
    
    # Convolve impulses with kernel
    noise = ndimage.convolve(impulses, kernel, mode='reflect')
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.10 Turbulence Noise
def add_turbulence_noise(image, octaves=6, intensity=40):
    """
    Formula: Absolute value of fBm
    T(x,y) = |fBm(x,y)|
    """
    row, col = image.shape
    
    # Parameters for fBm
    H = 0.5
    persistence = 0.5
    lacunarity = 2.0
    
    noise = np.zeros((row, col))
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        # Generate noise at current frequency
        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)
        
        # Add absolute value to accumulated noise (turbulence formula)
        noise += amplitude * np.abs(noise_layer)
        
        # Update parameters for next octave
        amplitude *= persistence ** H
        frequency *= lacunarity
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.11 Mosaic Noise (Voronoi-based)
def add_mosaic_noise(image, n_points=20, intensity=40):
    """
    Formula: Based on Voronoi diagram 
    Similar to Worley but using cell index instead of distance
    """
    row, col = image.shape
    
    # Generate random seed points
    points = np.random.rand(n_points, 2)
    points[:, 0] *= row
    points[:, 1] *= col
    
    # Create Voronoi cells
    xx, yy = np.meshgrid(np.arange(col), np.arange(row))
    noise = np.zeros((row, col))
    
    # For each pixel, find the closest seed point
    for i in range(n_points):
        dist = np.sqrt((yy - points[i, 0])**2 + (xx - points[i, 1])**2)
        
        # For the first point, initialize the closest indices and distances
        if i == 0:
            closest_idx = np.zeros((row, col), dtype=int)
            closest_dist = dist
        else:
            # Update if this point is closer
            mask = dist < closest_dist
            closest_dist[mask] = dist[mask]
            closest_idx[mask] = i
    
    # Assign random values to each cell
    cell_values = np.random.random(n_points) * 2 - 1  # Range [-1, 1]
    for i in range(n_points):
        noise[closest_idx == i] = cell_values[i]
    
    # Scale to desired intensity
    noise = intensity * noise
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.12 Ridge Noise
def add_ridge_noise(image, octaves=6, intensity=40):
    """
    Formula: Ridge = 1 - |fBm|
    Modified turbulence noise with ridge formation
    """
    row, col = image.shape
    
    # Parameters for fBm
    H = 1.0
    persistence = 0.5
    lacunarity = 2.0
    
    noise = np.zeros((row, col))
    amplitude = 1.0
    frequency = 1.0
    
    for i in range(octaves):
        # Generate noise at current frequency
        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)
        
        # Add ridge formula to accumulated noise
        noise += amplitude * (1.0 - np.abs(noise_layer))
        
        # Update parameters for next octave
        amplitude *= persistence ** H
        frequency *= lacunarity
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.13 Curl Noise
def add_curl_noise(image, scale=10.0, intensity=25):
    """
    Formula: Based on the curl of a potential field
    Curl(P) = (∂P_y/∂x - ∂P_x/∂y)
    """
    row, col = image.shape
    
    # Generate two potential fields
    p_x = gaussian_filter(np.random.randn(row+2, col+2), sigma=scale)
    p_y = gaussian_filter(np.random.randn(row+2, col+2), sigma=scale)
    
    # Calculate derivatives
    dy_px = p_x[2:, 1:-1] - p_x[:-2, 1:-1]
    dx_py = p_y[1:-1, 2:] - p_y[1:-1, :-2]
    
    # Calculate curl
    noise = dx_py - dy_px
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.14 Caustic Noise
def add_caustic_noise(image, scale=10.0, intensity=30, distortion=2.0):
    """
    Formula: Based on refracted/reflected light patterns
    Simulated using deformed noise fields
    """
    row, col = image.shape
    
    # Generate base noise
    base_noise = gaussian_filter(np.random.randn(row, col), sigma=scale)
    
    # Generate displacement fields
    disp_x = gaussian_filter(np.random.randn(row, col), sigma=scale*2) * distortion
    disp_y = gaussian_filter(np.random.randn(row, col), sigma=scale*2) * distortion
    
    # Create grid
    y, x = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    
    # Apply displacement
    x_new = np.clip(x + disp_x, 0, col-1).astype(int)
    y_new = np.clip(y + disp_y, 0, row-1).astype(int)
    
    # Sample from base noise with displacement
    noise = base_noise[y_new, x_new]
    
    # Enhance contrast to create caustic effect
    noise = np.tanh(noise * 3)
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 3.15 Cracks Noise
def add_cracks_noise(image, n_cracks=50, length=30, width=1, intensity=100):
    """
    Formula: Random walk-based crack pattern
    """
    row, col = image.shape
    
    # Initialize empty noise
    noise = np.zeros((row, col))
    
    # Generate cracks
    for _ in range(n_cracks):
        # Random starting point
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        
        # Random direction
        angle = np.random.random() * 2 * np.pi
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Generate crack as a random walk
        for i in range(length):
            # Small random perturbation to direction
            angle += np.random.normal(0, 0.1)
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Move to new position
            x += dx
            y += dy
            
            # Check boundaries
            if x < 0 or x >= col or y < 0 or y >= row:
                break
                
            # Draw point at current position
            x_int, y_int = int(x), int(y)
            
            # Draw with width
            for wy in range(-width, width+1):
                for wx in range(-width, width+1):
                    xi, yi = x_int + wx, y_int + wy
                    if 0 <= xi < col and 0 <= yi < row:
                        noise[yi, xi] = 1.0
    
    # Normalize and scale
    noise = intensity * noise - intensity/2
    
    noisy_img = image - noise  # Subtract to make cracks dark
    return np.clip(noisy_img, 0, 255).astype(np.uint8), -noise

# 3.16 Flow Noise
def add_flow_noise(image, scale=10.0, intensity=30, iterations=3):
    """
    Formula: Noise advected along a flow field
    """
    row, col = image.shape
    
    # Generate base noise
    noise = gaussian_filter(np.random.randn(row, col), sigma=scale)
    
    # Generate flow field (two components)
    flow_x = gaussian_filter(np.random.randn(row, col), sigma=scale*2)
    flow_y = gaussian_filter(np.random.randn(row, col), sigma=scale*2)
    
    # Create grid
    y, x = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')
    
    # Advect noise along flow field iteratively
    for _ in range(iterations):
        # Calculate new positions
        x_new = np.clip(x + flow_x, 0, col-1)
        y_new = np.clip(y + flow_y, 0, row-1)
        
        # Sample noise at new positions using bilinear interpolation
        x0 = np.floor(x_new).astype(int)
        y0 = np.floor(y_new).astype(int)
        x1 = np.minimum(x0 + 1, col-1)
        y1 = np.minimum(y0 + 1, row-1)
        
        wx = x_new - x0
        wy = y_new - y0
        
        # Bilinear interpolation
        new_noise = (noise[y0, x0] * (1-wx) * (1-wy) +
                    noise[y0, x1] * wx * (1-wy) +
                    noise[y1, x0] * (1-wx) * wy +
                    noise[y1, x1] * wx * wy)
        
        noise = new_noise
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 4. PHYSICAL AND DEVICE-BASED NOISE TYPES
# -----------------------------------------------

# 4.1 Film Grain Noise
def add_film_grain(image, intensity=0.5, size=3):
    """
    Formula: Approximated by filtered noise with log-normal characteristics
    """
    row, col = image.shape
    
    # Generate base noise (log-normal distribution)
    mu, sigma = 0, 0.5
    base_noise = np.random.lognormal(mu, sigma, (row, col)) - np.exp(mu + sigma**2/2)
    
    # Blur the noise to create grain structure
    grain = gaussian_filter(base_noise, sigma=size)
    
    # Normalize and scale
    grain = (grain - grain.min()) / (grain.max() - grain.min())
    
    # Apply grain (more visible in darker areas)
    darkness = 1.0 - (image / 255.0)
    weighted_grain = grain * darkness * intensity * 255
    
    noisy_img = image + weighted_grain
    return np.clip(noisy_img, 0, 255).astype(np.uint8), weighted_grain

# 4.2 CCD/CMOS Sensor Noise (continued)
def add_sensor_noise(image, shot_scale=2.0, read_sigma=5, dark_scale=0.2):
    """
    Formula: Combination of shot noise, read noise, and dark current noise
    g(x,y) = Poisson(f(x,y)/s1)*s1 + N(0,σ²) + exp(d)*darkness(x,y)
    """
    row, col = image.shape
    
    # Shot noise (photon noise)
    img_data = np.maximum(image, 0) / shot_scale
    shot_noise = np.random.poisson(img_data) * shot_scale - img_data * shot_scale
    
    # Read noise (electronics)
    read_noise = np.random.normal(0, read_sigma, (row, col))
    
    # Dark current noise (temperature dependent, more in shadows)
    darkness = 1.0 - (image / 255.0)
    dark_current = np.random.exponential(dark_scale, (row, col)) * darkness * 20
    
    # Combine all noise components
    total_noise = shot_noise + read_noise + dark_current
    noisy_img = image + total_noise
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), total_noise

# 4.3 Thermal Noise
def add_thermal_noise(image, mean=0, sigma=15, temp_factor=0.8):
    """
    Formula: Temperature-dependent Gaussian noise
    g(x,y) = f(x,y) + N(μ,σ²*T)
    """
    row, col = image.shape
    
    # Create temperature gradient (hotter at bottom, for example)
    temp_gradient = np.linspace(0.5, 1.0, row)[:, np.newaxis] * temp_factor
    
    # Adjust noise variance based on temperature
    local_sigma = sigma * temp_gradient
    
    # Generate noise
    noise = np.zeros((row, col))
    for i in range(row):
        noise[i, :] = np.random.normal(mean, local_sigma[i], col)
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 4.4 Fixed Pattern Noise
def add_fixed_pattern_noise(image, intensity=15, pattern_seed=42):
    """
    Formula: Fixed spatial noise pattern
    g(x,y) = f(x,y) + P(x,y)
    """
    row, col = image.shape
    
    # Set random seed to ensure pattern is fixed
    np.random.seed(pattern_seed)
    
    # Generate fixed pattern
    pattern = np.random.normal(0, 1, (row, col))
    
    # Apply some smoothing to make it more realistic
    pattern = gaussian_filter(pattern, sigma=1.0)
    
    # Normalize and scale
    pattern = intensity * (pattern - pattern.min()) / (pattern.max() - pattern.min()) - intensity/2
    
    # Reset random seed
    np.random.seed()
    
    noisy_img = image + pattern
    return np.clip(noisy_img, 0, 255).astype(np.uint8), pattern

# 4.5 Row/Column Noise
def add_row_column_noise(image, row_sigma=10, col_sigma=5, row_prob=0.1, col_prob=0.05):
    """
    Formula: Noise affecting entire rows/columns
    g(x,y) = f(x,y) + R(y) + C(x)
    """
    row, col = image.shape
    
    # Initialize noise
    noise = np.zeros((row, col))
    
    # Add row noise
    row_noise = np.zeros(row)
    row_affected = np.random.random(row) < row_prob
    row_noise[row_affected] = np.random.normal(0, row_sigma, np.sum(row_affected))
    noise += row_noise[:, np.newaxis]
    
    # Add column noise
    col_noise = np.zeros(col)
    col_affected = np.random.random(col) < col_prob
    col_noise[col_affected] = np.random.normal(0, col_sigma, np.sum(col_affected))
    noise += col_noise[np.newaxis, :]
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 4.6 Banding Noise
def add_banding_noise(image, amplitude=15, frequency=0.1, orientation='horizontal'):
    """
    Formula: Periodic pattern across image
    g(x,y) = f(x,y) + A*sin(2πfx) or g(x,y) = f(x,y) + A*sin(2πfy)
    """
    row, col = image.shape
    
    # Create coordinate grid
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    
    # Generate banding pattern
    if orientation == 'horizontal':
        noise = amplitude * np.sin(2 * np.pi * frequency * row * yy)
    elif orientation == 'vertical':
        noise = amplitude * np.sin(2 * np.pi * frequency * col * xx)
    else:  # diagonal
        noise = amplitude * np.sin(2 * np.pi * frequency * (xx + yy))
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 4.7 JPEG Compression Noise
def add_jpeg_noise(image, quality=40):
    """
    Formula: Artifacts from lossy JPEG compression
    """
    # Ensure we have a valid image
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save to memory buffer with JPEG compression
    ret, buffer = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # Decode from buffer
    compressed_img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    
    # Calculate noise (difference between original and compressed)
    noise = compressed_img.astype(float) - image
    
    return compressed_img, noise

# 4.8 Quantization Noise
def add_quantization_noise(image, levels=16):
    """
    Formula: Error introduced by reducing bit depth
    g(x,y) = round(f(x,y) * levels / 255) * 255 / levels
    """
    # Quantize the image
    quantized = np.round(image * (levels / 255.0)) * (255.0 / levels)
    quantized = quantized.astype(np.uint8)
    
    # Calculate noise (difference between original and quantized)
    noise = quantized.astype(float) - image
    
    return quantized, noise

# 4.9 Demosaicing Noise
def add_demosaicing_noise(image, pattern_size=2):
    """
    Formula: Artifacts from Bayer pattern interpolation
    """
    row, col = image.shape
    
    # Create a simulated Bayer pattern (temporarily reduce resolution)
    bayer = np.zeros((row, col), dtype=np.uint8)
    
    # Simulate R, G, B channels with a Bayer pattern
    # (RGGB pattern: R at (0,0), G at (0,1) and (1,0), B at (1,1))
    for i in range(0, row, pattern_size):
        for j in range(0, col, pattern_size):
            if i+1 < row and j+1 < col:
                # R channel
                bayer[i, j] = image[i, j]
                
                # G channels (two positions)
                if i+1 < row:
                    bayer[i+1, j] = image[i+1, j]
                if j+1 < col:
                    bayer[i, j+1] = image[i, j+1]
                
                # B channel
                if i+1 < row and j+1 < col:
                    bayer[i+1, j+1] = image[i+1, j+1]
    
    # Simple interpolation to simulate demosaicing
    interpolated = cv2.resize(
        cv2.resize(bayer, (col//pattern_size, row//pattern_size), 
                  interpolation=cv2.INTER_AREA),
        (col, row), interpolation=cv2.INTER_LINEAR)
    
    # Calculate noise (difference between original and interpolated)
    noise = interpolated.astype(float) - image
    
    return interpolated, noise

# 4.10 Lens Blur
def add_lens_blur(image, radius=3):
    """
    Formula: Convolution with disk kernel
    g(x,y) = f(x,y) ⊗ K(r)
    """
    # Create disk kernel
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2*radius+1, 2*radius+1))
    kernel[mask] = 1
    kernel /= kernel.sum()
    
    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)
    
    # Calculate "noise" (difference between original and blurred)
    noise = blurred.astype(float) - image
    
    return blurred, noise

# 4.11 Motion Blur
def add_motion_blur(image, length=15, angle=45):
    """
    Formula: Convolution with motion kernel
    g(x,y) = f(x,y) ⊗ M(length, angle)
    """
    # Create motion blur kernel
    kernel = np.zeros((length, length))
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate x, y components of the directional vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Create line of ones for the kernel
    center = length // 2
    for i in range(length):
        x = int(center + dx * (i - center))
        y = int(center + dy * (i - center))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    
    # Normalize kernel
    kernel /= kernel.sum()
    
    # Apply motion blur
    blurred = cv2.filter2D(image, -1, kernel)
    
    # Calculate "noise" (difference between original and blurred)
    noise = blurred.astype(float) - image
    
    return blurred, noise

# 4.12 Vignetting
def add_vignetting(image, strength=0.5):
    """
    Formula: Darkening around the edges of the image
    g(x,y) = f(x,y) * V(x,y) where V decreases with distance from center
    """
    row, col = image.shape
    
    # Create coordinate grid
    y, x = np.ogrid[:row, :col]
    
    # Calculate center
    center_y, center_x = row / 2, col / 2
    
    # Calculate squared distance from center
    dist_squared = (x - center_x)**2 + (y - center_y)**2
    
    # Normalize distance to [0, 1] range
    max_dist_squared = (max(center_x, center_y))**2
    normalized_dist = dist_squared / max_dist_squared
    
    # Create vignette mask
    vignette = 1 - strength * normalized_dist
    
    # Apply vignette
    vignetted = image * vignette
    
    # Calculate "noise" (difference due to vignetting)
    noise = vignetted - image
    
    return vignetted.astype(np.uint8), noise

# 4.13 Blooming
def add_blooming(image, threshold=200, spread=5):
    """
    Formula: Overflow of bright areas into neighboring pixels
    g(x,y) = f(x,y) + B(x,y) where B is the blooming effect
    """
    # Find bright areas
    bright_mask = image > threshold
    
    # Dilate bright areas to simulate blooming
    blooming_mask = cv2.dilate(bright_mask.astype(np.uint8), 
                              np.ones((spread, spread), np.uint8))
    
    # Create blooming effect (bright regions spread)
    blooming = np.zeros_like(image, dtype=float)
    blooming[blooming_mask > 0] = 255
    
    # Smooth the blooming
    blooming = gaussian_filter(blooming, sigma=spread/2)
    
    # Apply blooming only to areas not already bright
    effect_mask = (blooming_mask > 0) & (~bright_mask)
    blooming_effect = np.zeros_like(image, dtype=float)
    blooming_effect[effect_mask] = blooming[effect_mask] * 0.5
    
    # Add blooming to original image
    bloomed = image + blooming_effect
    
    return np.clip(bloomed, 0, 255).astype(np.uint8), blooming_effect

# 4.14 Chromatic Aberration (for demonstration, though implemented for grayscale)
def add_chromatic_aberration(image, shift=3):
    """
    Formula: Color channels shifted relative to each other
    For grayscale, we simulate by adding opposing edge responses
    """
    # Create edge-detected version
    edges = cv2.Canny(image, 50, 150).astype(float)
    
    # Create two shifted versions (one positive, one negative)
    shifted_pos = np.zeros_like(image, dtype=float)
    shifted_neg = np.zeros_like(image, dtype=float)
    
    if shift > 0:
        shifted_pos[:-shift, :-shift] = edges[shift:, shift:]
        shifted_neg[shift:, shift:] = -edges[:-shift, :-shift]
    
    # Combine effects for visualization in grayscale
    aberration = shifted_pos + shifted_neg
    
    # Apply to image
    aberrated = image + aberration * 0.5
    
    return np.clip(aberrated, 0, 255).astype(np.uint8), aberration

# 4.15 Hot Pixels
def add_hot_pixels(image, density=0.001, intensity=255):
    """
    Formula: Random pixels with very high values
    g(x,y) = intensity if (x,y) is a hot pixel, f(x,y) otherwise
    """
    row, col = image.shape
    
    # Create hot pixel mask
    hot_mask = np.random.random((row, col)) < density
    
    # Create hot pixel effect
    hot_pixels = np.zeros_like(image, dtype=float)
    hot_pixels[hot_mask] = intensity
    
    # Add hot pixels to image
    noisy_img = image.copy()
    noisy_img[hot_mask] = intensity
    
    # Calculate noise (just the hot pixels)
    noise = hot_pixels - image * hot_mask
    
    return noisy_img, noise

# 4.16 Dead Pixels
def add_dead_pixels(image, density=0.001):
    """
    Formula: Random pixels with zero value
    g(x,y) = 0 if (x,y) is a dead pixel, f(x,y) otherwise
    """
    row, col = image.shape
    
    # Create dead pixel mask
    dead_mask = np.random.random((row, col)) < density
    
    # Create dead pixel effect
    dead_pixels = np.zeros_like(image, dtype=float)
    
    # Add dead pixels to image
    noisy_img = image.copy()
    noisy_img[dead_mask] = 0
    
    # Calculate noise (negative of the original values at dead pixel locations)
    noise = dead_pixels - image * dead_mask
    
    return noisy_img, noise

# 4.17 Rolling Shutter Effect
def add_rolling_shutter(image, amplitude=5, frequency=1):
    """
    Formula: Horizontal shifting of rows based on sinusoidal pattern
    """
    row, col = image.shape
    result = np.zeros_like(image)
    
    # Create row-dependent shift
    shifts = (amplitude * np.sin(2 * np.pi * frequency * np.arange(row) / row)).astype(int)
    
    # Apply row-wise shifts
    for i in range(row):
        shift = shifts[i]
        if shift >= 0:
            result[i, shift:] = image[i, :(col-shift)]
        else:
            result[i, :col+shift] = image[i, -shift:]
    
    # Calculate noise (difference between original and shifted)
    noise = result.astype(float) - image
    
    return result, noise

# 4.18 Moiré Pattern
def add_moire_pattern(image, freq1=10, freq2=11, angle1=0, angle2=5, amplitude=20):
    """
    Formula: Interference pattern from superimposed regular patterns
    g(x,y) = f(x,y) + A*sin(2πf₁(x*cos(θ₁)+y*sin(θ₁)))*sin(2πf₂(x*cos(θ₂)+y*sin(θ₂)))
    """
    row, col = image.shape
    
    # Create coordinate grid
    y, x = np.mgrid[:row, :col]
    
    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)
    
    # Create pattern 1
    pattern1 = np.sin(2 * np.pi * freq1 * (x * np.cos(angle1_rad) + y * np.sin(angle1_rad)) / col)
    
    # Create pattern 2
    pattern2 = np.sin(2 * np.pi * freq2 * (x * np.cos(angle2_rad) + y * np.sin(angle2_rad)) / col)
    
    # Combine patterns to create moiré
    noise = amplitude * pattern1 * pattern2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 5. REAL-WORLD AND ENVIRONMENTAL NOISE TYPES
# -----------------------------------------------

# 5.1 Rain Noise
def add_rain_noise(image, density=0.01, length=10, brightness=200, angle=70):
    """
    Formula: Bright streaks simulating rain
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    dx = int(np.cos(angle_rad) * length)
    dy = int(np.sin(angle_rad) * length)
    
    # Number of raindrops
    num_drops = int(row * col * density)
    
    for _ in range(num_drops):
        # Random raindrop position
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        
        # Draw raindrop streak
        for i in range(length):
            new_x = x + int(i * np.cos(angle_rad))
            new_y = y + int(i * np.sin(angle_rad))
            
            if 0 <= new_x < col and 0 <= new_y < row:
                # Fade intensity along streak
                intensity = brightness * (1 - i/length)
                noise[new_y, new_x] = intensity
    
    # Add rain to image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 5.2 Snow Noise
def add_snow_noise(image, density=0.01, size_range=(1, 3), brightness=200):
    """
    Formula: Small white specks simulating snow
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    
    # Number of snowflakes
    num_flakes = int(row * col * density)
    
    for _ in range(num_flakes):
        # Random snowflake position
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        
        # Random snowflake size
        size = np.random.randint(size_range[0], size_range[1] + 1)
        
        # Draw snowflake (small white disk)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx**2 + dy**2 <= size**2:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < col and 0 <= new_y < row:
                        noise[new_y, new_x] = brightness
    
    # Add snow to image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 5.3 Fog/Haze
def add_fog(image, intensity=0.5):
    """
    Formula: g(x,y) = f(x,y)*(1-t) + L*t
    where t is fog transmission and L is fog light
    """
    # Create transmission map (decreases with distance)
    trans = np.ones_like(image, dtype=float) * (1 - intensity)
    
    # Fog light (usually bright gray)
    fog_light = 240
    
    # Apply fog model
    foggy = image * trans + fog_light * (1 - trans)
    
    # Calculate noise (difference due to fog)
    noise = foggy - image
    
    return np.clip(foggy, 0, 255).astype(np.uint8), noise

# 5.4 Dust and Scratches
def add_dust_scratches(image, dust_density=0.001, scratch_count=20):
    """
    Formula: Combination of random specks (dust) and lines (scratches)
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    
    # Add dust
    dust_mask = np.random.random((row, col)) < dust_density
    dust_sizes = np.random.randint(1, 5, size=np.sum(dust_mask))
    
    # Place dust with varying sizes
    dust_indices = np.where(dust_mask)
    for i, (y, x) in enumerate(zip(*dust_indices)):
        size = dust_sizes[i]
        y1, y2 = max(0, y-size), min(row, y+size+1)
        x1, x2 = max(0, x-size), min(col, x+size+1)
        
        # Draw dust speck with random intensity
        intensity = np.random.randint(-150, -50)
        for dy in range(y1, y2):
            for dx in range(x1, x2):
                if (dy-y)**2 + (dx-x)**2 <= size**2:
                    noise[dy, dx] = intensity
    
    # Add scratches
    for _ in range(scratch_count):
        # Random scratch starting point
        x1, y1 = np.random.randint(0, col), np.random.randint(0, row)
        
        # Random scratch length and angle
        length = np.random.randint(10, 50)
        angle = np.random.random() * 2 * np.pi
        
        # Calculate end point
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Draw scratch line
        rr, cc = line(y1, x1, y2, x2)
        valid_idx = (rr >= 0) & (rr < row) & (cc >= 0) & (cc < col)
        rr, cc = rr[valid_idx], cc[valid_idx]
        
        # Set scratch intensity (usually dark)
        intensity = np.random.randint(-150, -50)
        noise[rr, cc] = intensity
    
    # Apply to image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# Helper function for dust_scratches
def line(y0, x0, y1, x1):
    """
    Implementation of Bresenham's line algorithm
    Returns coordinates of the line from (x0, y0) to (x1, y1)
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    line_x = []
    line_y = []
    
    while True:
        line_y.append(y0)
        line_x.append(x0)
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    
    return np.array(line_y), np.array(line_x)

# 5.5 Lens Flare
def add_lens_flare(image, intensity=0.7, num_flares=5):
    """
    Formula: Bright spots and streaks simulating lens flare
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    
    # Create a main flare source
    center_y, center_x = row // 2, col // 2
    
    # Add main glare around center
    y, x = np.ogrid[:row, :col]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    glare = 255 * np.exp(-dist**2 / (2 * (max_dist/4)**2)) * intensity
    noise += glare
    
    # Add secondary flares along a line
    for i in range(1, num_flares + 1):
        # Position flares along line through center
        flare_pos = i / (num_flares + 1)
        fx = int(center_x * (1 - flare_pos))
        fy = int(center_y * (1 - flare_pos))
        
        # Random flare size
        size = np.random.randint(10, 30)
        
        # Create flare
        dist = np.sqrt((x - fx)**2 + (y - fy)**2)
        flare = 255 * np.exp(-dist**2 / (2 * size**2)) * intensity * (1 - i/(num_flares+1))
        noise += flare
    
    # Apply lens flare to image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 5.6 Water Droplets
def add_water_droplets(image, num_droplets=20, min_size=5, max_size=20):
    """
    Formula: Refraction-like distortion in circular regions
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    result = image.copy()
    
    for _ in range(num_droplets):
        # Random droplet position and size
        cx = np.random.randint(0, col)
        cy = np.random.randint(0, row)
        radius = np.random.randint(min_size, max_size)
        
        # Create a coordinate grid centered on the droplet
        y, x = np.ogrid[:row, :col]
        
        # Create a circular mask for the droplet
        mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
        
        # Skip if droplet is too close to the edge (no pixels inside the mask)
        if not np.any(mask):
            continue
        
        # Create spherical distortion map
        dx = x - cx
        dy = y - cy
        
        # Calculate distance from center (avoid division by zero)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist[dist == 0] = 1  # Avoid division by zero
        
        # Calculate distortion factor (simulating lens effect)
        factor = 0.5 * radius / dist
        factor[~mask] = 0  # Only apply within the droplet
        
        # Calculate new coordinates
        new_x = x - dx * factor
        new_y = y - dy * factor
        
        # Ensure coordinates are within bounds
        new_x = np.clip(new_x, 0, col-1).astype(int)
        new_y = np.clip(new_y, 0, row-1).astype(int)
        
        # Apply distortion only within droplet
        for i in range(row):
            for j in range(col):
                if mask[i, j]:
                    result[i, j] = image[new_y[i, j], new_x[i, j]]
        
        # Calculate noise for visualization
        noise[mask] = result[mask] - image[mask]
        
        # Add highlight to droplet edge
        edge_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2) & ((x - cx)**2 + (y - cy)**2 >= (radius*0.9)**2)
        result[edge_mask] = np.minimum(result[edge_mask] + 50, 255)
        noise[edge_mask] += 50
    
    return result.astype(np.uint8), noise

# 5.7 Smoke Effect
def add_smoke(image, intensity=0.5, scale=20):
    """
    Formula: Semi-transparent overlay with fractal noise
    """
    row, col = image.shape
    
    # Generate fractal noise for smoke pattern
    noise = np.zeros((row, col))
    octaves = 5
    persistence = 0.5
    lacunarity = 2.0
    
    for i in range(octaves):
        frequency = lacunarity ** i
        amplitude = persistence ** i
        
        # Generate basic noise at current frequency
        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=scale/frequency)
        noise += amplitude * noise_layer
    
    # Normalize and adjust contrast
    smoke = (noise - noise.min()) / (noise.max() - noise.min())
    smoke = np.power(smoke, 0.3)  # Increase contrast
    
    # Apply smoke with variable opacity
    opacity = intensity * smoke
    smoky = image * (1 - opacity) + 200 * opacity
    
    # Calculate noise effect
    noise_effect = smoky - image
    
    return np.clip(smoky, 0, 255).astype(np.uint8), noise_effect

# 5.8 Scratched Film
def add_scratched_film(image, num_scratches=20, width_range=(1, 3), intensity=150):
    """
    Formula: Vertical bright lines simulating film scratches
    """
    row, col = image.shape
    noise = np.zeros_like(image, dtype=float)
    
    for _ in range(num_scratches):
        # Random scratch position and width
        x = np.random.randint(0, col)
        width = np.random.randint(width_range[0], width_range[1] + 1)
        
        # Random scratch brightness
        brightness = np.random.randint(intensity//2, intensity)
        
        # Random scratch length (partial or full length)
        if np.random.random() < 0.3:  # 30% chance of partial scratch
            start_y = np.random.randint(0, row//2)
            end_y = np.random.randint(row//2, row)
        else:
            start_y = 0
            end_y = row
        
        # Draw scratch
        for y in range(start_y, end_y):
            for w in range(width):
                if 0 <= x + w < col:
                    noise[y, x + w] = brightness
    
    # Apply scratches to image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 5.9 Glitch Effect
def add_glitch(image, intensity=0.5, num_glitches=10):
    """
    Formula: Random horizontal shifts of image segments
    """
    row, col = image.shape
    result = image.copy()
    noise = np.zeros_like(image, dtype=float)
    
    # Define glitch regions
    glitch_lines = np.random.randint(0, row, size=num_glitches)
    glitch_lines.sort()  # Sort to get proper segments
    
    # Add start and end points
    glitch_lines = np.append(np.insert(glitch_lines, 0, 0), row)
    
    # Apply glitches to segments
    for i in range(len(glitch_lines) - 1):
        start_y = glitch_lines[i]
        end_y = glitch_lines[i+1]
        
        if end_y > start_y:
            # Random shift for this segment
            shift = np.random.randint(-int(col * intensity), int(col * intensity))
            
            if shift != 0:
                segment = result[start_y:end_y, :]
                shifted = np.zeros_like(segment)
                
                if shift > 0:
                    shifted[:, shift:] = segment[:, :-shift]
                else:
                    shifted[:, :shift] = segment[:, -shift:]
                
                # Apply shifted segment
                result[start_y:end_y, :] = shifted
                
                # Calculate noise for this segment
                noise[start_y:end_y, :] = shifted - image[start_y:end_y, :]
    
    return result.astype(np.uint8), noise

# 5.10 Pixelation
def add_pixelation(image, block_size=8):
    """
    Formula: Reduction of resolution by averaging blocks of pixels
    """
    row, col = image.shape
    
    # Ensure block_size is at least 2
    block_size = max(2, block_size)
    
    # Calculate new dimensions
    new_row = row // block_size
    new_col = col // block_size
    
    # Resize down and then up to create pixelation
    small = cv2.resize(image, (new_col, new_row), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (col, row), interpolation=cv2.INTER_NEAREST)
    
    # Calculate noise (difference due to pixelation)
    noise = pixelated.astype(float) - image
    
    return pixelated, noise


# -----------------------------------------------
# 6. ДОПОЛНИТЕЛЬНЫЕ СТАТИСТИЧЕСКИЕ РАСПРЕДЕЛЕНИЯ
# -----------------------------------------------

# 6.1 Rice / Rician Noise (используется в МРТ)
def add_rician_noise(image, s=0, sigma=15):
    """
    Formula: PDF: p(z) = (z/σ²)*exp(-(z²+s²)/(2σ²))*I₀(zs/σ²) for z ≥ 0
    I₀ is the modified Bessel function of the first kind with order zero
    """
    row, col = image.shape
    noise = rice.rvs(s, scale=sigma, size=(row, col))
    noise = noise - np.mean(noise)  # Center around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 6.2 Von Mises Noise (круговое распределение)
def add_vonmises_noise(image, kappa=3, loc=0, scale=15):
    """
    Formula: PDF: p(z) = [e^(κ*cos(z-μ))] / [2π*I₀(κ)]
    """
    row, col = image.shape
    noise = vonmises.rvs(kappa, loc=loc, size=(row, col)) * scale
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 6.3 Gumbel / Extreme Value Noise
def add_gumbel_noise(image, loc=0, scale=15):
    """
    Formula: PDF: p(z) = (1/β)*exp(-(z-μ)/β - exp(-(z-μ)/β))
    """
    row, col = image.shape
    noise = gumbel_r.rvs(loc=loc, scale=scale, size=(row, col))
    noise = noise - np.mean(noise)  # Center around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 6.4 Lévy Noise (тяжёлые хвосты)
def add_levy_noise(image, loc=0, scale=1):
    """
    Formula: PDF: p(z) = sqrt(c/(2π))*exp(-c/(2(z-μ)))/((z-μ)^(3/2)) for z > μ
    """
    row, col = image.shape
    # Lévy distribution has very heavy tails, so we need to scale it down
    noise = levy.rvs(loc=loc, scale=scale, size=(row, col))
    # Clip extreme values
    noise = np.clip(noise, -30, 30)
    # Scale to reasonable intensity
    noise = noise / 3
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 6.5 Nakagami Noise
def add_nakagami_noise(image, nu=1, loc=0, scale=15):
    """
    Formula: PDF: p(z) = (2*m^m)/(Γ(m)*Ω^m)*z^(2m-1)*exp(-m*z²/Ω) for z ≥ 0
    """
    row, col = image.shape
    noise = nakagami.rvs(nu, loc=loc, scale=scale, size=(row, col))
    noise = noise - np.mean(noise)  # Center around zero
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 7. ФИЗИЧЕСКИЕ И КВАНТОВЫЕ ШУМЫ
# -----------------------------------------------

# 7.1 Johnson-Nyquist Noise (тепловой шум в электронике)
def add_johnson_nyquist_noise(image, temperature=300, bandwidth=10000, resistance=1000):
    """
    Formula: P = 4kTR∆f
    P - noise power, k - Boltzmann constant, T - temperature,
    R - resistance, ∆f - bandwidth
    """
    row, col = image.shape
    # Boltzmann constant
    k = 1.38e-23
    
    # Calculate power of thermal noise
    noise_power = 4 * k * temperature * resistance * bandwidth
    
    # Standard deviation is sqrt of power
    sigma = np.sqrt(noise_power) * 1e10  # Scale to make it visible
    
    # Generate Gaussian noise with calculated power
    noise = np.random.normal(0, sigma, (row, col))
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 7.2 Flicker Noise (1/f шум в электронике)
def add_flicker_noise(image, intensity=25, alpha=1.0):
    """
    Formula: S(f) ∝ 1/f^α where α ≈ 1
    Similar to pink noise but specifically for electronic circuits
    """
    # This is very similar to pink noise but specifically for electronic contexts
    return add_pink_noise(image, intensity)

# 7.3 Shot Noise (квантовый шум)
def add_quantum_shot_noise(image, photons_per_pixel=100):
    """
    Formula: Shot noise follows Poisson(λ) where λ is photon count
    """
    row, col = image.shape
    
    # Convert image to "photon counts"
    photon_scale = photons_per_pixel / 255
    photon_counts = np.maximum(image * photon_scale, 1)  # At least 1 photon
    
    # Generate Poisson noise based on photon counts
    noisy_photons = np.random.poisson(photon_counts)
    
    # Convert back to intensity
    noisy_img = noisy_photons / photon_scale
    
    # Calculate the noise component
    noise = noisy_img - image
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 7.4 Avalanche Noise
def add_avalanche_noise(image, prob_spike=0.01, amplitude=50):
    """
    Formula: Random spikes due to avalanche breakdown in semiconductors
    """
    row, col = image.shape
    
    # Base thermal noise
    base_noise = np.random.normal(0, 10, (row, col))
    
    # Add random avalanche spikes
    spike_mask = np.random.random((row, col)) < prob_spike
    spikes = np.random.normal(0, amplitude, (row, col)) * spike_mask
    
    noise = base_noise + spikes
    noisy_img = image + noise
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 7.5 Generation-Recombination Noise
def add_gr_noise(image, rate=0.1, amplitude=20):
    """
    Formula: Random two-state noise caused by generation-recombination processes
    """
    row, col = image.shape
    
    # Create a random two-state process (0 or 1)
    state = np.zeros((row, col))
    for i in range(row):
        current_state = 0
        for j in range(col):
            # Chance to flip state
            if np.random.random() < rate:
                current_state = 1 - current_state
            state[i, j] = current_state
    
    # Convert to noise with amplitude
    noise = (state * 2 - 1) * amplitude
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 8. СПЕЦИАЛИЗИРОВАННЫЕ ШУМЫ КОНКРЕТНЫХ ОБЛАСТЕЙ
# -----------------------------------------------

# 8.1 K-Distribution Noise (радар)
def add_k_distribution_noise(image, shape=2.0, scale=10.0):
    """
    Formula: Product of Gamma and Square root of Gamma random variables
    """
    row, col = image.shape
    
    # K-distribution can be simulated as a compound distribution
    gamma1 = np.random.gamma(shape, scale, (row, col))
    gamma2 = np.random.gamma(shape, 1.0, (row, col))
    
    # K-distributed random variable
    k_dist = np.sqrt(gamma1 * gamma2)
    
    # Center around zero for additive noise
    noise = k_dist - np.mean(k_dist)
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 8.2 Photon-Limited Noise (астрономия, низкоуровневая визуализация)
def add_photon_limited_noise(image, photon_factor=0.5):
    """
    Formula: Extremely low light imaging where photon counting is limiting factor
    """
    row, col = image.shape
    
    # Scale image to very low photon counts
    max_photons = 255 * photon_factor
    
    # Convert to "photon counts" (0 to max_photons)
    photon_counts = np.maximum(image * (max_photons/255), 0.001)
    
    # Generate Poisson noise based on photon counts
    noisy_photons = np.random.poisson(photon_counts)
    
    # Convert back to intensity
    noisy_img = noisy_photons * (255/max_photons)
    
    # Calculate the noise component
    noise = noisy_img - image
    
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 8.3 Phase Noise (осцилляторы)
def add_phase_noise(image, intensity=5, frequency=10):
    """
    Formula: Random phase modulation of periodic signal
    """
    row, col = image.shape
    
    # Create coordinate grid
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    
    # Create phase noise by adding a filtered random component to the phase
    phase_error = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=5) * intensity
    
    # Apply phase noise to a sinusoidal pattern
    noise = np.sin(2 * np.pi * frequency * (xx + yy + phase_error/100))
    
    # Scale to appropriate intensity
    noise = noise * intensity
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 8.4 Cosmic Noise (высокоэнергетические частицы)
def add_cosmic_noise(image, rate=0.001, energy_range=(50, 200)):
    """
    Formula: Random bright streaks from cosmic ray hits
    """
    row, col = image.shape
    noise = np.zeros((row, col))
    
    # Number of cosmic rays
    num_rays = int(row * col * rate)
    
    for _ in range(num_rays):
        # Random starting point
        x = np.random.randint(0, col)
        y = np.random.randint(0, row)
        
        # Random length and angle
        length = np.random.randint(3, 20)
        angle = np.random.random() * 2 * np.pi
        
        # Random energy
        energy = np.random.uniform(energy_range[0], energy_range[1])
        
        # Create the ray
        for i in range(length):
            x_new = int(x + i * np.cos(angle))
            y_new = int(y + i * np.sin(angle))
            
            if 0 <= x_new < col and 0 <= y_new < row:
                # Energy decreases along the track
                pixel_energy = energy * (1 - i/length)
                noise[y_new, x_new] = pixel_energy
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 8.5 Jitter Noise (дрожание сигнала/временной сдвиг)
def add_jitter_noise(image, jitter_max=2):
    """
    Formula: Random horizontal shift of each scanline
    """
    row, col = image.shape
    result = np.zeros_like(image)
    
    # Apply random shift to each row
    for i in range(row):
        jitter = np.random.randint(-jitter_max, jitter_max+1)
        if jitter >= 0:
            result[i, jitter:] = image[i, :col-jitter]
        else:
            result[i, :col+jitter] = image[i, -jitter:]
    
    # Calculate noise as difference
    noise = result.astype(float) - image
    
    return result, noise

# -----------------------------------------------
# 9. МАТЕМАТИЧЕСКИЕ И ТЕОРЕТИЧЕСКИЕ ШУМЫ
# -----------------------------------------------

# 9.1 Lévy Stable Distribution Noise (обобщение многих распределений)
def add_levy_stable_noise(image, alpha=1.5, beta=0, scale=10):
    """
    Formula: Generalization of many distributions including Gaussian, Cauchy, Lévy
    """
    row, col = image.shape
    
    # Parameters for the Lévy stable distribution
    # alpha ∈ (0, 2], beta ∈ [-1, 1]
    alpha = min(max(alpha, 0.01), 2.0)
    beta = min(max(beta, -1.0), 1.0)
    
    # Generate uniformly distributed random variables
    u = np.random.uniform(-np.pi/2, np.pi/2, (row, col))
    w = np.random.exponential(1.0, (row, col))
    
    # Generate Lévy stable distribution based on formulas
    if alpha == 1.0:
        # Special case
        noise = (2/np.pi) * ((np.pi/2 + beta * u) * np.tan(u) - 
                              beta * np.log(w * np.cos(u) / (np.pi/2 + beta * u)))
    else:
        # General case
        zeta = -beta * np.tan(np.pi * alpha / 2)
        term1 = np.sin(alpha * (u + zeta)) / np.power(np.cos(u), 1/alpha)
        term2 = np.power(np.cos(u - alpha * (u + zeta)) / w, (1-alpha)/alpha)
        noise = term1 * term2
    
    # Scale and center around zero
    noise = (noise - np.median(noise)) * scale
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 9.2 Chaotic Noise (детерминированный хаос)
def add_chaotic_noise(image, r=3.9, iterations=100, scale=25):
    """
    Formula: Based on chaotic maps like logistic map: x_{n+1} = r*x_n*(1-x_n)
    """
    row, col = image.shape
    
    # Initialize with random values between 0 and 1
    noise = np.random.random((row, col))
    
    # Apply logistic map iterations
    for _ in range(iterations):
        noise = r * noise * (1 - noise)
    
    # Scale to appropriate range and center around zero
    noise = (noise - 0.5) * scale
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 9.3 Brownian Bridge Noise
def add_brownian_bridge_noise(image, num_steps=20, intensity=25):
    """
    Formula: Brownian motion constrained to return to starting point
    B_t = W_t - t*W_1 where W_t is a Wiener process (Brownian motion)
    """
    row, col = image.shape
    
    # Generate a 2D grid of Brownian bridges
    t = np.linspace(0, 1, col)
    noise = np.zeros((row, col))
    
    for i in range(row):
        # Generate standard Brownian motion
        dW = np.random.normal(0, 1/np.sqrt(col), col)
        W = np.cumsum(dW)
        
        # Convert to Brownian bridge (equals 0 at t=0 and t=1)
        B = W - t * W[-1]
        
        # Scale to desired intensity
        noise[i, :] = B * intensity
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 9.4 Multifractal Noise
def add_multifractal_noise(image, intensity=30, octaves=6, lacunarity=2.0):
    """
    Formula: Extension of fractals with non-constant fractal dimension
    """
    row, col = image.shape
    
    # Base noise
    noise = np.zeros((row, col))
    
    # Create different fractal dimensions for different regions
    H_map = gaussian_filter(np.random.random((row, col)), sigma=30)
    H_map = 0.2 + 0.6 * H_map  # Range from 0.2 to 0.8
    
    # Initialize values
    frequency = 1.0
    amplitude = 1.0
    
    for i in range(octaves):
        # Generate noise layer
        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)
        
        # Apply varying H parameter from H_map
        octave_contribution = np.zeros((row, col))
        for y in range(row):
            for x in range(col):
                # Calculate amplitude based on local H value
                local_amplitude = amplitude * (frequency ** (-H_map[y, x]))
                octave_contribution[y, x] = local_amplitude * noise_layer[y, x]
        
        # Add contribution to noise
        noise += octave_contribution
        
        # Update parameters for next octave
        frequency *= lacunarity
        amplitude *= 0.5
    
    # Normalize and scale
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = intensity * noise - intensity/2
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 10. ШУМЫ НОВЫХ ТЕХНОЛОГИЙ
# -----------------------------------------------

# 10.1 Compression Artifacts (современные кодеки)
def add_advanced_compression_artifacts(image, quality=30, method='dct'):
    """
    Formula: Artifacts from modern compression methods
    """
    row, col = image.shape
    
    if method == 'dct':
        # DCT-based compression similar to JPEG but with customization
        # Divide image into 8x8 blocks
        block_size = 8
        result = np.copy(image)
        
        for i in range(0, row, block_size):
            for j in range(0, col, block_size):
                # Extract block
                i_end = min(i + block_size, row)
                j_end = min(j + block_size, col)
                block = image[i:i_end, j:j_end]
                
                # Pad if necessary
                if block.shape[0] < block_size or block.shape[1] < block_size:
                    padded = np.zeros((block_size, block_size))
                    padded[:block.shape[0], :block.shape[1]] = block
                    block = padded
                
                # Apply DCT
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Quantization step (simulating compression)
                quantization_matrix = np.ones((block_size, block_size))
                for k in range(block_size):
                    for l in range(block_size):
                        quantization_matrix[k, l] = 1 + (k + l) * (100 - quality) / 10
                
                quantized = np.round(dct_block / quantization_matrix) * quantization_matrix
                
                # Inverse DCT
                reconstructed = cv2.idct(quantized)
                
                # Place back into result
                result[i:i_end, j:j_end] = reconstructed[:i_end-i, :j_end-j]
    
    elif method == 'wavelet':
        # Wavelet-based compression (similar to JPEG2000)
        coeffs = pywt.wavedec2(image, 'haar', level=3)
        
        # Threshold coefficients based on quality
        threshold = np.percentile(np.abs(coeffs[0]), 100 - quality)
        
        # Apply thresholding to detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation coefficients
        for detail_coeffs in coeffs[1:]:
            new_detail = []
            for component in detail_coeffs:
                # Soft thresholding
                thresholded = np.sign(component) * np.maximum(np.abs(component) - threshold, 0)
                new_detail.append(thresholded)
            new_coeffs.append(tuple(new_detail))
        
        # Reconstruct image
        result = pywt.waverec2(new_coeffs, 'haar')
        
        # Ensure original dimensions
        result = result[:row, :col]
    
    else:  # Default to simple method
        # Save as JPEG to introduce compression artifacts
        result = np.clip(image, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', result, encode_param)
        result = cv2.imdecode(buffer, 0)
    
    # Calculate noise
    noise = result.astype(float) - image
    
    return result, noise

# 10.2 HDR Noise (артефакты слияния экспозиций)
def add_hdr_artifacts(image, intensity=0.5):
    """
    Formula: Artifacts from merging different exposures
    """
    row, col = image.shape
    
    # Simulate multiple exposures
    exp_low = np.clip(image * 0.5, 0, 255).astype(np.uint8)
    exp_mid = np.clip(image, 0, 255).astype(np.uint8)
    exp_high = np.clip(image * 2, 0, 255).astype(np.uint8)
    
    # Add noise to each exposure
    exp_low = np.clip(exp_low + np.random.normal(0, 20, (row, col)), 0, 255).astype(np.uint8)
    exp_mid = np.clip(exp_mid + np.random.normal(0, 10, (row, col)), 0, 255).astype(np.uint8)
    exp_high = np.clip(exp_high + np.random.normal(0, 30, (row, col)), 0, 255).astype(np.uint8)
    
    # Create weight maps
    weight_low = np.clip((exp_low.astype(float) - 0) * (255 - exp_low.astype(float)) / 255, 0, 1)
    weight_mid = np.clip((exp_mid.astype(float) - 0) * (255 - exp_mid.astype(float)) / 255, 0, 1)
    weight_high = np.clip((exp_high.astype(float) - 0) * (255 - exp_high.astype(float)) / 255, 0, 1)
    
    # Normalize weights
    sum_weights = weight_low + weight_mid + weight_high
    sum_weights[sum_weights == 0] = 1  # Avoid division by zero
    
    weight_low /= sum_weights
    weight_mid /= sum_weights
    weight_high /= sum_weights
    
    # Merge exposures
    hdr = (exp_low.astype(float) * weight_low + 
           exp_mid.astype(float) * weight_mid + 
           exp_high.astype(float) * weight_high)
    
    # Tone mapping (simple gamma correction)
    gamma = 1.0 - intensity * 0.5
    mapped = np.power(hdr / 255, gamma) * 255
    
    # Create ghost artifacts in areas of movement
    ghost_mask = np.random.random((row, col)) < 0.02 * intensity
    ghost_regions = gaussian_filter(ghost_mask.astype(float), sigma=2) * 30 * intensity
    
    # Apply ghost artifacts
    hdr_with_ghosts = mapped + ghost_regions
    
    # Calculate noise (difference from original)
    noise = hdr_with_ghosts - image
    
    return np.clip(hdr_with_ghosts, 0, 255).astype(np.uint8), noise

# 10.3 Deep Learning Artifacts
def add_dl_artifacts(image, block_size=8, intensity=0.7):
    """
    Formula: Artifacts typical in deep learning image processing
    """
    row, col = image.shape
    
    # Create a grid pattern (common in some neural networks)
    grid_pattern = np.zeros((row, col))
    for i in range(0, row, block_size):
        for j in range(0, col, block_size):
            grid_pattern[i:i+1, j:j+block_size] = 1
            grid_pattern[i:i+block_size, j:j+1] = 1
    
    # Over-smoothing in some areas (common in denoisers/GANs)
    smooth_mask = np.random.random((row, col)) < 0.3
    smooth_regions = np.zeros((row, col))
    
    for i in range(0, row, block_size):
        for j in range(0, col, block_size):
            i_end = min(i + block_size, row)
            j_end = min(j + block_size, col)
            
            if np.random.random() < 0.4:  # 40% of blocks are over-smoothed
                # Calculate block average
                block = image[i:i_end, j:j_end]
                avg_value = np.mean(block)
                
                # Create smooth transition to average
                smooth_regions[i:i_end, j:j_end] = avg_value - block
    
    # "Checkerboard" artifacts (common in deconvolution networks)
    checkerboard = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if (i // (block_size//2) + j // (block_size//2)) % 2 == 0:
                checkerboard[i, j] = 5
            else:
                checkerboard[i, j] = -5
    
    # Combine artifacts
    noise = (grid_pattern * 10 + smooth_regions + checkerboard * 0.5) * intensity
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 11. КОМБИНАЦИОННЫЕ И СПЕЦИФИЧЕСКИЕ ШУМЫ
# -----------------------------------------------

# 11.1 Non-Stationary Noise (изменяющийся во времени/пространстве)
def add_nonstationary_noise(image, intensity_range=(5, 30)):
    """
    Formula: Noise with spatially varying statistics
    """
    row, col = image.shape
    
    # Create gradually varying intensity map
    t = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(t, y)
    
    # Create varying intensity pattern
    intensity_map = intensity_range[0] + (intensity_range[1] - intensity_range[0]) * (
        0.5 * np.sin(xx * 6 * np.pi) * np.cos(yy * 4 * np.pi) + 0.5)
    
    # Create varying noise type (mix of Gaussian and salt-and-pepper)
    gaussian_noise = np.random.normal(0, 1, (row, col))
    salt_mask = np.random.random((row, col)) < 0.01
    pepper_mask = np.random.random((row, col)) < 0.01
    
    # Combine noise types with spatially varying intensity
    noise = gaussian_noise * intensity_map
    noise[salt_mask] = 255
    noise[pepper_mask] = -255
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 11.2 Heteroskedastic Noise (с переменной дисперсией)
def add_heteroskedastic_noise(image, min_sigma=5, max_sigma=30):
    """
    Formula: Gaussian noise with variance dependent on pixel intensity
    """
    row, col = image.shape
    
    # Make noise variance proportional to pixel intensity
    sigma = min_sigma + (max_sigma - min_sigma) * (image / 255.0)
    
    # Generate noise with pixel-dependent variance
    noise = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            noise[i, j] = np.random.normal(0, sigma[i, j])
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 11.3 Anisotropic Noise (направленный шум)
def add_anisotropic_noise(image, intensity=20, angle=45, ratio=0.2):
    """
    Formula: Noise with directional preference
    """
    row, col = image.shape
    
    # Create base isotropic noise
    base_noise = np.random.normal(0, 1, (row*2, col*2))
    
    # Apply directional blur
    angle_rad = np.deg2rad(angle)
    dx = int(np.cos(angle_rad) * 10)
    dy = int(np.sin(angle_rad) * 10)
    
    # Create directional kernel
    kernel_size = max(3, int(np.sqrt(dx**2 + dy**2)))
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Draw line in kernel
    for i in range(-center, center+1):
        x = center + int(i * dx / 10)
        y = center + int(i * dy / 10)
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply directional filtering
    filtered_noise = cv2.filter2D(base_noise, -1, kernel)
    
    # Crop to original size
    noise = filtered_noise[row//2:row//2+row, col//2:col//2+col]
    
    # Rescale to desired intensity
    noise = noise * intensity
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 11.4 Structured Correlated Noise
def add_structured_correlated_noise(image, scale=20, correlation_length=10):
    """
    Formula: Noise with specific spatial correlation structure
    """
    row, col = image.shape
    
    # Generate base white noise
    white_noise = np.random.normal(0, 1, (row, col))
    
    # Apply specific correlation structure
    correlated_noise = gaussian_filter(white_noise, sigma=correlation_length)
    
    # Add structure by multiplying with pattern
    x = np.linspace(0, 10, col)
    y = np.linspace(0, 10, row)
    xx, yy = np.meshgrid(x, y)
    
    pattern = np.sin(xx) * np.cos(yy) * 0.5 + 0.5
    
    # Apply pattern to correlated noise
    structured_noise = correlated_noise * pattern * scale
    
    noisy_img = image + structured_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), structured_noise

# -----------------------------------------------
# 12. ШУМЫ ОКРУЖАЮЩЕЙ СРЕДЫ И СПЕЦИАЛЬНЫЕ ЭФФЕКТЫ
# -----------------------------------------------

# 12.1 Atmospheric Turbulence
def add_atmospheric_turbulence(image, strength=2.0, scale=10):
    """
    Formula: Wavefront distortion due to atmospheric effects
    """
    row, col = image.shape
    result = np.zeros_like(image)
    
    # Generate displacement maps based on turbulence model
    displacement_x = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=scale) * strength
    displacement_y = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=scale) * strength
    
    # Create mesh grid
    y_coords, x_coords = np.meshgrid(np.arange(col), np.arange(row))
    
    # Apply displacements
    x_new = x_coords + displacement_x
    y_new = y_coords + displacement_y
    
    # Clip to ensure valid coordinates
    x_new = np.clip(x_new, 0, col-1).astype(np.int32)
    y_new = np.clip(y_new, 0, row-1).astype(np.int32)
    
    # Remap image
    for i in range(row):
        for j in range(col):
            result[i, j] = image[x_new[i, j], y_new[i, j]]
    
    # Calculate noise (difference from original)
    noise = result.astype(float) - image
    
    return result, noise

# 12.2 Electrical Interference Patterns
def add_electrical_interference(image, frequency_range=(10, 50), amplitude=15):
    """
    Formula: Various electrical interference patterns
    """
    row, col = image.shape
    
    # Create coordinate grid
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    
    # Choose random frequencies
    freq_x = np.random.uniform(frequency_range[0], frequency_range[1])
    freq_y = np.random.uniform(frequency_range[0], frequency_range[1])
    
    # Generate primary interference pattern
    interference = amplitude * np.sin(2 * np.pi * freq_x * xx) * np.sin(2 * np.pi * freq_y * yy)
    
    # Add some TV-like scan lines
    scanlines = np.zeros((row, col))
    scanline_pattern = np.sin(2 * np.pi * 100 * yy)
    scanline_mask = np.random.random((row, col)) < 0.1
    scanlines[scanline_mask] = scanline_pattern[scanline_mask] * amplitude
    
    # Add occasional stronger interference bursts
    burst_mask = np.random.random((row, col)) < 0.02
    bursts = burst_mask * np.random.normal(0, amplitude * 2, (row, col))
    
    # Combine all interference effects
    noise = interference + scanlines + bursts
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 12.3 Scanner Artifacts
def add_scanner_artifacts(image, line_intensity=30, dust_density=0.001):
    """
    Formula: Artifacts typical in document scanners
    """
    row, col = image.shape
    noise = np.zeros((row, col))
    
    # Add horizontal scan lines
    for i in range(0, row, np.random.randint(20, 50)):
        if np.random.random() < 0.3:  # 30% of potential scan lines are visible
            thickness = np.random.randint(1, 3)
            intensity = np.random.uniform(0.5, 1.5) * line_intensity
            if i + thickness < row:
                noise[i:i+thickness, :] = intensity
    
    # Add dust spots
    dust_mask = np.random.random((row, col)) < dust_density
    dust_sizes = np.random.randint(1, 4, size=np.sum(dust_mask))
    
    dust_indices = np.where(dust_mask)
    for i, (y, x) in enumerate(zip(*dust_indices)):
        size = dust_sizes[i]
        y1, y2 = max(0, y-size), min(row, y+size+1)
        x1, x2 = max(0, x-size), min(col, x+size+1)
        
        # Draw dust speck with random intensity
        intensity = np.random.randint(-150, -50)
        for dy in range(y1, y2):
            for dx in range(x1, x2):
                if (dy-y)**2 + (dx-x)**2 <= size**2:
                    noise[dy, dx] = intensity
    
    # Add slight page curl/shadow effect
    y, x = np.ogrid[:row, :col]
    distance_from_edge = np.minimum(x, col-x) / col
    shadow = -20 * np.exp(-distance_from_edge * 10) * (np.random.random() < 0.5)
    noise += shadow
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 12.4 Print Artifacts
def add_print_artifacts(image, dot_size=2, pattern_scale=5):
    """
    Formula: Dot patterns and artifacts from printing process
    """
    row, col = image.shape
    
    # Create halftone-like pattern
    result = np.zeros_like(image)
    
    for i in range(0, row, pattern_scale):
        for j in range(0, col, pattern_scale):
            i_end = min(i + pattern_scale, row)
            j_end = min(j + pattern_scale, col)
            
            # Get average intensity in this region
            region = image[i:i_end, j:j_end]
            avg_intensity = np.mean(region)
            
            # Calculate dot size based on intensity
            relative_dot_size = 1.0 - avg_intensity / 255
            current_dot_size = int(dot_size * relative_dot_size * 2) + 1
            
            # Draw dot
            center_i = i + pattern_scale // 2
            center_j = j + pattern_scale // 2
            
            if center_i < row and center_j < col:
                dot_i_start = max(0, center_i - current_dot_size // 2)
                dot_i_end = min(row, center_i + current_dot_size // 2 + 1)
                dot_j_start = max(0, center_j - current_dot_size // 2)
                dot_j_end = min(col, center_j + current_dot_size // 2 + 1)
                
                result[dot_i_start:dot_i_end, dot_j_start:dot_j_end] = 0
    
    # Add some random ink splatters
    splatter_mask = np.random.random((row, col)) < 0.001
    result[splatter_mask] = 0
    
    # Add paper texture
    paper_texture = np.random.normal(240, 10, (row, col))
    texture_weight = 0.2
    blended = (1 - texture_weight) * result + texture_weight * paper_texture
    
    # Calculate noise
    noise = blended - image
    
    return np.clip(blended, 0, 255).astype(np.uint8), noise

# 12.5 Polarization Noise
def add_polarization_noise(image, strength=20):
    """
    Formula: Artifacts from polarization effects in imaging
    """
    row, col = image.shape
    
    # Create coordinate grid
    x = np.linspace(0, 1, col)
    y = np.linspace(0, 1, row)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate angle from center
    center_x, center_y = 0.5, 0.5
    dx, dy = xx - center_x, yy - center_y
    angle = np.arctan2(dy, dx)
    
    # Create polarization pattern
    pol_pattern = np.cos(2 * angle) ** 2  # Malus' law pattern
    
    # Apply intensity variation based on polarization
    noise = (pol_pattern - 0.5) * strength
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 13. ШУМЫ В СПЕЦИАЛИЗИРОВАННЫХ ОБЛАСТЯХ НАУКИ
# -----------------------------------------------

# 13.1 Спектроскопический шум
def add_spectroscopic_noise(image, shot_scale=1.0, baseline_drift=True, cosmic_ray_prob=0.01):
    """
    Formula: Combination of shot noise, baseline drift, and cosmic ray spikes
    """
    row, col = image.shape
    
    # Shot noise component (Poisson)
    img_data = np.maximum(image, 0) / shot_scale
    shot_noise = np.random.poisson(img_data) * shot_scale - img_data * shot_scale
    
    # Baseline drift (slow variation across columns)
    if baseline_drift:
        t = np.linspace(0, 6*np.pi, col)
        drift = 10 * np.sin(t/3) + 5 * np.sin(t/10)
        baseline = np.tile(drift, (row, 1))
    else:
        baseline = np.zeros((row, col))
    
    # Cosmic ray spikes (very sharp, intense peaks)
    cosmic = np.zeros((row, col))
    if cosmic_ray_prob > 0:
        for i in range(int(row * col * cosmic_ray_prob)):
            x = np.random.randint(0, col)
            y = np.random.randint(0, row)
            intensity = np.random.uniform(50, 200)
            width = np.random.randint(1, 3)
            
            for dx in range(-width, width+1):
                for dy in range(-width, width+1):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < col and 0 <= ny < row:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= width:
                            cosmic[ny, nx] = intensity * (1 - dist/width)
    
    # Combined noise
    noise = shot_noise + baseline + cosmic
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 13.2 Гравитационно-волновой шум
def add_gravitational_wave_noise(image, seismic_amplitude=10, newtonian_scale=5, quantum_scale=2):
    """
    Formula: Models noise sources in gravitational wave detection
    """
    row, col = image.shape
    
    # Seismic noise (low frequency)
    t = np.linspace(0, 10*np.pi, col)
    seismic_base = np.sin(t/5) + 0.5*np.sin(t/2) + 0.2*np.sin(t/1.5)
    seismic = np.tile(seismic_base, (row, 1)) * seismic_amplitude
    
    # Newtonian noise (changes in gravitational field)
    newtonian = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=20) * newtonian_scale
    
    # Quantum noise (shot noise in laser interferometer)
    quantum = np.random.normal(0, quantum_scale, (row, col))
    
    # Combined noise
    noise = seismic + newtonian + quantum
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 13.3 Астрономический шум
def add_astronomical_noise(image, sky_brightness=10, seeing=3, tracking_error=1):
    """
    Formula: Sky background, atmospheric seeing, and telescope tracking errors
    """
    row, col = image.shape
    
    # Sky background (additive brightness with Poisson characteristics)
    sky = np.random.poisson(sky_brightness, (row, col))
    
    # Atmospheric seeing (blurring that varies over image)
    blurred = gaussian_filter(image.astype(float), sigma=seeing)
    seeing_noise = blurred - image
    
    # Telescope tracking error (slight motion blur in random direction)
    angle = np.random.uniform(0, 2*np.pi)
    dx, dy = tracking_error * np.cos(angle), tracking_error * np.sin(angle)
    
    kernel_size = max(3, int(2 * tracking_error) + 1)
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # Create motion blur kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            dist = np.sqrt((i-center-dy)**2 + (j-center-dx)**2)
            if dist < tracking_error:
                kernel[i, j] = 1
    
    if kernel.sum() > 0:
        kernel /= kernel.sum()
        tracking_result = cv2.filter2D(image.astype(float), -1, kernel)
        tracking_noise = tracking_result - image
    else:
        tracking_noise = np.zeros_like(image, dtype=float)
    
    # Combined noise
    noise = sky + seeing_noise + tracking_noise
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 13.4 Нейровизуализационный шум (fMRI, EEG)
def add_neuroimaging_noise(image, physiological_scale=15, motion_scale=10, scanner_drift=5):
    """
    Formula: Physiological noise, head motion, and scanner drift
    """
    row, col = image.shape
    
    # Physiological noise (cardiac, respiratory cycles)
    t = np.linspace(0, 2*np.pi, col)
    cardiac = np.sin(t*15)  # Faster oscillation for cardiac cycle
    respiratory = np.sin(t*4)  # Slower oscillation for respiratory cycle
    
    physio_pattern = cardiac + respiratory
    physio_pattern = physio_pattern / np.max(np.abs(physio_pattern))
    
    # Apply spatial variation
    y_gradient = np.linspace(0, 1, row)[:, np.newaxis]
    physiological = physio_pattern * y_gradient * physiological_scale
    
    # Head motion (random translations/rotations between volumes)
    if np.random.random() < 0.3:  # 30% chance of motion artifact
        shift_y = np.random.randint(-motion_scale, motion_scale+1)
        shift_x = np.random.randint(-motion_scale, motion_scale+1)
        
        motion = np.zeros((row, col))
        if shift_y != 0 or shift_x != 0:
            # Apply shift
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(image.astype(float), M, (col, row))
            motion = shifted - image
    else:
        motion = np.zeros((row, col))
    
    # Scanner drift (slow signal change over time)
    drift = np.linspace(0, scanner_drift, col)
    drift_2d = np.tile(drift, (row, 1))
    
    # Combined noise
    noise = physiological + motion + drift_2d
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 13.5 Квантово-вычислительный шум
def add_quantum_computing_noise(image, decoherence=0.1, gate_error=0.05, readout_error=0.03):
    """
    Formula: Models decoherence, gate errors, and readout errors in quantum systems
    """
    row, col = image.shape
    
    # Create binary representation
    binary = (image > 128).astype(np.uint8) * 255
    
    # Decoherence noise (random bit flips with probability proportional to decoherence)
    decoherence_mask = np.random.random((row, col)) < decoherence
    decoherence_noise = np.zeros((row, col))
    decoherence_noise[decoherence_mask] = 255 - 2 * binary[decoherence_mask]
    
    # Gate error (correlated bit flips)
    gate_noise = np.zeros((row, col))
    for i in range(0, row, 8):  # Process in 8x8 blocks
        for j in range(0, col, 8):
            if np.random.random() < gate_error:
                i_end = min(i + 8, row)
                j_end = min(j + 8, col)
                gate_noise[i:i_end, j:j_end] = np.random.choice([-50, 50])
    
    # Readout error (measurement errors)
    readout_mask = np.random.random((row, col)) < readout_error
    readout_noise = np.zeros((row, col))
    readout_noise[readout_mask] = np.random.normal(0, 50, np.sum(readout_mask))
    
    # Combined noise
    noise = decoherence_noise + gate_noise + readout_noise
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# -----------------------------------------------
# 14. НОВЕЙШИЕ ТИПЫ ШУМОВ И БУДУЩИЕ НАПРАВЛЕНИЯ
# -----------------------------------------------

# 14.1 Голографический шум
def add_holographic_noise(image, interference_scale=15, speckle_scale=0.2):
    """
    Formula: Interference patterns and speckle noise in holographic displays
    """
    row, col = image.shape
    
    # Create coordinate grid
    x = np.linspace(0, col-1, col)
    y = np.linspace(0, row-1, row)
    xx, yy = np.meshgrid(x, y)
    
    # Reference beam pattern
    ref_angle = np.random.uniform(0, 2*np.pi)
    kx_ref = np.cos(ref_angle)
    ky_ref = np.sin(ref_angle)
    reference = np.cos(2*np.pi*(kx_ref*xx/col + ky_ref*yy/row)*10)
    
    # Object beam (simulated from image)
    object_beam = image / 255.0
    
    # Interference pattern
    interference = interference_scale * ((reference + object_beam)**2 - reference**2 - object_beam**2)
    
    # Add laser speckle
    speckle = np.random.normal(0, 1, (row, col))
    speckle = gaussian_filter(speckle, sigma=1)
    speckle = speckle_scale * speckle * image / 255.0 * 50
    
    # Combined noise
    noise = interference + speckle
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 14.2 Дифракционный шум
def add_diffraction_noise(image, wavelength=0.5, aperture=20):
    """
    Formula: Diffraction patterns from wave optics
    """
    row, col = image.shape
    
    # Create coordinate grid centered at image center
    x = np.linspace(-col/2, col/2, col)
    y = np.linspace(-row/2, row/2, row)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # Circular aperture diffraction pattern (Airy disk)
    k = 2*np.pi/wavelength
    airy_argument = k * aperture * rr / np.sqrt(row**2 + col**2)
    airy_argument[airy_argument == 0] = 1e-10  # Avoid division by zero
    airy_pattern = (2 * special.j1(airy_argument) / airy_argument)**2
    
    # Scale and center the pattern
    diffraction = 20 * (airy_pattern - np.min(airy_pattern)) / (np.max(airy_pattern) - np.min(airy_pattern)) - 10
    
    # Make it more visible by applying it more strongly to bright areas
    intensity_mask = image / 255.0
    noise = diffraction * intensity_mask
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 14.3 Шум смешанной реальности (AR/VR)
def add_mixed_reality_noise(image, display_artifacts=True, tracking_jitter=True, reprojection=True):
    """
    Formula: Display, tracking, and reprojection artifacts in AR/VR systems
    """
    row, col = image.shape
    noise = np.zeros((row, col))
    
    # Display artifacts (screen door effect, mura)
    if display_artifacts:
        # Screen door effect (visible pixel grid)
        grid = np.zeros((row, col))
        for i in range(0, row, 3):
            grid[i:i+1, :] = -10
        for j in range(0, col, 3):
            grid[:, j:j+1] = -10
        
        # Mura (panel inconsistency)
        mura = np.zeros((row, col))
        for i in range(0, row, 20):
            for j in range(0, col, 20):
                i_end = min(i + 20, row)
                j_end = min(j + 20, col)
                mura[i:i_end, j:j_end] = np.random.normal(0, 3)
        
        noise += grid + mura
    
    # Tracking jitter (small random movements)
    if tracking_jitter:
        jitter_x = np.random.normal(0, 1, row)
        jitter_y = np.random.normal(0, 1, row)
        
        jittered = np.zeros((row, col))
        for i in range(row):
            dx, dy = int(jitter_x[i]), int(jitter_y[i])
            
            for j in range(col):
                new_j, new_i = j + dx, i + dy
                if 0 <= new_i < row and 0 <= new_j < col:
                    jittered[i, j] = image[new_i, new_j]
                else:
                    jittered[i, j] = image[i, j]
        
        jitter_noise = jittered - image
        noise += jitter_noise
    
    # Reprojection artifacts (misalignment when head moves)
    if reprojection:
        # Simulate depth map (closer objects have higher values)
        depth_map = np.ones((row, col)) * 128
        
        # Add some random depth variation
        for _ in range(20):
            cx, cy = np.random.randint(0, col), np.random.randint(0, row)
            radius = np.random.randint(10, 50)
            
            # Create circle
            y, x = np.ogrid[:row, :col]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            # Assign depth
            depth = np.random.randint(50, 200)
            depth_map[mask] = depth
        
        # Apply reprojection based on depth
        shift_x, shift_y = np.random.randint(-3, 4, 2)
        
        reprojected = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                # Scale shift based on depth (closer objects move more)
                depth_factor = (255 - depth_map[i, j]) / 255.0
                dx = int(shift_x * depth_factor)
                dy = int(shift_y * depth_factor)
                
                new_j, new_i = j + dx, i + dy
                if 0 <= new_i < row and 0 <= new_j < col:
                    reprojected[i, j] = image[new_i, new_j]
                else:
                    reprojected[i, j] = 0  # Missing data (black)
        
        reprojection_noise = reprojected - image
        noise += reprojection_noise
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 14.4 Шум системы машинного зрения
def add_machine_vision_noise(image, lighting_variation=True, motion_blur=True, sensor_effects=True):
    """
    Formula: Noise specific to machine vision and robotic imaging systems
    """
    row, col = image.shape
    noise = np.zeros((row, col))
    
    # Lighting variation (flicker, shadows)
    if lighting_variation:
        # Create lighting map
        t = np.linspace(0, 1, col)
        flicker = 1.0 + 0.2 * np.sin(2*np.pi*t*10)  # Temporal flicker
        
        # Spatial lighting variation (shadows)
        shadow_mask = np.ones((row, col))
        for _ in range(3):
            cx, cy = np.random.randint(0, col), np.random.randint(0, row)
            radius = np.random.randint(30, 100)
            
            # Create shadow
            y, x = np.ogrid[:row, :col]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            shadow = np.exp(-dist**2 / (2*radius**2))
            shadow_mask -= shadow * 0.3
        
        shadow_mask = np.clip(shadow_mask, 0.6, 1.0)
        
        # Apply lighting variations
        lighting_noise = image * (np.outer(np.ones(row), flicker) * shadow_mask - 1.0)
        noise += lighting_noise
    
    # Motion blur from robot movement
    if motion_blur:
        kernel_size = np.random.randint(3, 9)
        angle = np.random.uniform(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        angle_rad = np.deg2rad(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        for i in range(kernel_size):
            x = center + dx * (i - center)
            y = center + dy * (i - center)
            x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
            
            if 0 <= x_floor < kernel_size and 0 <= y_floor < kernel_size:
                kernel[y_floor, x_floor] = 1
        
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            
            # Apply blur
            blurred = cv2.filter2D(image.astype(float), -1, kernel)
            motion_noise = blurred - image
            noise += motion_noise
    
    # Industrial sensor effects (banding, gain inconsistency)
    if sensor_effects:
        # Row/column banding
        for i in range(0, row, np.random.randint(10, 30)):
            if np.random.random() < 0.2:
                width = np.random.randint(1, 3)
                if i + width < row:
                    noise[i:i+width, :] += np.random.normal(0, 5)
        
        # Gain inconsistency
        gain_map = np.ones((row, col))
        for _ in range(5):
            cx, cy = np.random.randint(0, col), np.random.randint(0, row)
            radius = np.random.randint(20, 60)
            
            # Create gain variation
            y, x = np.ogrid[:row, :col]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            gain_variation = 1.0 + 0.1 * np.exp(-dist**2 / (2*radius**2))
            gain_map *= gain_variation
        
        gain_noise = image * (gain_map - 1.0)
        noise += gain_noise
    
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise

# 14.5 Высокодинамический шум (HDR-specific)
def add_hdr_specific_noise(image, photon_scale=0.1, over_exposure=True, tone_mapping=True):
    """
    Formula: Noise specifically affecting high dynamic range imaging
    """
    row, col = image.shape
    
    # Simulate HDR image by stretching dynamic range
    # (This is just a simulation for noise purposes, not actual HDR)
    hdr_sim = image.astype(float) ** 0.5 * 255
    
    # Photon noise proportional to brightness
    photon_noise = np.random.poisson(np.maximum(hdr_sim * photon_scale, 1)) / photon_scale - hdr_sim
    
    # Sensor saturation and over-exposure
    if over_exposure:
        bright_mask = hdr_sim > 240
        saturation = np.zeros((row, col))
        
        # Add blooming around saturated pixels
        if np.any(bright_mask):
            dilated_mask = cv2.dilate(bright_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
            bloom_mask = dilated_mask & (~bright_mask)
            saturation[bloom_mask] = 20
            
            # Hard clip saturated pixels to 255
            saturation[bright_mask] = 255 - hdr_sim[bright_mask]
    else:
        saturation = np.zeros((row, col))
    
    # Tone mapping artifacts
    if tone_mapping:
        # Simulate local tone mapping artifacts
        tone_map = np.zeros((row, col))
        
        # Local contrast enhancement leading to halos
        edges = cv2.Canny(image, 50, 150).astype(float)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        edges_expanded = gaussian_filter(edges_dilated, sigma=3) / 255.0
        
        # Create halo effect around edges
        tone_map = edges_expanded * 20 * np.sign(128 - image)
    else:
        tone_map = np.zeros((row, col))
    
    # Combined noise
    noise = photon_noise + saturation + tone_map
    
    # Apply to original image
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise


def main():
    # Create a directory for the output images
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Dictionary of all noise functions with titles
    noise_functions = {
        # 1. STATISTICAL DISTRIBUTION-BASED NOISE TYPES
        "1.1 Gaussian Noise": add_gaussian_noise,
        "1.2 Salt and Pepper Noise": add_salt_and_pepper_noise,
        "1.3 Poisson Noise": add_poisson_noise,
        "1.4 Speckle Noise": add_speckle_noise,
        "1.5 Uniform Noise": add_uniform_noise,
        "1.6 Rayleigh Noise": add_rayleigh_noise,
        "1.7 Gamma Noise": add_gamma_noise,
        "1.8 Exponential Noise": add_exponential_noise,
        "1.9 Laplacian Noise": add_laplacian_noise,
        "1.10 Cauchy Noise": add_cauchy_noise,
        "1.11 Chi-Square Noise": add_chi2_noise,
        "1.12 Beta Noise": add_beta_noise,
        "1.13 Weibull Noise": add_weibull_noise,
        "1.14 Logistic Noise": add_logistic_noise,
        "1.15 Student's t-Noise": add_t_noise,
        "1.16 F-Distribution Noise": add_f_noise,
        "1.17 Lognormal Noise": add_lognormal_noise,
        "1.18 Binomial Noise": add_binomial_noise,
        "1.19 Negative Binomial Noise": add_nbinom_noise,
        "1.20 Hypergeometric Noise": add_hypergeom_noise,
        "1.21 Pareto Noise": add_pareto_noise,
        "1.22 Maxwell-Boltzmann Noise": add_maxwell_noise,
        
        # 2. COLORED NOISE TYPES
        "2.1 White Noise": add_white_noise,
        "2.2 Pink Noise": add_pink_noise,
        "2.3 Brown Noise": add_brown_noise,
        "2.4 Blue Noise": add_blue_noise,
        "2.5 Violet Noise": add_violet_noise,
        "2.6 Grey Noise": add_grey_noise,
        "2.7 Red Noise": add_red_noise,
        "2.8 Orange Noise": add_orange_noise,
        "2.9 Green Noise": add_green_noise,
        "2.10 Black Noise": add_black_noise,
        
        # 3. PROCEDURAL AND SYNTHETIC NOISE TYPES
        "3.1 Perlin Noise": add_perlin_noise if has_perlin else add_simplex_noise,
        "3.2 Simplex Noise": add_simplex_noise,
        "3.3 Worley Noise": add_worley_noise,
        "3.4 Fractional Brownian Motion": add_fbm_noise,
        "3.5 Value Noise": add_value_noise,
        "3.6 Wavelet Noise": add_wavelet_noise,
        "3.7 Diamond-Square Noise": add_diamond_square_noise,
        "3.8 Gabor Noise": add_gabor_noise,
        "3.9 Sparse Convolution Noise": add_sparse_convolution_noise,
        "3.10 Turbulence Noise": add_turbulence_noise,
        "3.11 Mosaic Noise": add_mosaic_noise,
        "3.12 Ridge Noise": add_ridge_noise,
        "3.13 Curl Noise": add_curl_noise,
        "3.14 Caustic Noise": add_caustic_noise,
        "3.15 Cracks Noise": add_cracks_noise,
        "3.16 Flow Noise": add_flow_noise,
        
        # 4. PHYSICAL AND DEVICE-BASED NOISE TYPES
        "4.1 Film Grain Noise": add_film_grain,
        "4.2 CCD/CMOS Sensor Noise": add_sensor_noise,
        "4.3 Thermal Noise": add_thermal_noise,
        "4.4 Fixed Pattern Noise": add_fixed_pattern_noise,
        "4.5 Row/Column Noise": add_row_column_noise,
        "4.6 Banding Noise": add_banding_noise,
        "4.7 JPEG Compression Noise": add_jpeg_noise,
        "4.8 Quantization Noise": add_quantization_noise,
        "4.9 Demosaicing Noise": add_demosaicing_noise,
        "4.10 Lens Blur": add_lens_blur,
        "4.11 Motion Blur": add_motion_blur,
        "4.12 Vignetting": add_vignetting,
        "4.13 Blooming": add_blooming,
        "4.14 Chromatic Aberration": add_chromatic_aberration,
        "4.15 Hot Pixels": add_hot_pixels,
        "4.16 Dead Pixels": add_dead_pixels,
        "4.17 Rolling Shutter Effect": add_rolling_shutter,
        "4.18 Moiré Pattern": add_moire_pattern,
        
        # 5. REAL-WORLD AND ENVIRONMENTAL NOISE TYPES
        "5.1 Rain Noise": add_rain_noise,
        "5.2 Snow Noise": add_snow_noise,
        "5.3 Fog/Haze": add_fog,
        "5.4 Dust and Scratches": add_dust_scratches,
        "5.5 Lens Flare": add_lens_flare,
        "5.6 Water Droplets": add_water_droplets,
        "5.7 Smoke Effect": add_smoke,
        "5.8 Scratched Film": add_scratched_film,
        "5.9 Glitch Effect": add_glitch,
        "5.10 Pixelation": add_pixelation,
        
        # 6. ADDITIONAL STATISTICAL DISTRIBUTIONS
        "6.1 Rice/Rician Noise": add_rician_noise,
        "6.2 Von Mises Noise": add_vonmises_noise,
        "6.3 Gumbel/Extreme Value Noise": add_gumbel_noise,
        "6.4 Lévy Noise": add_levy_noise,
        "6.5 Nakagami Noise": add_nakagami_noise,
        
        # 7. PHYSICAL AND QUANTUM NOISE
        "7.1 Johnson-Nyquist Noise": add_johnson_nyquist_noise,
        "7.2 Flicker Noise": add_flicker_noise,
        "7.3 Quantum Shot Noise": add_quantum_shot_noise,
        "7.4 Avalanche Noise": add_avalanche_noise,
        "7.5 Generation-Recombination Noise": add_gr_noise,
        
        # 8. SPECIALIZED FIELDS NOISE
        "8.1 K-Distribution Noise": add_k_distribution_noise,
        "8.2 Photon-Limited Noise": add_photon_limited_noise,
        "8.3 Phase Noise": add_phase_noise,
        "8.4 Cosmic Noise": add_cosmic_noise,
        "8.5 Jitter Noise": add_jitter_noise,
        
        # 9. MATHEMATICAL AND THEORETICAL NOISE
        "9.1 Lévy Stable Distribution Noise": add_levy_stable_noise,
        "9.2 Chaotic Noise": add_chaotic_noise,
        "9.3 Brownian Bridge Noise": add_brownian_bridge_noise,
        "9.4 Multifractal Noise": add_multifractal_noise,
        
        # 10. NEW TECHNOLOGY NOISE
        "10.1 Advanced Compression Artifacts": add_advanced_compression_artifacts,
        "10.2 HDR Artifacts Noise": add_hdr_artifacts,
        "10.3 Deep Learning Artifacts": add_dl_artifacts,
        
        # 11. COMBINATIONAL AND SPECIFIC NOISE
        "11.1 Non-Stationary Noise": add_nonstationary_noise,
        "11.2 Heteroskedastic Noise": add_heteroskedastic_noise,
        "11.3 Anisotropic Noise": add_anisotropic_noise,
        "11.4 Structured Correlated Noise": add_structured_correlated_noise,
        
        # 12. ENVIRONMENT AND SPECIAL EFFECTS NOISE
        "12.1 Atmospheric Turbulence": add_atmospheric_turbulence,
        "12.2 Electrical Interference Patterns": add_electrical_interference,
        "12.3 Scanner Artifacts": add_scanner_artifacts,
        "12.4 Print Artifacts": add_print_artifacts,
        "12.5 Polarization Noise": add_polarization_noise,
        
        # 13. SPECIALIZED SCIENTIFIC FIELDS NOISE
        "13.1 Spectroscopic Noise": add_spectroscopic_noise,
        "13.2 Gravitational Wave Noise": add_gravitational_wave_noise,
        "13.3 Astronomical Noise": add_astronomical_noise,
        "13.4 Neuroimaging Noise": add_neuroimaging_noise,
        "13.5 Quantum Computing Noise": add_quantum_computing_noise,
        
        # 14. NEWEST NOISE TYPES AND FUTURE DIRECTIONS
        "14.1 Holographic Noise": add_holographic_noise,
        "14.2 Diffraction Noise": add_diffraction_noise,
        "14.3 Mixed Reality Noise": add_mixed_reality_noise,
        "14.4 Machine Vision Noise": add_machine_vision_noise,
        "14.5 HDR-Specific Noise": add_hdr_specific_noise
    }
    
    # Create different test images
    test_images = {
        "gradient": create_sample_image(pattern="gradient"),
        "constant": create_sample_image(pattern="constant"),
        "circles": create_sample_image(pattern="circles"),
        "checkerboard": create_sample_image(pattern="checkerboard")
    }
    
    # Default test image
    default_image = test_images["gradient"]
    
    # Process each noise type
    total_noise_types = len(noise_functions)
    print(f"Processing {total_noise_types} noise types...")
    
    # Process each noise type first to generate all images
    for i, (title, func) in enumerate(noise_functions.items(), 1):
        print(f"Processing {i}/{total_noise_types}: {title}")
        
        try:
            file_index = title.split()[0].replace('.', '_')
            filename = f"{OUTPUT_DIR}/{file_index}.png"
            
            # Check if file already exists to avoid reprocessing
            if not os.path.exists(filename):
                noisy_img, noise = func(default_image)
                display_noise_example_with_hist(title, default_image, noisy_img, noise, i, total_noise_types)
        except Exception as e:
            print(f"Error processing {title}: {e}")
    
    # Create an index HTML file with embedded images
    with open(f"{OUTPUT_DIR}/index.html", 'w') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Complete Noise Type Catalog</title>\n")
        f.write("<meta charset='UTF-8'>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n")
        f.write("h1 { color: #333; text-align: center; margin-bottom: 30px; }\n")
        f.write("h2 { color: #555; margin-top: 40px; background-color: #e0e0e0; padding: 10px; border-radius: 5px; }\n")
        f.write(".noise-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }\n")
        f.write(".noise-item { background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); overflow: hidden; transition: transform 0.3s; }\n")
        f.write(".noise-item:hover { transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }\n")
        f.write(".noise-title { padding: 10px; background-color: #0066cc; color: white; font-weight: bold; text-align: center; }\n")
        f.write(".noise-image { width: 100%; height: auto; display: block; }\n")
        f.write(".back-to-top { position: fixed; bottom: 20px; right: 20px; background-color: #0066cc; color: white; padding: 10px 15px; border-radius: 5px; text-decoration: none; }\n")
        f.write(".back-to-top:hover { background-color: #004c99; }\n")
        f.write(".toc { background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }\n")
        f.write(".toc h3 { margin-top: 0; }\n")
        f.write(".toc ul { list-style-type: none; padding: 0; column-count: 3; column-gap: 20px; }\n")
        f.write(".toc li { margin: 5px 0; }\n")
        f.write(".toc a { color: #0066cc; text-decoration: none; }\n")
        f.write(".toc a:hover { text-decoration: underline; }\n")
        f.write("@media (max-width: 768px) { .toc ul { column-count: 1; } .noise-grid { grid-template-columns: 1fr; } }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Complete Noise Type Catalog</h1>\n")
        
        # Create table of contents
        f.write("<div class='toc'>\n<h3>Table of Contents</h3>\n<ul>\n")
        for category_id in sorted(set([key.split('.')[0] for key in noise_functions.keys()])):
            category_name = {
                "1": "Statistical Distribution-Based Noise Types",
                "2": "Colored Noise Types",
                "3": "Procedural and Synthetic Noise Types",
                "4": "Physical and Device-Based Noise Types",
                "5": "Real-World and Environmental Noise Types",
                "6": "Additional Statistical Distributions",
                "7": "Physical and Quantum Noise",
                "8": "Specialized Fields Noise",
                "9": "Mathematical and Theoretical Noise",
                "10": "New Technology Noise",
                "11": "Combinational and Specific Noise",
                "12": "Environment and Special Effects Noise",
                "13": "Specialized Scientific Fields Noise",
                "14": "Newest Noise Types and Future Directions"
            }.get(category_id, f"Category {category_id}")
            
            f.write(f"<li><a href='#category-{category_id}'>{category_id}. {category_name}</a></li>\n")
        f.write("</ul>\n</div>\n")
        
        # Group by categories
        categories = {}
        for title in noise_functions.keys():
            category_id = title.split('.')[0]
            if category_id not in categories:
                categories[category_id] = []
            categories[category_id].append(title)
        
        # Write images by category
        for category_id in sorted(categories.keys()):
            category_name = {
                "1": "Statistical Distribution-Based Noise Types",
                "2": "Colored Noise Types",
                "3": "Procedural and Synthetic Noise Types",
                "4": "Physical and Device-Based Noise Types",
                "5": "Real-World and Environmental Noise Types",
                "6": "Additional Statistical Distributions",
                "7": "Physical and Quantum Noise",
                "8": "Specialized Fields Noise",
                "9": "Mathematical and Theoretical Noise",
                "10": "New Technology Noise",
                "11": "Combinational and Specific Noise",
                "12": "Environment and Special Effects Noise",
                "13": "Specialized Scientific Fields Noise",
                "14": "Newest Noise Types and Future Directions"
            }.get(category_id, f"Category {category_id}")
            
            f.write(f"<h2 id='category-{category_id}'>{category_id}. {category_name}</h2>\n")
            f.write("<div class='noise-grid'>\n")
            
            for title in sorted(categories[category_id]):
                file_index = title.split()[0].replace('.', '_')
                f.write("<div class='noise-item'>\n")
                f.write(f"<div class='noise-title'>{title}</div>\n")
                f.write(f"<img class='noise-image' src='{file_index}.png' alt='{title}' loading='lazy'>\n")
                f.write("</div>\n")
            
            f.write("</div>\n")
        
        # Add back to top button
        f.write("<a href='#' class='back-to-top'>Back to Top</a>\n")
        
        # Add footer with generation date
        import datetime
        generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"<div style='text-align: center; margin-top: 50px; color: #777; font-size: 0.8em;'>\n")
        f.write(f"Generated on: {generation_date}<br>")
        f.write(f"Total noise types: {total_noise_types}\n")
        f.write("</div>\n")
        
        f.write("</body>\n</html>\n")
    
    print(f"All noise examples saved to {OUTPUT_DIR}/")
    print(f"An index.html file has been created with embedded images for easy browsing.")

# Function to display and save noise examples with histogram
def display_noise_example_with_hist(title, image, noisy_image, noise, index, total):
    """Display and save a noise example with histogram of noise distribution"""
    # Create a more appropriate index for the file name
    file_index = title.split()[0].replace('.', '_')
    
    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(2, 3, height_ratios=[2, 1])
    
    # Original image
    ax_orig = plt.subplot(grid[0, 0])
    ax_orig.imshow(image, cmap='gray')
    ax_orig.set_title('Original Image')
    ax_orig.axis('off')
    
    # Noisy image
    ax_noisy = plt.subplot(grid[0, 1])
    ax_noisy.imshow(np.clip(noisy_image, 0, 255), cmap='gray')
    ax_noisy.set_title(f'Image with {title}')
    ax_noisy.axis('off')
    
    # Noise pattern
    ax_noise = plt.subplot(grid[0, 2])
    im = ax_noise.imshow(noise, cmap='viridis')
    ax_noise.set_title(f'{title} Pattern')
    ax_noise.axis('off')
    
    # Add colorbar
    divider = make_axes_locatable(ax_noise)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Noise histogram
    ax_hist = plt.subplot(grid[1, :])
    
    # Flatten noise for histogram
    flat_noise = noise.flatten()
    
    # For very sparse noise patterns (like salt & pepper, hot pixels),
    # filter out zeros to see the distribution better
    if np.count_nonzero(flat_noise) / flat_noise.size < 0.1:
        flat_noise = flat_noise[flat_noise != 0]
        if len(flat_noise) == 0:  # If no non-zero values, use original
            flat_noise = noise.flatten()
    
    # Remove outliers for better visualization
    if len(flat_noise) > 0:
        q1, q3 = np.percentile(flat_noise, [1, 99])
        bin_data = flat_noise[(flat_noise >= q1) & (flat_noise <= q3)]
        
        if len(bin_data) > 10:  # Only if we have enough data points
            # Calculate optimal number of bins using Freedman-Diaconis rule
            iqr = q3 - q1
            bin_width = 2 * iqr * (len(bin_data) ** (-1/3))
            if bin_width > 0:
                num_bins = int(np.ceil((np.max(bin_data) - np.min(bin_data)) / bin_width))
                num_bins = min(max(10, num_bins), 100)  # Keep bins between 10 and 100
            else:
                num_bins = 30
        else:
            num_bins = 30
        
        # Draw histogram
        counts, bins, patches = ax_hist.hist(
            bin_data, bins=num_bins, alpha=0.7, color='steelblue', 
            edgecolor='black', density=True
        )
        
        # Add a kernel density estimate if we have enough data
        if len(bin_data) > 30:
            try:
                from scipy.stats import gaussian_kde
                x = np.linspace(np.min(bin_data), np.max(bin_data), 1000)
                kde = gaussian_kde(bin_data)
                ax_hist.plot(x, kde(x), 'r-', linewidth=2)
            except:
                pass  # Skip KDE if it fails
    else:
        ax_hist.text(0.5, 0.5, "Insufficient data for histogram", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax_hist.transAxes)
    
    ax_hist.set_title(f'Noise Distribution Histogram - {title}')
    ax_hist.set_xlabel('Noise Value')
    ax_hist.set_ylabel('Probability Density')
    ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle(f"{index}/{total}: {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    filename = f"{OUTPUT_DIR}/{file_index}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()