NOISE_METADATA = {
    "advanced_compression_artifacts": {
        "title": "Advanced Compression Artifacts",
        "category": "New Technology",
        "description": "Formula: Artifacts from modern compression methods",
        "formula": "Artifacts from modern compression methods",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "quality": {
                "default": 30.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter quality"
            }
        },
        "code_snippet": "def add_advanced_compression_artifacts(image, quality=30, method='dct'):\n    \"\"\"\n    Formula: Artifacts from modern compression methods\n    \"\"\"\n    row, col = image.shape\n    \n    if method == 'dct':\n        # DCT-based compression similar to JPEG but with customization\n        # Divide image into 8x8 blocks\n        block_size = 8\n        result = np.copy(image)\n        \n        for i in range(0, row, block_size):\n            for j in range(0, col, block_size):\n                # Extract block\n                i_end = min(i + block_size, row)\n                j_end = min(j + block_size, col)\n                block = image[i:i_end, j:j_end]\n                \n                # Pad if necessary\n                if block.shape[0] < block_size or block.shape[1] < block_size:\n                    padded = np.zeros((block_size, block_size))\n                    padded[:block.shape[0], :block.shape[1]] = block\n                    block = padded\n                \n                # Apply DCT\n                dct_block = cv2.dct(block.astype(np.float32))\n                \n                # Quantization step (simulating compression)\n                quantization_matrix = np.ones((block_size, block_size))\n                for k in range(block_size):\n                    for l in range(block_size):\n                        quantization_matrix[k, l] = 1 + (k + l) * (100 - quality) / 10\n                \n                quantized = np.round(dct_block / quantization_matrix) * quantization_matrix\n                \n                # Inverse DCT\n                reconstructed = cv2.idct(quantized)\n                \n                # Place back into result\n                result[i:i_end, j:j_end] = reconstructed[:i_end-i, :j_end-j]\n    \n    elif method == 'wavelet':\n        # Wavelet-based compression (similar to JPEG2000)\n        coeffs = pywt.wavedec2(image, 'haar', level=3)\n        \n        # Threshold coefficients based on quality\n        threshold = np.percentile(np.abs(coeffs[0]), 100 - quality)\n        \n        # Apply thresholding to detail coefficients\n        new_coeffs = [coeffs[0]]  # Keep approximation coefficients\n        for detail_coeffs in coeffs[1:]:\n            new_detail = []\n            for component in detail_coeffs:\n                # Soft thresholding\n                thresholded = np.sign(component) * np.maximum(np.abs(component) - threshold, 0)\n                new_detail.append(thresholded)\n            new_coeffs.append(tuple(new_detail))\n        \n        # Reconstruct image\n        result = pywt.waverec2(new_coeffs, 'haar')\n        \n        # Ensure original dimensions\n        result = result[:row, :col]\n    \n    else:  # Default to simple method\n        # Save as JPEG to introduce compression artifacts\n        result = np.clip(image, 0, 255).astype(np.uint8)\n        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]\n        _, buffer = cv2.imencode('.jpg', result, encode_param)\n        result = cv2.imdecode(buffer, 0)\n    \n    # Calculate noise\n    noise = result.astype(float) - image\n    \n    return result, noise\n"
    },
    "anisotropic_noise": {
        "title": "Anisotropic Noise",
        "category": "Combinational and Specific",
        "description": "Formula: Noise with directional preference",
        "formula": "Noise with directional preference",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "angle": {
                "default": 45.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter angle"
            },
            "ratio": {
                "default": 0.2,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter ratio"
            }
        },
        "code_snippet": "def add_anisotropic_noise(image, intensity=20, angle=45, ratio=0.2):\n    \"\"\"\n    Formula: Noise with directional preference\n    \"\"\"\n    row, col = image.shape\n    \n    # Create base isotropic noise\n    base_noise = np.random.normal(0, 1, (row*2, col*2))\n    \n    # Apply directional blur\n    angle_rad = np.deg2rad(angle)\n    dx = int(np.cos(angle_rad) * 10)\n    dy = int(np.sin(angle_rad) * 10)\n    \n    # Create directional kernel\n    kernel_size = max(3, int(np.sqrt(dx**2 + dy**2)))\n    kernel = np.zeros((kernel_size, kernel_size))\n    center = kernel_size // 2\n    \n    # Draw line in kernel\n    for i in range(-center, center+1):\n        x = center + int(i * dx / 10)\n        y = center + int(i * dy / 10)\n        if 0 <= x < kernel_size and 0 <= y < kernel_size:\n            kernel[y, x] = 1\n    \n    # Normalize kernel\n    kernel = kernel / np.sum(kernel)\n    \n    # Apply directional filtering\n    filtered_noise = cv2.filter2D(base_noise, -1, kernel)\n    \n    # Crop to original size\n    noise = filtered_noise[row//2:row//2+row, col//2:col//2+col]\n    \n    # Rescale to desired intensity\n    noise = noise * intensity\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "astronomical_noise": {
        "title": "Astronomical Noise",
        "category": "Specialized Scientific Fields",
        "description": "Formula: Sky background, atmospheric seeing, and telescope tracking errors",
        "formula": "Sky background, atmospheric seeing, and telescope tracking errors",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "sky_brightness": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter sky_brightness"
            },
            "seeing": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter seeing"
            },
            "tracking_error": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter tracking_error"
            }
        },
        "code_snippet": "def add_astronomical_noise(image, sky_brightness=10, seeing=3, tracking_error=1):\n    \"\"\"\n    Formula: Sky background, atmospheric seeing, and telescope tracking errors\n    \"\"\"\n    row, col = image.shape\n    \n    # Sky background (additive brightness with Poisson characteristics)\n    sky = np.random.poisson(sky_brightness, (row, col))\n    \n    # Atmospheric seeing (blurring that varies over image)\n    blurred = gaussian_filter(image.astype(float), sigma=seeing)\n    seeing_noise = blurred - image\n    \n    # Telescope tracking error (slight motion blur in random direction)\n    angle = np.random.uniform(0, 2*np.pi)\n    dx, dy = tracking_error * np.cos(angle), tracking_error * np.sin(angle)\n    \n    kernel_size = max(3, int(2 * tracking_error) + 1)\n    kernel = np.zeros((kernel_size, kernel_size))\n    center = kernel_size // 2\n    \n    # Create motion blur kernel\n    for i in range(kernel_size):\n        for j in range(kernel_size):\n            dist = np.sqrt((i-center-dy)**2 + (j-center-dx)**2)\n            if dist < tracking_error:\n                kernel[i, j] = 1\n    \n    if kernel.sum() > 0:\n        kernel /= kernel.sum()\n        tracking_result = cv2.filter2D(image.astype(float), -1, kernel)\n        tracking_noise = tracking_result - image\n    else:\n        tracking_noise = np.zeros_like(image, dtype=float)\n    \n    # Combined noise\n    noise = sky + seeing_noise + tracking_noise\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "atmospheric_turbulence": {
        "title": "Atmospheric Turbulence",
        "category": "Procedural and Synthetic",
        "description": "Formula: Wavefront distortion due to atmospheric effects",
        "formula": "Wavefront distortion due to atmospheric effects",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "strength": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter strength"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_atmospheric_turbulence(image, strength=2.0, scale=10):\n    \"\"\"\n    Formula: Wavefront distortion due to atmospheric effects\n    \"\"\"\n    row, col = image.shape\n    result = np.zeros_like(image)\n    \n    # Generate displacement maps based on turbulence model\n    displacement_x = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=scale) * strength\n    displacement_y = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=scale) * strength\n    \n    # Create mesh grid\n    y_coords, x_coords = np.meshgrid(np.arange(col), np.arange(row))\n    \n    # Apply displacements\n    x_new = x_coords + displacement_x\n    y_new = y_coords + displacement_y\n    \n    # Clip to ensure valid coordinates\n    x_new = np.clip(x_new, 0, col-1).astype(np.int32)\n    y_new = np.clip(y_new, 0, row-1).astype(np.int32)\n    \n    # Remap image\n    for i in range(row):\n        for j in range(col):\n            result[i, j] = image[x_new[i, j], y_new[i, j]]\n    \n    # Calculate noise (difference from original)\n    noise = result.astype(float) - image\n    \n    return result, noise\n"
    },
    "avalanche_noise": {
        "title": "Avalanche Noise",
        "category": "Physical and Quantum",
        "description": "Formula: Random spikes due to avalanche breakdown in semiconductors",
        "formula": "Random spikes due to avalanche breakdown in semiconductors",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "prob_spike": {
                "default": 0.01,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter prob_spike"
            },
            "amplitude": {
                "default": 50.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            }
        },
        "code_snippet": "def add_avalanche_noise(image, prob_spike=0.01, amplitude=50):\n    \"\"\"\n    Formula: Random spikes due to avalanche breakdown in semiconductors\n    \"\"\"\n    row, col = image.shape\n    \n    # Base thermal noise\n    base_noise = np.random.normal(0, 10, (row, col))\n    \n    # Add random avalanche spikes\n    spike_mask = np.random.random((row, col)) < prob_spike\n    spikes = np.random.normal(0, amplitude, (row, col)) * spike_mask\n    \n    noise = base_noise + spikes\n    noisy_img = image + noise\n    \n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "banding_noise": {
        "title": "Banding Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Periodic pattern across image",
        "formula": "Periodic pattern across image",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "amplitude": {
                "default": 15.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            },
            "frequency": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter frequency"
            }
        },
        "code_snippet": "def add_banding_noise(image, amplitude=15, frequency=0.1, orientation='horizontal'):\n    \"\"\"\n    Formula: Periodic pattern across image\n    g(x,y) = f(x,y) + A*sin(2\u03c0fx) or g(x,y) = f(x,y) + A*sin(2\u03c0fy)\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    # Generate banding pattern\n    if orientation == 'horizontal':\n        noise = amplitude * np.sin(2 * np.pi * frequency * row * yy)\n    elif orientation == 'vertical':\n        noise = amplitude * np.sin(2 * np.pi * frequency * col * xx)\n    else:  # diagonal\n        noise = amplitude * np.sin(2 * np.pi * frequency * (xx + yy))\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "beta_noise": {
        "title": "Beta Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Beta(\u03b1,\u03b2)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Beta(\u03b1,\u03b2)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "a": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter a"
            },
            "b": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter b"
            },
            "scale": {
                "default": 100.0,
                "min": 0.0,
                "max": 400.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_beta_noise(image, a=2, b=2, scale=100):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Beta(\u03b1,\u03b2)\n    p(z) = (z^(\u03b1-1)*(1-z)^(\u03b2-1))/B(\u03b1,\u03b2) for 0 < z < 1\n    \"\"\"\n    row, col = image.shape\n    noise = (stats.beta.rvs(a, b, size=(row, col)) - 0.5) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "binomial_noise": {
        "title": "Binomial Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Bin(n,p)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Bin(n,p)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "n": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n"
            },
            "p": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter p"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_binomial_noise(image, n=20, p=0.5, scale=10):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Bin(n,p)\n    p(k) = (n choose k)*p^k*(1-p)^(n-k) for k \u2208 {0,1,...,n}\n    \"\"\"\n    row, col = image.shape\n    noise = (stats.binom.rvs(n, p, size=(row, col)) - n*p) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "black_noise": {
        "title": "Black Noise",
        "category": "Colored Noise",
        "description": "Formula: Silence with occasional spikes",
        "formula": "Silence with occasional spikes",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "sparsity": {
                "default": 0.99,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter sparsity"
            }
        },
        "code_snippet": "def add_black_noise(image, intensity=10, sparsity=0.99):\n    \"\"\"\n    Formula: Silence with occasional spikes\n    S(f) = 0 for most f, with rare, unpredictable spikes\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base noise\n    base_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create sparse mask\n    mask = np.random.random((row, col)) > sparsity\n    \n    # Apply mask to create sparse noise\n    noise = np.zeros((row, col))\n    noise[mask] = base_noise[mask] * intensity * 5  # Amplify the sparse points\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "blooming": {
        "title": "Blooming",
        "category": "Physical and Device-Based",
        "description": "Formula: Overflow of bright areas into neighboring pixels",
        "formula": "Overflow of bright areas into neighboring pixels",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "threshold": {
                "default": 200.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter threshold"
            },
            "spread": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter spread"
            }
        },
        "code_snippet": "def add_blooming(image, threshold=200, spread=5):\n    \"\"\"\n    Formula: Overflow of bright areas into neighboring pixels\n    g(x,y) = f(x,y) + B(x,y) where B is the blooming effect\n    \"\"\"\n    # Find bright areas\n    bright_mask = image > threshold\n    \n    # Dilate bright areas to simulate blooming\n    blooming_mask = cv2.dilate(bright_mask.astype(np.uint8), \n                              np.ones((spread, spread), np.uint8))\n    \n    # Create blooming effect (bright regions spread)\n    blooming = np.zeros_like(image, dtype=float)\n    blooming[blooming_mask > 0] = 255\n    \n    # Smooth the blooming\n    blooming = gaussian_filter(blooming, sigma=spread/2)\n    \n    # Apply blooming only to areas not already bright\n    effect_mask = (blooming_mask > 0) & (~bright_mask)\n    blooming_effect = np.zeros_like(image, dtype=float)\n    blooming_effect[effect_mask] = blooming[effect_mask] * 0.5\n    \n    # Add blooming to original image\n    bloomed = image + blooming_effect\n    \n    return np.clip(bloomed, 0, 255).astype(np.uint8), blooming_effect\n"
    },
    "blue_noise": {
        "title": "Blue Noise",
        "category": "Colored Noise",
        "description": "Formula: Power spectrum: S(f) \u221d f",
        "formula": "Power spectrum: S(f) \\propto f",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_blue_noise(image, intensity=20):\n    \"\"\"\n    Formula: Power spectrum: S(f) \u221d f\n    Generated by high-pass filtering white noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center\n    r = np.sqrt(xx**2 + yy**2)\n    \n    # Blue noise filter (emphasize high frequencies)\n    noise_fft = np.fft.fft2(white_noise)\n    filt = r\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "brown_noise": {
        "title": "Brown Noise",
        "category": "Colored Noise",
        "description": "Formula: B(t) = B(t-1) + W(t), where W(t) is white noise",
        "formula": "B(t) = B(t-1) + W(t), where W(t) is white noise",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_brown_noise(image, intensity=1.0):\n    \"\"\"\n    Formula: B(t) = B(t-1) + W(t), where W(t) is white noise\n    Power spectrum: S(f) \u221d 1/f\u00b2\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 1D Brown noise\n    brown_1d = np.zeros(row * col)\n    white_noise = np.random.normal(0, 1, row * col)\n    \n    for i in range(1, row * col):\n        brown_1d[i] = brown_1d[i-1] + white_noise[i] * intensity\n    \n    # Normalize and reshape\n    brown_1d = (brown_1d - brown_1d.min()) / (brown_1d.max() - brown_1d.min())\n    noise = (brown_1d.reshape(row, col) * 100) - 50\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "brownian_bridge_noise": {
        "title": "Brownian Bridge Noise",
        "category": "Colored Noise",
        "description": "Formula: Brownian motion constrained to return to starting point",
        "formula": "Brownian motion constrained to return to starting point",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "num_steps": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter num_steps"
            },
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_brownian_bridge_noise(image, num_steps=20, intensity=25):\n    \"\"\"\n    Formula: Brownian motion constrained to return to starting point\n    B_t = W_t - t*W_1 where W_t is a Wiener process (Brownian motion)\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate a 2D grid of Brownian bridges\n    t = np.linspace(0, 1, col)\n    noise = np.zeros((row, col))\n    \n    for i in range(row):\n        # Generate standard Brownian motion\n        dW = np.random.normal(0, 1/np.sqrt(col), col)\n        W = np.cumsum(dW)\n        \n        # Convert to Brownian bridge (equals 0 at t=0 and t=1)\n        B = W - t * W[-1]\n        \n        # Scale to desired intensity\n        noise[i, :] = B * intensity\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "cauchy_noise": {
        "title": "Cauchy Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Cauchy(x\u2080,\u03b3)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Cauchy(x\u2080,\u03b3)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_cauchy_noise(image, loc=0, scale=5):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Cauchy(x\u2080,\u03b3)\n    p(z) = 1/(\u03c0*\u03b3*(1+((z-x\u2080)/\u03b3)\u00b2))\n    \"\"\"\n    row, col = image.shape\n    noise = stats.cauchy.rvs(loc=loc, scale=scale, size=(row, col))\n    # Clip extreme values (Cauchy has heavy tails)\n    noise = np.clip(noise, -100, 100)\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "caustic_noise": {
        "title": "Caustic Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Based on refracted/reflected light patterns",
        "formula": "Based on refracted/reflected light patterns",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "distortion": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter distortion"
            }
        },
        "code_snippet": "def add_caustic_noise(image, scale=10.0, intensity=30, distortion=2.0):\n    \"\"\"\n    Formula: Based on refracted/reflected light patterns\n    Simulated using deformed noise fields\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base noise\n    base_noise = gaussian_filter(np.random.randn(row, col), sigma=scale)\n    \n    # Generate displacement fields\n    disp_x = gaussian_filter(np.random.randn(row, col), sigma=scale*2) * distortion\n    disp_y = gaussian_filter(np.random.randn(row, col), sigma=scale*2) * distortion\n    \n    # Create grid\n    y, x = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')\n    \n    # Apply displacement\n    x_new = np.clip(x + disp_x, 0, col-1).astype(int)\n    y_new = np.clip(y + disp_y, 0, row-1).astype(int)\n    \n    # Sample from base noise with displacement\n    noise = base_noise[y_new, x_new]\n    \n    # Enhance contrast to create caustic effect\n    noise = np.tanh(noise * 3)\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "chaotic_noise": {
        "title": "Chaotic Noise",
        "category": "Mathematical and Theoretical",
        "description": "Formula: Based on chaotic maps like logistic map: x_{n+1} = r*x_n*(1-x_n)",
        "formula": "Based on chaotic maps like logistic map: x_{n+1} = r*x_n*(1-x_n)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "r": {
                "default": 3.9,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter r"
            },
            "iterations": {
                "default": 100.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter iterations"
            },
            "scale": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_chaotic_noise(image, r=3.9, iterations=100, scale=25):\n    \"\"\"\n    Formula: Based on chaotic maps like logistic map: x_{n+1} = r*x_n*(1-x_n)\n    \"\"\"\n    row, col = image.shape\n    \n    # Initialize with random values between 0 and 1\n    noise = np.random.random((row, col))\n    \n    # Apply logistic map iterations\n    for _ in range(iterations):\n        noise = r * noise * (1 - noise)\n    \n    # Scale to appropriate range and center around zero\n    noise = (noise - 0.5) * scale\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "chi2_noise": {
        "title": "Chi2 Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows \u03c7\u00b2(k)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows \u03c7\u00b2(k)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "df": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter df"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_chi2_noise(image, df=1, scale=10):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows \u03c7\u00b2(k)\n    p(z) = (z^(k/2-1)*e^(-z/2))/(2^(k/2)*\u0393(k/2)) for z > 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.chi2.rvs(df, size=(row, col)) * scale - df * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "chromatic_aberration": {
        "title": "Chromatic Aberration",
        "category": "Physical and Device-Based",
        "description": "Formula: Color channels shifted relative to each other",
        "formula": "Color channels shifted relative to each other",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "shift": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter shift"
            }
        },
        "code_snippet": "def add_chromatic_aberration(image, shift=3):\n    \"\"\"\n    Formula: Color channels shifted relative to each other\n    For grayscale, we simulate by adding opposing edge responses\n    \"\"\"\n    # Create edge-detected version\n    edges = cv2.Canny(image, 50, 150).astype(float)\n    \n    # Create two shifted versions (one positive, one negative)\n    shifted_pos = np.zeros_like(image, dtype=float)\n    shifted_neg = np.zeros_like(image, dtype=float)\n    \n    if shift > 0:\n        shifted_pos[:-shift, :-shift] = edges[shift:, shift:]\n        shifted_neg[shift:, shift:] = -edges[:-shift, :-shift]\n    \n    # Combine effects for visualization in grayscale\n    aberration = shifted_pos + shifted_neg\n    \n    # Apply to image\n    aberrated = image + aberration * 0.5\n    \n    return np.clip(aberrated, 0, 255).astype(np.uint8), aberration\n"
    },
    "cosmic_noise": {
        "title": "Cosmic Noise",
        "category": "Specialized Fields",
        "description": "Formula: Random bright streaks from cosmic ray hits",
        "formula": "Random bright streaks from cosmic ray hits",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "rate": {
                "default": 0.001,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter rate"
            },
            "energy_range": {
                "default": 125.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter energy_range (default: (50, 200))"
            }
        },
        "code_snippet": "def add_cosmic_noise(image, rate=0.001, energy_range=(50, 200)):\n    \"\"\"\n    Formula: Random bright streaks from cosmic ray hits\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros((row, col))\n    \n    # Number of cosmic rays\n    num_rays = int(row * col * rate)\n    \n    for _ in range(num_rays):\n        # Random starting point\n        x = np.random.randint(0, col)\n        y = np.random.randint(0, row)\n        \n        # Random length and angle\n        length = np.random.randint(3, 20)\n        angle = np.random.random() * 2 * np.pi\n        \n        # Random energy\n        energy = np.random.uniform(energy_range[0], energy_range[1])\n        \n        # Create the ray\n        for i in range(length):\n            x_new = int(x + i * np.cos(angle))\n            y_new = int(y + i * np.sin(angle))\n            \n            if 0 <= x_new < col and 0 <= y_new < row:\n                # Energy decreases along the track\n                pixel_energy = energy * (1 - i/length)\n                noise[y_new, x_new] = pixel_energy\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "cracks_noise": {
        "title": "Cracks Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Random walk-based crack pattern",
        "formula": "Random walk-based crack pattern",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "n_cracks": {
                "default": 50.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n_cracks"
            },
            "length": {
                "default": 30.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter length"
            },
            "width": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter width"
            },
            "intensity": {
                "default": 100.0,
                "min": 0.0,
                "max": 400.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_cracks_noise(image, n_cracks=50, length=30, width=1, intensity=100):\n    \"\"\"\n    Formula: Random walk-based crack pattern\n    \"\"\"\n    row, col = image.shape\n    \n    # Initialize empty noise\n    noise = np.zeros((row, col))\n    \n    # Generate cracks\n    for _ in range(n_cracks):\n        # Random starting point\n        x = np.random.randint(0, col)\n        y = np.random.randint(0, row)\n        \n        # Random direction\n        angle = np.random.random() * 2 * np.pi\n        dx = np.cos(angle)\n        dy = np.sin(angle)\n        \n        # Generate crack as a random walk\n        for i in range(length):\n            # Small random perturbation to direction\n            angle += np.random.normal(0, 0.1)\n            dx = np.cos(angle)\n            dy = np.sin(angle)\n            \n            # Move to new position\n            x += dx\n            y += dy\n            \n            # Check boundaries\n            if x < 0 or x >= col or y < 0 or y >= row:\n                break\n                \n            # Draw point at current position\n            x_int, y_int = int(x), int(y)\n            \n            # Draw with width\n            for wy in range(-width, width+1):\n                for wx in range(-width, width+1):\n                    xi, yi = x_int + wx, y_int + wy\n                    if 0 <= xi < col and 0 <= yi < row:\n                        noise[yi, xi] = 1.0\n    \n    # Normalize and scale\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image - noise  # Subtract to make cracks dark\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), -noise\n"
    },
    "curl_noise": {
        "title": "Curl Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Based on the curl of a potential field",
        "formula": "Based on the curl of a potential field",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_curl_noise(image, scale=10.0, intensity=25):\n    \"\"\"\n    Formula: Based on the curl of a potential field\n    Curl(P) = (\u2202P_y/\u2202x - \u2202P_x/\u2202y)\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate two potential fields\n    p_x = gaussian_filter(np.random.randn(row+2, col+2), sigma=scale)\n    p_y = gaussian_filter(np.random.randn(row+2, col+2), sigma=scale)\n    \n    # Calculate derivatives\n    dy_px = p_x[2:, 1:-1] - p_x[:-2, 1:-1]\n    dx_py = p_y[1:-1, 2:] - p_y[1:-1, :-2]\n    \n    # Calculate curl\n    noise = dx_py - dy_px\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "dead_pixels": {
        "title": "Dead Pixels",
        "category": "Physical and Device-Based",
        "description": "Formula: Random pixels with zero value",
        "formula": "Random pixels with zero value",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "density": {
                "default": 0.001,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter density"
            }
        },
        "code_snippet": "def add_dead_pixels(image, density=0.001):\n    \"\"\"\n    Formula: Random pixels with zero value\n    g(x,y) = 0 if (x,y) is a dead pixel, f(x,y) otherwise\n    \"\"\"\n    row, col = image.shape\n    \n    # Create dead pixel mask\n    dead_mask = np.random.random((row, col)) < density\n    \n    # Create dead pixel effect\n    dead_pixels = np.zeros_like(image, dtype=float)\n    \n    # Add dead pixels to image\n    noisy_img = image.copy()\n    noisy_img[dead_mask] = 0\n    \n    # Calculate noise (negative of the original values at dead pixel locations)\n    noise = dead_pixels - image * dead_mask\n    \n    return noisy_img, noise\n"
    },
    "demosaicing_noise": {
        "title": "Demosaicing Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Artifacts from Bayer pattern interpolation",
        "formula": "Artifacts from Bayer pattern interpolation",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "pattern_size": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter pattern_size"
            }
        },
        "code_snippet": "def add_demosaicing_noise(image, pattern_size=2):\n    \"\"\"\n    Formula: Artifacts from Bayer pattern interpolation\n    \"\"\"\n    row, col = image.shape\n    \n    # Create a simulated Bayer pattern (temporarily reduce resolution)\n    bayer = np.zeros((row, col), dtype=np.uint8)\n    \n    # Simulate R, G, B channels with a Bayer pattern\n    # (RGGB pattern: R at (0,0), G at (0,1) and (1,0), B at (1,1))\n    for i in range(0, row, pattern_size):\n        for j in range(0, col, pattern_size):\n            if i+1 < row and j+1 < col:\n                # R channel\n                bayer[i, j] = image[i, j]\n                \n                # G channels (two positions)\n                if i+1 < row:\n                    bayer[i+1, j] = image[i+1, j]\n                if j+1 < col:\n                    bayer[i, j+1] = image[i, j+1]\n                \n                # B channel\n                if i+1 < row and j+1 < col:\n                    bayer[i+1, j+1] = image[i+1, j+1]\n    \n    # Simple interpolation to simulate demosaicing\n    interpolated = cv2.resize(\n        cv2.resize(bayer, (col//pattern_size, row//pattern_size), \n                  interpolation=cv2.INTER_AREA),\n        (col, row), interpolation=cv2.INTER_LINEAR)\n    \n    # Calculate noise (difference between original and interpolated)\n    noise = interpolated.astype(float) - image\n    \n    return interpolated, noise\n"
    },
    "diamond_square_noise": {
        "title": "Diamond Square Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Recursive subdivision algorithm with random displacement",
        "formula": "Recursive subdivision algorithm with random displacement",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "roughness": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter roughness"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_diamond_square_noise(image, roughness=0.5, intensity=40):\n    \"\"\"\n    Formula: Recursive subdivision algorithm with random displacement\n    Also known as plasma noise or cloud noise\n    \"\"\"\n    # Get closest power of 2 plus 1\n    row, col = image.shape\n    size = max(row, col)\n    power = int(np.ceil(np.log2(size - 1)))\n    n = 2**power + 1\n    \n    # Create grid for diamond-square algorithm\n    grid = np.zeros((n, n))\n    \n    # Set four corners to random values\n    grid[0, 0] = np.random.random()\n    grid[0, n-1] = np.random.random()\n    grid[n-1, 0] = np.random.random()\n    grid[n-1, n-1] = np.random.random()\n    \n    # Run diamond-square algorithm\n    step = n - 1\n    while step > 1:\n        half_step = step // 2\n        \n        # Diamond step\n        for i in range(half_step, n, step):\n            for j in range(half_step, n, step):\n                avg = (grid[i-half_step, j-half_step] + \n                       grid[i-half_step, j+half_step] + \n                       grid[i+half_step, j-half_step] + \n                       grid[i+half_step, j+half_step]) / 4.0\n                grid[i, j] = avg + (np.random.random() - 0.5) * roughness * step\n        \n        # Square step\n        for i in range(0, n, half_step):\n            j_start = half_step if i % step == 0 else 0\n            for j in range(j_start, n, step):\n                # Count valid neighbors\n                count = 0\n                avg = 0\n                \n                if i - half_step >= 0:\n                    avg += grid[i - half_step, j]\n                    count += 1\n                if i + half_step < n:\n                    avg += grid[i + half_step, j]\n                    count += 1\n                if j - half_step >= 0:\n                    avg += grid[i, j - half_step]\n                    count += 1\n                if j + half_step < n:\n                    avg += grid[i, j + half_step]\n                    count += 1\n                \n                avg /= count\n                grid[i, j] = avg + (np.random.random() - 0.5) * roughness * step\n        \n        step = half_step\n        roughness *= 0.5\n    \n    # Crop to original size\n    noise = grid[:row, :col]\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "diffraction_noise": {
        "title": "Diffraction Noise",
        "category": "Newest and Future Directions",
        "description": "Formula: Diffraction patterns from wave optics",
        "formula": "Diffraction patterns from wave optics",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "wavelength": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter wavelength"
            },
            "aperture": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter aperture"
            }
        },
        "code_snippet": "def add_diffraction_noise(image, wavelength=0.5, aperture=20):\n    \"\"\"\n    Formula: Diffraction patterns from wave optics\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid centered at image center\n    x = np.linspace(-col/2, col/2, col)\n    y = np.linspace(-row/2, row/2, row)\n    xx, yy = np.meshgrid(x, y)\n    rr = np.sqrt(xx**2 + yy**2)\n    \n    # Circular aperture diffraction pattern (Airy disk)\n    k = 2*np.pi/wavelength\n    airy_argument = k * aperture * rr / np.sqrt(row**2 + col**2)\n    airy_argument[airy_argument == 0] = 1e-10  # Avoid division by zero\n    airy_pattern = (2 * special.j1(airy_argument) / airy_argument)**2\n    \n    # Scale and center the pattern\n    diffraction = 20 * (airy_pattern - np.min(airy_pattern)) / (np.max(airy_pattern) - np.min(airy_pattern)) - 10\n    \n    # Make it more visible by applying it more strongly to bright areas\n    intensity_mask = image / 255.0\n    noise = diffraction * intensity_mask\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "dl_artifacts": {
        "title": "Dl Artifacts",
        "category": "New Technology",
        "description": "Formula: Artifacts typical in deep learning image processing",
        "formula": "Artifacts typical in deep learning image processing",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "block_size": {
                "default": 8.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter block_size"
            },
            "intensity": {
                "default": 0.7,
                "min": 0.0,
                "max": 2.8,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_dl_artifacts(image, block_size=8, intensity=0.7):\n    \"\"\"\n    Formula: Artifacts typical in deep learning image processing\n    \"\"\"\n    row, col = image.shape\n    \n    # Create a grid pattern (common in some neural networks)\n    grid_pattern = np.zeros((row, col))\n    for i in range(0, row, block_size):\n        for j in range(0, col, block_size):\n            grid_pattern[i:i+1, j:j+block_size] = 1\n            grid_pattern[i:i+block_size, j:j+1] = 1\n    \n    # Over-smoothing in some areas (common in denoisers/GANs)\n    smooth_mask = np.random.random((row, col)) < 0.3\n    smooth_regions = np.zeros((row, col))\n    \n    for i in range(0, row, block_size):\n        for j in range(0, col, block_size):\n            i_end = min(i + block_size, row)\n            j_end = min(j + block_size, col)\n            \n            if np.random.random() < 0.4:  # 40% of blocks are over-smoothed\n                # Calculate block average\n                block = image[i:i_end, j:j_end]\n                avg_value = np.mean(block)\n                \n                # Create smooth transition to average\n                smooth_regions[i:i_end, j:j_end] = avg_value - block\n    \n    # \"Checkerboard\" artifacts (common in deconvolution networks)\n    checkerboard = np.zeros((row, col))\n    for i in range(row):\n        for j in range(col):\n            if (i // (block_size//2) + j // (block_size//2)) % 2 == 0:\n                checkerboard[i, j] = 5\n            else:\n                checkerboard[i, j] = -5\n    \n    # Combine artifacts\n    noise = (grid_pattern * 10 + smooth_regions + checkerboard * 0.5) * intensity\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "dust_scratches": {
        "title": "Dust Scratches",
        "category": "Real-World and Environmental",
        "description": "Formula: Combination of random specks (dust) and lines (scratches)",
        "formula": "Combination of random specks (dust) and lines (scratches)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "dust_density": {
                "default": 0.001,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter dust_density"
            },
            "scratch_count": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter scratch_count"
            }
        },
        "code_snippet": "def add_dust_scratches(image, dust_density=0.001, scratch_count=20):\n    \"\"\"\n    Formula: Combination of random specks (dust) and lines (scratches)\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Add dust\n    dust_mask = np.random.random((row, col)) < dust_density\n    dust_sizes = np.random.randint(1, 5, size=np.sum(dust_mask))\n    \n    # Place dust with varying sizes\n    dust_indices = np.where(dust_mask)\n    for i, (y, x) in enumerate(zip(*dust_indices)):\n        size = dust_sizes[i]\n        y1, y2 = max(0, y-size), min(row, y+size+1)\n        x1, x2 = max(0, x-size), min(col, x+size+1)\n        \n        # Draw dust speck with random intensity\n        intensity = np.random.randint(-150, -50)\n        for dy in range(y1, y2):\n            for dx in range(x1, x2):\n                if (dy-y)**2 + (dx-x)**2 <= size**2:\n                    noise[dy, dx] = intensity\n    \n    # Add scratches\n    for _ in range(scratch_count):\n        # Random scratch starting point\n        x1, y1 = np.random.randint(0, col), np.random.randint(0, row)\n        \n        # Random scratch length and angle\n        length = np.random.randint(10, 50)\n        angle = np.random.random() * 2 * np.pi\n        \n        # Calculate end point\n        x2 = int(x1 + length * np.cos(angle))\n        y2 = int(y1 + length * np.sin(angle))\n        \n        # Draw scratch line\n        rr, cc = line(y1, x1, y2, x2)\n        valid_idx = (rr >= 0) & (rr < row) & (cc >= 0) & (cc < col)\n        rr, cc = rr[valid_idx], cc[valid_idx]\n        \n        # Set scratch intensity (usually dark)\n        intensity = np.random.randint(-150, -50)\n        noise[rr, cc] = intensity\n    \n    # Apply to image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "electrical_interference": {
        "title": "Electrical Interference",
        "category": "Environment and Special Effects",
        "description": "Formula: Various electrical interference patterns",
        "formula": "Various electrical interference patterns",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "frequency_range": {
                "default": 30.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter frequency_range (default: (10, 50))"
            },
            "amplitude": {
                "default": 15.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            }
        },
        "code_snippet": "def add_electrical_interference(image, frequency_range=(10, 50), amplitude=15):\n    \"\"\"\n    Formula: Various electrical interference patterns\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    # Choose random frequencies\n    freq_x = np.random.uniform(frequency_range[0], frequency_range[1])\n    freq_y = np.random.uniform(frequency_range[0], frequency_range[1])\n    \n    # Generate primary interference pattern\n    interference = amplitude * np.sin(2 * np.pi * freq_x * xx) * np.sin(2 * np.pi * freq_y * yy)\n    \n    # Add some TV-like scan lines\n    scanlines = np.zeros((row, col))\n    scanline_pattern = np.sin(2 * np.pi * 100 * yy)\n    scanline_mask = np.random.random((row, col)) < 0.1\n    scanlines[scanline_mask] = scanline_pattern[scanline_mask] * amplitude\n    \n    # Add occasional stronger interference bursts\n    burst_mask = np.random.random((row, col)) < 0.02\n    bursts = burst_mask * np.random.normal(0, amplitude * 2, (row, col))\n    \n    # Combine all interference effects\n    noise = interference + scanlines + bursts\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "exponential_noise": {
        "title": "Exponential Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Exp(\u03bb)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Exp(\\lambda)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_exponential_noise(image, scale=25):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Exp(\u03bb)\n    p(z) = \u03bb*e^(-\u03bbz) for z \u2265 0\n    \u03bb = 1/scale\n    \"\"\"\n    row, col = image.shape\n    noise = stats.expon.rvs(scale=scale, size=(row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "f_noise": {
        "title": "F Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows F(d1,d2)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows F(d1,d2)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "dfn": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter dfn"
            },
            "dfd": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter dfd"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_f_noise(image, dfn=5, dfd=2, scale=10):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows F(d1,d2)\n    Complex PDF formula based on beta function\n    \"\"\"\n    row, col = image.shape\n    noise = stats.f.rvs(dfn, dfd, size=(row, col)) * scale\n    noise -= np.mean(noise)  # Center the noise around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "fbm_noise": {
        "title": "Fbm Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Sum of octaves of noise with frequency and amplitude scaling",
        "formula": "Sum of octaves of noise with frequency and amplitude scaling",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "H": {
                "default": 0.7,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter H"
            },
            "octaves": {
                "default": 8.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_fbm_noise(image, H=0.7, octaves=8, intensity=40):\n    \"\"\"\n    Formula: Sum of octaves of noise with frequency and amplitude scaling\n    fBm(x,y) = \u03a3\u1d62\u208c\u2080\u1d57\u1d52 \u2099\u208b\u2081 amplitude_i * noise(frequency_i * (x,y))\n    where amplitude_i = persistence^(i*H) and frequency_i = lacunarity^i\n    \"\"\"\n    row, col = image.shape\n    \n    # Parameters\n    persistence = 0.5\n    lacunarity = 2.0\n    \n    noise = np.zeros((row, col))\n    amplitude = 1.0\n    frequency = 1.0\n    \n    for i in range(octaves):\n        # Generate noise at current frequency\n        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)\n        \n        # Add to accumulated noise\n        noise += amplitude * noise_layer\n        \n        # Update parameters for next octave\n        amplitude *= persistence ** H\n        frequency *= lacunarity\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "film_grain": {
        "title": "Film Grain",
        "category": "Physical and Device-Based",
        "description": "Formula: Approximated by filtered noise with log-normal characteristics",
        "formula": "Approximated by filtered noise with log-normal characteristics",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "size": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter size"
            }
        },
        "code_snippet": "def add_film_grain(image, intensity=0.5, size=3):\n    \"\"\"\n    Formula: Approximated by filtered noise with log-normal characteristics\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base noise (log-normal distribution)\n    mu, sigma = 0, 0.5\n    base_noise = np.random.lognormal(mu, sigma, (row, col)) - np.exp(mu + sigma**2/2)\n    \n    # Blur the noise to create grain structure\n    grain = gaussian_filter(base_noise, sigma=size)\n    \n    # Normalize and scale\n    grain = (grain - grain.min()) / (grain.max() - grain.min())\n    \n    # Apply grain (more visible in darker areas)\n    darkness = 1.0 - (image / 255.0)\n    weighted_grain = grain * darkness * intensity * 255\n    \n    noisy_img = image + weighted_grain\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), weighted_grain\n"
    },
    "fixed_pattern_noise": {
        "title": "Fixed Pattern Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Fixed spatial noise pattern",
        "formula": "Fixed spatial noise pattern",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "pattern_seed": {
                "default": 42.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter pattern_seed"
            }
        },
        "code_snippet": "def add_fixed_pattern_noise(image, intensity=15, pattern_seed=42):\n    \"\"\"\n    Formula: Fixed spatial noise pattern\n    g(x,y) = f(x,y) + P(x,y)\n    \"\"\"\n    row, col = image.shape\n    \n    # Set random seed to ensure pattern is fixed\n    np.random.seed(pattern_seed)\n    \n    # Generate fixed pattern\n    pattern = np.random.normal(0, 1, (row, col))\n    \n    # Apply some smoothing to make it more realistic\n    pattern = gaussian_filter(pattern, sigma=1.0)\n    \n    # Normalize and scale\n    pattern = intensity * (pattern - pattern.min()) / (pattern.max() - pattern.min()) - intensity/2\n    \n    # Reset random seed\n    np.random.seed()\n    \n    noisy_img = image + pattern\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), pattern\n"
    },
    "flicker_noise": {
        "title": "Flicker Noise",
        "category": "Physical and Quantum",
        "description": "Formula: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 1",
        "formula": "S(f) \\propto 1/f^\u03b1 where \u03b1 \u2248 1",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "alpha": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter alpha"
            }
        },
        "code_snippet": "def add_flicker_noise(image, intensity=25, alpha=1.0):\n    \"\"\"\n    Formula: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 1\n    Similar to pink noise but specifically for electronic circuits\n    \"\"\"\n    # This is very similar to pink noise but specifically for electronic contexts\n    return add_pink_noise(image, intensity)\n"
    },
    "flow_noise": {
        "title": "Flow Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Noise advected along a flow field",
        "formula": "Noise advected along a flow field",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "iterations": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter iterations"
            }
        },
        "code_snippet": "def add_flow_noise(image, scale=10.0, intensity=30, iterations=3):\n    \"\"\"\n    Formula: Noise advected along a flow field\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base noise\n    noise = gaussian_filter(np.random.randn(row, col), sigma=scale)\n    \n    # Generate flow field (two components)\n    flow_x = gaussian_filter(np.random.randn(row, col), sigma=scale*2)\n    flow_y = gaussian_filter(np.random.randn(row, col), sigma=scale*2)\n    \n    # Create grid\n    y, x = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')\n    \n    # Advect noise along flow field iteratively\n    for _ in range(iterations):\n        # Calculate new positions\n        x_new = np.clip(x + flow_x, 0, col-1)\n        y_new = np.clip(y + flow_y, 0, row-1)\n        \n        # Sample noise at new positions using bilinear interpolation\n        x0 = np.floor(x_new).astype(int)\n        y0 = np.floor(y_new).astype(int)\n        x1 = np.minimum(x0 + 1, col-1)\n        y1 = np.minimum(y0 + 1, row-1)\n        \n        wx = x_new - x0\n        wy = y_new - y0\n        \n        # Bilinear interpolation\n        new_noise = (noise[y0, x0] * (1-wx) * (1-wy) +\n                    noise[y0, x1] * wx * (1-wy) +\n                    noise[y1, x0] * (1-wx) * wy +\n                    noise[y1, x1] * wx * wy)\n        \n        noise = new_noise\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "fog": {
        "title": "Fog",
        "category": "Real-World and Environmental",
        "description": "Formula: g(x,y) = f(x,y)*(1-t) + L*t",
        "formula": "g(x,y) = f(x,y)*(1-t) + L*t",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_fog(image, intensity=0.5):\n    \"\"\"\n    Formula: g(x,y) = f(x,y)*(1-t) + L*t\n    where t is fog transmission and L is fog light\n    \"\"\"\n    # Create transmission map (decreases with distance)\n    trans = np.ones_like(image, dtype=float) * (1 - intensity)\n    \n    # Fog light (usually bright gray)\n    fog_light = 240\n    \n    # Apply fog model\n    foggy = image * trans + fog_light * (1 - trans)\n    \n    # Calculate noise (difference due to fog)\n    noise = foggy - image\n    \n    return np.clip(foggy, 0, 255).astype(np.uint8), noise\n"
    },
    "gabor_noise": {
        "title": "Gabor Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Based on summation of randomly positioned Gabor kernels",
        "formula": "Based on summation of randomly positioned Gabor kernels",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "k_scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter k_scale"
            },
            "intensity": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "orient": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter orient"
            },
            "freq": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter freq"
            }
        },
        "code_snippet": "def add_gabor_noise(image, k_scale=5.0, intensity=20, orient=0, freq=0.1):\n    \"\"\"\n    Formula: Based on summation of randomly positioned Gabor kernels\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate random positions\n    n_kernels = 300\n    positions = np.random.rand(n_kernels, 2)\n    positions[:, 0] *= row\n    positions[:, 1] *= col\n    \n    # Create grid\n    yy, xx = np.meshgrid(np.arange(row), np.arange(col), indexing='ij')\n    \n    # Initialize noise\n    noise = np.zeros((row, col))\n    \n    # Parameters for Gabor kernel\n    sigma = 1.0 / freq\n    theta = orient * np.pi / 180  # Convert to radians\n    lambda_val = 1.0 / freq\n    gamma = 1.0  # Aspect ratio\n    \n    # Calculate rotated coordinates once\n    x_theta = xx * np.cos(theta) + yy * np.sin(theta)\n    y_theta = -xx * np.sin(theta) + yy * np.cos(theta)\n    \n    # Create Gabor kernel\n    gabor = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / lambda_val)\n    \n    # Generate noise by summing shifted Gabor kernels\n    for pos in positions:\n        y, x = int(pos[0]), int(pos[1])\n        if 0 <= y < row and 0 <= x < col:\n            # Random weight\n            weight = np.random.normal(0, 1)\n            \n            # Calculate bounds\n            kernel_size = int(k_scale * sigma)\n            y1 = max(0, y - kernel_size)\n            y2 = min(row, y + kernel_size + 1)\n            x1 = max(0, x - kernel_size)\n            x2 = min(col, x + kernel_size + 1)\n            \n            # Calculate kernel bounds\n            ky1 = kernel_size - (y - y1)\n            ky2 = kernel_size + (y2 - y)\n            kx1 = kernel_size - (x - x1)\n            kx2 = kernel_size + (x2 - x)\n            \n            # Ensure bounds are valid\n            if y2 > y1 and x2 > x1 and ky2 > ky1 and kx2 > kx1:\n                gabor_small = gabor[ky1:ky2, kx1:kx2]\n                try:\n                    noise[y1:y2, x1:x2] += weight * gabor_small\n                except ValueError:\n                    # Handle potential dimension mismatch\n                    continue\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "gamma_noise": {
        "title": "Gamma Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Gamma(k, \u03b8)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Gamma(k, \u03b8)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "shape": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter shape"
            },
            "scale": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_gamma_noise(image, shape=1.0, scale=40.0):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Gamma(k, \u03b8)\n    p(z) = (z^(k-1) * e^(-z/\u03b8)) / (\u03b8^k * \u0393(k)) for z > 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.gamma.rvs(shape, scale=scale, size=(row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "gaussian_noise": {
        "title": "Gaussian Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows N(mean, sigma\u00b2)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows N(mean, sigma\u00b2)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "mean": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter mean"
            },
            "sigma": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter sigma"
            }
        },
        "code_snippet": "def add_gaussian_noise(image, mean=0, sigma=25):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows N(mean, sigma\u00b2)\n    \u03b7(x,y) ~ (1/(sigma*sqrt(2\u03c0))) * e^(-(z-mean)\u00b2/(2*sigma\u00b2))\n    \"\"\"\n    row, col = image.shape\n    noise = np.random.normal(mean, sigma, (row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "glitch": {
        "title": "Glitch",
        "category": "Real-World and Environmental",
        "description": "Formula: Random horizontal shifts of image segments",
        "formula": "Random horizontal shifts of image segments",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "num_glitches": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter num_glitches"
            }
        },
        "code_snippet": "def add_glitch(image, intensity=0.5, num_glitches=10):\n    \"\"\"\n    Formula: Random horizontal shifts of image segments\n    \"\"\"\n    row, col = image.shape\n    result = image.copy()\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Define glitch regions\n    glitch_lines = np.random.randint(0, row, size=num_glitches)\n    glitch_lines.sort()  # Sort to get proper segments\n    \n    # Add start and end points\n    glitch_lines = np.append(np.insert(glitch_lines, 0, 0), row)\n    \n    # Apply glitches to segments\n    for i in range(len(glitch_lines) - 1):\n        start_y = glitch_lines[i]\n        end_y = glitch_lines[i+1]\n        \n        if end_y > start_y:\n            # Random shift for this segment\n            shift = np.random.randint(-int(col * intensity), int(col * intensity))\n            \n            if shift != 0:\n                segment = result[start_y:end_y, :]\n                shifted = np.zeros_like(segment)\n                \n                if shift > 0:\n                    shifted[:, shift:] = segment[:, :-shift]\n                else:\n                    shifted[:, :shift] = segment[:, -shift:]\n                \n                # Apply shifted segment\n                result[start_y:end_y, :] = shifted\n                \n                # Calculate noise for this segment\n                noise[start_y:end_y, :] = shifted - image[start_y:end_y, :]\n    \n    return result.astype(np.uint8), noise\n"
    },
    "gr_noise": {
        "title": "Gr Noise",
        "category": "Physical and Quantum",
        "description": "Formula: Random two-state noise caused by generation-recombination processes",
        "formula": "Random two-state noise caused by generation-recombination processes",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "rate": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter rate"
            },
            "amplitude": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            }
        },
        "code_snippet": "def add_gr_noise(image, rate=0.1, amplitude=20):\n    \"\"\"\n    Formula: Random two-state noise caused by generation-recombination processes\n    \"\"\"\n    row, col = image.shape\n    \n    # Create a random two-state process (0 or 1)\n    state = np.zeros((row, col))\n    for i in range(row):\n        current_state = 0\n        for j in range(col):\n            # Chance to flip state\n            if np.random.random() < rate:\n                current_state = 1 - current_state\n            state[i, j] = current_state\n    \n    # Convert to noise with amplitude\n    noise = (state * 2 - 1) * amplitude\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "gravitational_wave_noise": {
        "title": "Gravitational Wave Noise",
        "category": "Specialized Scientific Fields",
        "description": "Formula: Models noise sources in gravitational wave detection",
        "formula": "Models noise sources in gravitational wave detection",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "seismic_amplitude": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter seismic_amplitude"
            },
            "newtonian_scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter newtonian_scale"
            },
            "quantum_scale": {
                "default": 2.0,
                "min": 0.0,
                "max": 8.0,
                "step": 1,
                "description": "Parameter quantum_scale"
            }
        },
        "code_snippet": "def add_gravitational_wave_noise(image, seismic_amplitude=10, newtonian_scale=5, quantum_scale=2):\n    \"\"\"\n    Formula: Models noise sources in gravitational wave detection\n    \"\"\"\n    row, col = image.shape\n    \n    # Seismic noise (low frequency)\n    t = np.linspace(0, 10*np.pi, col)\n    seismic_base = np.sin(t/5) + 0.5*np.sin(t/2) + 0.2*np.sin(t/1.5)\n    seismic = np.tile(seismic_base, (row, 1)) * seismic_amplitude\n    \n    # Newtonian noise (changes in gravitational field)\n    newtonian = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=20) * newtonian_scale\n    \n    # Quantum noise (shot noise in laser interferometer)\n    quantum = np.random.normal(0, quantum_scale, (row, col))\n    \n    # Combined noise\n    noise = seismic + newtonian + quantum\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "green_noise": {
        "title": "Green Noise",
        "category": "Colored Noise",
        "description": "Formula: Band-limited noise with mid-range frequencies only",
        "formula": "Band-limited noise with mid-range frequencies only",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "band_center": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter band_center"
            },
            "band_width": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter band_width"
            }
        },
        "code_snippet": "def add_green_noise(image, intensity=25, band_center=0.5, band_width=0.1):\n    \"\"\"\n    Formula: Band-limited noise with mid-range frequencies only\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center\n    r = np.sqrt(xx**2 + yy**2)\n    \n    # Band-pass filter centered on mid-range frequencies\n    filt = np.exp(-((r - band_center)**2) / (2 * band_width**2))\n    \n    # Apply filter in frequency domain\n    noise_fft = np.fft.fft2(white_noise)\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "grey_noise": {
        "title": "Grey Noise",
        "category": "Colored Noise",
        "description": "Formula: White noise filtered to match human auditory perception",
        "formula": "White noise filtered to match human auditory perception",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_grey_noise(image, intensity=25):\n    \"\"\"\n    Formula: White noise filtered to match human auditory perception\n    In images, can be simulated with perceptual weighting\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base white noise\n    noise = np.random.normal(0, intensity, (row, col))\n    \n    # Apply perceptual weighting (simple approximation)\n    y = np.linspace(0, 1, row)\n    perceptual_weight = np.sqrt(y).reshape(-1, 1)\n    weighted_noise = noise * perceptual_weight\n    \n    noisy_img = image + weighted_noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), weighted_noise\n"
    },
    "gumbel_noise": {
        "title": "Gumbel Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: PDF: p(z) = (1/\u03b2)*exp(-(z-\u03bc)/\u03b2 - exp(-(z-\u03bc)/\u03b2))",
        "formula": "PDF: p(z) = (1/\u03b2)*exp(-(z-\\mu)/\u03b2 - exp(-(z-\\mu)/\u03b2))",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_gumbel_noise(image, loc=0, scale=15):\n    \"\"\"\n    Formula: PDF: p(z) = (1/\u03b2)*exp(-(z-\u03bc)/\u03b2 - exp(-(z-\u03bc)/\u03b2))\n    \"\"\"\n    row, col = image.shape\n    noise = gumbel_r.rvs(loc=loc, scale=scale, size=(row, col))\n    noise = noise - np.mean(noise)  # Center around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "hdr_artifacts": {
        "title": "Hdr Artifacts",
        "category": "New Technology",
        "description": "Formula: Artifacts from merging different exposures",
        "formula": "Artifacts from merging different exposures",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_hdr_artifacts(image, intensity=0.5):\n    \"\"\"\n    Formula: Artifacts from merging different exposures\n    \"\"\"\n    row, col = image.shape\n    \n    # Simulate multiple exposures\n    exp_low = np.clip(image * 0.5, 0, 255).astype(np.uint8)\n    exp_mid = np.clip(image, 0, 255).astype(np.uint8)\n    exp_high = np.clip(image * 2, 0, 255).astype(np.uint8)\n    \n    # Add noise to each exposure\n    exp_low = np.clip(exp_low + np.random.normal(0, 20, (row, col)), 0, 255).astype(np.uint8)\n    exp_mid = np.clip(exp_mid + np.random.normal(0, 10, (row, col)), 0, 255).astype(np.uint8)\n    exp_high = np.clip(exp_high + np.random.normal(0, 30, (row, col)), 0, 255).astype(np.uint8)\n    \n    # Create weight maps\n    weight_low = np.clip((exp_low.astype(float) - 0) * (255 - exp_low.astype(float)) / 255, 0, 1)\n    weight_mid = np.clip((exp_mid.astype(float) - 0) * (255 - exp_mid.astype(float)) / 255, 0, 1)\n    weight_high = np.clip((exp_high.astype(float) - 0) * (255 - exp_high.astype(float)) / 255, 0, 1)\n    \n    # Normalize weights\n    sum_weights = weight_low + weight_mid + weight_high\n    sum_weights[sum_weights == 0] = 1  # Avoid division by zero\n    \n    weight_low /= sum_weights\n    weight_mid /= sum_weights\n    weight_high /= sum_weights\n    \n    # Merge exposures\n    hdr = (exp_low.astype(float) * weight_low + \n           exp_mid.astype(float) * weight_mid + \n           exp_high.astype(float) * weight_high)\n    \n    # Tone mapping (simple gamma correction)\n    gamma = 1.0 - intensity * 0.5\n    mapped = np.power(hdr / 255, gamma) * 255\n    \n    # Create ghost artifacts in areas of movement\n    ghost_mask = np.random.random((row, col)) < 0.02 * intensity\n    ghost_regions = gaussian_filter(ghost_mask.astype(float), sigma=2) * 30 * intensity\n    \n    # Apply ghost artifacts\n    hdr_with_ghosts = mapped + ghost_regions\n    \n    # Calculate noise (difference from original)\n    noise = hdr_with_ghosts - image\n    \n    return np.clip(hdr_with_ghosts, 0, 255).astype(np.uint8), noise\n"
    },
    "hdr_specific_noise": {
        "title": "Hdr Specific Noise",
        "category": "New Technology",
        "description": "Formula: Noise specifically affecting high dynamic range imaging",
        "formula": "Noise specifically affecting high dynamic range imaging",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "photon_scale": {
                "default": 0.1,
                "min": 0.0,
                "max": 0.4,
                "step": 0.01,
                "description": "Parameter photon_scale"
            },
            "over_exposure": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter over_exposure"
            },
            "tone_mapping": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter tone_mapping"
            }
        },
        "code_snippet": "def add_hdr_specific_noise(image, photon_scale=0.1, over_exposure=True, tone_mapping=True):\n    \"\"\"\n    Formula: Noise specifically affecting high dynamic range imaging\n    \"\"\"\n    row, col = image.shape\n    \n    # Simulate HDR image by stretching dynamic range\n    # (This is just a simulation for noise purposes, not actual HDR)\n    hdr_sim = image.astype(float) ** 0.5 * 255\n    \n    # Photon noise proportional to brightness\n    photon_noise = np.random.poisson(np.maximum(hdr_sim * photon_scale, 1)) / photon_scale - hdr_sim\n    \n    # Sensor saturation and over-exposure\n    if over_exposure:\n        bright_mask = hdr_sim > 240\n        saturation = np.zeros((row, col))\n        \n        # Add blooming around saturated pixels\n        if np.any(bright_mask):\n            dilated_mask = cv2.dilate(bright_mask.astype(np.uint8), np.ones((5, 5), np.uint8))\n            bloom_mask = dilated_mask & (~bright_mask)\n            saturation[bloom_mask] = 20\n            \n            # Hard clip saturated pixels to 255\n            saturation[bright_mask] = 255 - hdr_sim[bright_mask]\n    else:\n        saturation = np.zeros((row, col))\n    \n    # Tone mapping artifacts\n    if tone_mapping:\n        # Simulate local tone mapping artifacts\n        tone_map = np.zeros((row, col))\n        \n        # Local contrast enhancement leading to halos\n        edges = cv2.Canny(image, 50, 150).astype(float)\n        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))\n        edges_expanded = gaussian_filter(edges_dilated, sigma=3) / 255.0\n        \n        # Create halo effect around edges\n        tone_map = edges_expanded * 20 * np.sign(128 - image)\n    else:\n        tone_map = np.zeros((row, col))\n    \n    # Combined noise\n    noise = photon_noise + saturation + tone_map\n    \n    # Apply to original image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "heteroskedastic_noise": {
        "title": "Heteroskedastic Noise",
        "category": "Combinational and Specific",
        "description": "Formula: Gaussian noise with variance dependent on pixel intensity",
        "formula": "Gaussian noise with variance dependent on pixel intensity",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "min_sigma": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter min_sigma"
            },
            "max_sigma": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter max_sigma"
            }
        },
        "code_snippet": "def add_heteroskedastic_noise(image, min_sigma=5, max_sigma=30):\n    \"\"\"\n    Formula: Gaussian noise with variance dependent on pixel intensity\n    \"\"\"\n    row, col = image.shape\n    \n    # Make noise variance proportional to pixel intensity\n    sigma = min_sigma + (max_sigma - min_sigma) * (image / 255.0)\n    \n    # Generate noise with pixel-dependent variance\n    noise = np.zeros((row, col))\n    for i in range(row):\n        for j in range(col):\n            noise[i, j] = np.random.normal(0, sigma[i, j])\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "holographic_noise": {
        "title": "Holographic Noise",
        "category": "Newest and Future Directions",
        "description": "Formula: Interference patterns and speckle noise in holographic displays",
        "formula": "Interference patterns and speckle noise in holographic displays",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "interference_scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter interference_scale"
            },
            "speckle_scale": {
                "default": 0.2,
                "min": 0.0,
                "max": 0.8,
                "step": 0.01,
                "description": "Parameter speckle_scale"
            }
        },
        "code_snippet": "def add_holographic_noise(image, interference_scale=15, speckle_scale=0.2):\n    \"\"\"\n    Formula: Interference patterns and speckle noise in holographic displays\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    x = np.linspace(0, col-1, col)\n    y = np.linspace(0, row-1, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    # Reference beam pattern\n    ref_angle = np.random.uniform(0, 2*np.pi)\n    kx_ref = np.cos(ref_angle)\n    ky_ref = np.sin(ref_angle)\n    reference = np.cos(2*np.pi*(kx_ref*xx/col + ky_ref*yy/row)*10)\n    \n    # Object beam (simulated from image)\n    object_beam = image / 255.0\n    \n    # Interference pattern\n    interference = interference_scale * ((reference + object_beam)**2 - reference**2 - object_beam**2)\n    \n    # Add laser speckle\n    speckle = np.random.normal(0, 1, (row, col))\n    speckle = gaussian_filter(speckle, sigma=1)\n    speckle = speckle_scale * speckle * image / 255.0 * 50\n    \n    # Combined noise\n    noise = interference + speckle\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "hot_pixels": {
        "title": "Hot Pixels",
        "category": "Physical and Device-Based",
        "description": "Formula: Random pixels with very high values",
        "formula": "Random pixels with very high values",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "density": {
                "default": 0.001,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter density"
            },
            "intensity": {
                "default": 255.0,
                "min": 0.0,
                "max": 1020.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_hot_pixels(image, density=0.001, intensity=255):\n    \"\"\"\n    Formula: Random pixels with very high values\n    g(x,y) = intensity if (x,y) is a hot pixel, f(x,y) otherwise\n    \"\"\"\n    row, col = image.shape\n    \n    # Create hot pixel mask\n    hot_mask = np.random.random((row, col)) < density\n    \n    # Create hot pixel effect\n    hot_pixels = np.zeros_like(image, dtype=float)\n    hot_pixels[hot_mask] = intensity\n    \n    # Add hot pixels to image\n    noisy_img = image.copy()\n    noisy_img[hot_mask] = intensity\n    \n    # Calculate noise (just the hot pixels)\n    noise = hot_pixels - image * hot_mask\n    \n    return noisy_img, noise\n"
    },
    "hypergeom_noise": {
        "title": "Hypergeom Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Hypergeometric(M,n,N)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Hypergeometric(M,n,N)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "M": {
                "default": 100.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter M"
            },
            "n": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n"
            },
            "N": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter N"
            },
            "scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_hypergeom_noise(image, M=100, n=20, N=10, scale=5):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Hypergeometric(M,n,N)\n    p(k) = (n choose k)*(M-n choose N-k)/(M choose N) for max(0,N+n-M) \u2264 k \u2264 min(n,N)\n    \"\"\"\n    row, col = image.shape\n    noise = (stats.hypergeom.rvs(M, n, N, size=(row, col)) - n*N/M) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "jitter_noise": {
        "title": "Jitter Noise",
        "category": "Specialized Fields",
        "description": "Formula: Random horizontal shift of each scanline",
        "formula": "Random horizontal shift of each scanline",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "jitter_max": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter jitter_max"
            }
        },
        "code_snippet": "def add_jitter_noise(image, jitter_max=2):\n    \"\"\"\n    Formula: Random horizontal shift of each scanline\n    \"\"\"\n    row, col = image.shape\n    result = np.zeros_like(image)\n    \n    # Apply random shift to each row\n    for i in range(row):\n        jitter = np.random.randint(-jitter_max, jitter_max+1)\n        if jitter >= 0:\n            result[i, jitter:] = image[i, :col-jitter]\n        else:\n            result[i, :col+jitter] = image[i, -jitter:]\n    \n    # Calculate noise as difference\n    noise = result.astype(float) - image\n    \n    return result, noise\n"
    },
    "johnson_nyquist_noise": {
        "title": "Johnson Nyquist Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: P = 4kTR\u2206f",
        "formula": "P = 4kTR\u2206f",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "temperature": {
                "default": 300.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter temperature"
            },
            "bandwidth": {
                "default": 10000.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter bandwidth"
            },
            "resistance": {
                "default": 1000.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter resistance"
            }
        },
        "code_snippet": "def add_johnson_nyquist_noise(image, temperature=300, bandwidth=10000, resistance=1000):\n    \"\"\"\n    Formula: P = 4kTR\u2206f\n    P - noise power, k - Boltzmann constant, T - temperature,\n    R - resistance, \u2206f - bandwidth\n    \"\"\"\n    row, col = image.shape\n    # Boltzmann constant\n    k = 1.38e-23\n    \n    # Calculate power of thermal noise\n    noise_power = 4 * k * temperature * resistance * bandwidth\n    \n    # Standard deviation is sqrt of power\n    sigma = np.sqrt(noise_power) * 1e10  # Scale to make it visible\n    \n    # Generate Gaussian noise with calculated power\n    noise = np.random.normal(0, sigma, (row, col))\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "jpeg_noise": {
        "title": "Jpeg Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Artifacts from lossy JPEG compression",
        "formula": "Artifacts from lossy JPEG compression",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "quality": {
                "default": 40.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter quality"
            }
        },
        "code_snippet": "def add_jpeg_noise(image, quality=40):\n    \"\"\"\n    Formula: Artifacts from lossy JPEG compression\n    \"\"\"\n    # Ensure we have a valid image\n    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)\n    \n    # Save to memory buffer with JPEG compression\n    ret, buffer = cv2.imencode('.jpg', img_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])\n    \n    # Decode from buffer\n    compressed_img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)\n    \n    # Calculate noise (difference between original and compressed)\n    noise = compressed_img.astype(float) - image\n    \n    return compressed_img, noise\n"
    },
    "k_distribution_noise": {
        "title": "K Distribution Noise",
        "category": "Specialized Fields",
        "description": "Formula: Product of Gamma and Square root of Gamma random variables",
        "formula": "Product of Gamma and Square root of Gamma random variables",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "shape": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter shape"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_k_distribution_noise(image, shape=2.0, scale=10.0):\n    \"\"\"\n    Formula: Product of Gamma and Square root of Gamma random variables\n    \"\"\"\n    row, col = image.shape\n    \n    # K-distribution can be simulated as a compound distribution\n    gamma1 = np.random.gamma(shape, scale, (row, col))\n    gamma2 = np.random.gamma(shape, 1.0, (row, col))\n    \n    # K-distributed random variable\n    k_dist = np.sqrt(gamma1 * gamma2)\n    \n    # Center around zero for additive noise\n    noise = k_dist - np.mean(k_dist)\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "laplacian_noise": {
        "title": "Laplacian Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Laplace(\u03bc,b)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Laplace(\\mu,b)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_laplacian_noise(image, loc=0, scale=20):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Laplace(\u03bc,b)\n    p(z) = (1/(2b))*e^(-|z-\u03bc|/b)\n    \"\"\"\n    row, col = image.shape\n    noise = stats.laplace.rvs(loc=loc, scale=scale, size=(row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "lens_blur": {
        "title": "Lens Blur",
        "category": "Physical and Device-Based",
        "description": "Formula: Convolution with disk kernel",
        "formula": "Convolution with disk kernel",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "radius": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter radius"
            }
        },
        "code_snippet": "def add_lens_blur(image, radius=3):\n    \"\"\"\n    Formula: Convolution with disk kernel\n    g(x,y) = f(x,y) \u2297 K(r)\n    \"\"\"\n    # Create disk kernel\n    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]\n    mask = x**2 + y**2 <= radius**2\n    kernel = np.zeros((2*radius+1, 2*radius+1))\n    kernel[mask] = 1\n    kernel /= kernel.sum()\n    \n    # Apply blur\n    blurred = cv2.filter2D(image, -1, kernel)\n    \n    # Calculate \"noise\" (difference between original and blurred)\n    noise = blurred.astype(float) - image\n    \n    return blurred, noise\n"
    },
    "lens_flare": {
        "title": "Lens Flare",
        "category": "Real-World and Environmental",
        "description": "Formula: Bright spots and streaks simulating lens flare",
        "formula": "Bright spots and streaks simulating lens flare",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.7,
                "min": 0.0,
                "max": 2.8,
                "step": 1,
                "description": "Parameter intensity"
            },
            "num_flares": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter num_flares"
            }
        },
        "code_snippet": "def add_lens_flare(image, intensity=0.7, num_flares=5):\n    \"\"\"\n    Formula: Bright spots and streaks simulating lens flare\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Create a main flare source\n    center_y, center_x = row // 2, col // 2\n    \n    # Add main glare around center\n    y, x = np.ogrid[:row, :col]\n    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)\n    max_dist = np.sqrt(center_x**2 + center_y**2)\n    glare = 255 * np.exp(-dist**2 / (2 * (max_dist/4)**2)) * intensity\n    noise += glare\n    \n    # Add secondary flares along a line\n    for i in range(1, num_flares + 1):\n        # Position flares along line through center\n        flare_pos = i / (num_flares + 1)\n        fx = int(center_x * (1 - flare_pos))\n        fy = int(center_y * (1 - flare_pos))\n        \n        # Random flare size\n        size = np.random.randint(10, 30)\n        \n        # Create flare\n        dist = np.sqrt((x - fx)**2 + (y - fy)**2)\n        flare = 255 * np.exp(-dist**2 / (2 * size**2)) * intensity * (1 - i/(num_flares+1))\n        noise += flare\n    \n    # Apply lens flare to image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "levy_noise": {
        "title": "Levy Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: PDF: p(z) = sqrt(c/(2\u03c0))*exp(-c/(2(z-\u03bc)))/((z-\u03bc)^(3/2)) for z > \u03bc",
        "formula": "PDF: p(z) = sqrt(c/(2\u03c0))*exp(-c/(2(z-\\mu)))/((z-\\mu)^(3/2)) for z > \\mu",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_levy_noise(image, loc=0, scale=1):\n    \"\"\"\n    Formula: PDF: p(z) = sqrt(c/(2\u03c0))*exp(-c/(2(z-\u03bc)))/((z-\u03bc)^(3/2)) for z > \u03bc\n    \"\"\"\n    row, col = image.shape\n    # L\u00e9vy distribution has very heavy tails, so we need to scale it down\n    noise = levy.rvs(loc=loc, scale=scale, size=(row, col))\n    # Clip extreme values\n    noise = np.clip(noise, -30, 30)\n    # Scale to reasonable intensity\n    noise = noise / 3\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "levy_stable_noise": {
        "title": "Levy Stable Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: Generalization of many distributions including Gaussian, Cauchy, L\u00e9vy",
        "formula": "Generalization of many distributions including Gaussian, Cauchy, L\u00e9vy",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "alpha": {
                "default": 1.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter alpha"
            },
            "beta": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter beta"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_levy_stable_noise(image, alpha=1.5, beta=0, scale=10):\n    \"\"\"\n    Formula: Generalization of many distributions including Gaussian, Cauchy, L\u00e9vy\n    \"\"\"\n    row, col = image.shape\n    \n    # Parameters for the L\u00e9vy stable distribution\n    # alpha \u2208 (0, 2], beta \u2208 [-1, 1]\n    alpha = min(max(alpha, 0.01), 2.0)\n    beta = min(max(beta, -1.0), 1.0)\n    \n    # Generate uniformly distributed random variables\n    u = np.random.uniform(-np.pi/2, np.pi/2, (row, col))\n    w = np.random.exponential(1.0, (row, col))\n    \n    # Generate L\u00e9vy stable distribution based on formulas\n    if alpha == 1.0:\n        # Special case\n        noise = (2/np.pi) * ((np.pi/2 + beta * u) * np.tan(u) - \n                              beta * np.log(w * np.cos(u) / (np.pi/2 + beta * u)))\n    else:\n        # General case\n        zeta = -beta * np.tan(np.pi * alpha / 2)\n        term1 = np.sin(alpha * (u + zeta)) / np.power(np.cos(u), 1/alpha)\n        term2 = np.power(np.cos(u - alpha * (u + zeta)) / w, (1-alpha)/alpha)\n        noise = term1 * term2\n    \n    # Scale and center around zero\n    noise = (noise - np.median(noise)) * scale\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "logistic_noise": {
        "title": "Logistic Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Logistic(\u03bc,s)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Logistic(\\mu,s)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_logistic_noise(image, loc=0, scale=15):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Logistic(\u03bc,s)\n    p(z) = e^(-(z-\u03bc)/s)/(s*(1+e^(-(z-\u03bc)/s))\u00b2)\n    \"\"\"\n    row, col = image.shape\n    noise = stats.logistic.rvs(loc=loc, scale=scale, size=(row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "lognormal_noise": {
        "title": "Lognormal Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows LogN(\u03bc,\u03c3)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows LogN(\\mu,\\sigma)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "mean": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter mean"
            },
            "sigma": {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 1,
                "description": "Parameter sigma"
            },
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_lognormal_noise(image, mean=0, sigma=1, scale=10):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows LogN(\u03bc,\u03c3)\n    p(z) = (1/(z\u03c3\u221a(2\u03c0)))*e^(-(ln(z)-\u03bc)\u00b2/(2\u03c3\u00b2)) for z > 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.lognorm.rvs(s=sigma, scale=np.exp(mean), size=(row, col)) * scale\n    noise -= np.mean(noise)  # Center the noise around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "machine_vision_noise": {
        "title": "Machine Vision Noise",
        "category": "Newest and Future Directions",
        "description": "Formula: Noise specific to machine vision and robotic imaging systems",
        "formula": "Noise specific to machine vision and robotic imaging systems",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "lighting_variation": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter lighting_variation"
            },
            "motion_blur": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter motion_blur"
            },
            "sensor_effects": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter sensor_effects"
            }
        },
        "code_snippet": "def add_machine_vision_noise(image, lighting_variation=True, motion_blur=True, sensor_effects=True):\n    \"\"\"\n    Formula: Noise specific to machine vision and robotic imaging systems\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros((row, col))\n    \n    # Lighting variation (flicker, shadows)\n    if lighting_variation:\n        # Create lighting map\n        t = np.linspace(0, 1, col)\n        flicker = 1.0 + 0.2 * np.sin(2*np.pi*t*10)  # Temporal flicker\n        \n        # Spatial lighting variation (shadows)\n        shadow_mask = np.ones((row, col))\n        for _ in range(3):\n            cx, cy = np.random.randint(0, col), np.random.randint(0, row)\n            radius = np.random.randint(30, 100)\n            \n            # Create shadow\n            y, x = np.ogrid[:row, :col]\n            dist = np.sqrt((x - cx)**2 + (y - cy)**2)\n            shadow = np.exp(-dist**2 / (2*radius**2))\n            shadow_mask -= shadow * 0.3\n        \n        shadow_mask = np.clip(shadow_mask, 0.6, 1.0)\n        \n        # Apply lighting variations\n        lighting_noise = image * (np.outer(np.ones(row), flicker) * shadow_mask - 1.0)\n        noise += lighting_noise\n    \n    # Motion blur from robot movement\n    if motion_blur:\n        kernel_size = np.random.randint(3, 9)\n        angle = np.random.uniform(0, 180)\n        \n        # Create motion blur kernel\n        kernel = np.zeros((kernel_size, kernel_size))\n        center = kernel_size // 2\n        \n        angle_rad = np.deg2rad(angle)\n        dx = np.cos(angle_rad)\n        dy = np.sin(angle_rad)\n        \n        for i in range(kernel_size):\n            x = center + dx * (i - center)\n            y = center + dy * (i - center)\n            x_floor, y_floor = int(np.floor(x)), int(np.floor(y))\n            \n            if 0 <= x_floor < kernel_size and 0 <= y_floor < kernel_size:\n                kernel[y_floor, x_floor] = 1\n        \n        if kernel.sum() > 0:\n            kernel /= kernel.sum()\n            \n            # Apply blur\n            blurred = cv2.filter2D(image.astype(float), -1, kernel)\n            motion_noise = blurred - image\n            noise += motion_noise\n    \n    # Industrial sensor effects (banding, gain inconsistency)\n    if sensor_effects:\n        # Row/column banding\n        for i in range(0, row, np.random.randint(10, 30)):\n            if np.random.random() < 0.2:\n                width = np.random.randint(1, 3)\n                if i + width < row:\n                    noise[i:i+width, :] += np.random.normal(0, 5)\n        \n        # Gain inconsistency\n        gain_map = np.ones((row, col))\n        for _ in range(5):\n            cx, cy = np.random.randint(0, col), np.random.randint(0, row)\n            radius = np.random.randint(20, 60)\n            \n            # Create gain variation\n            y, x = np.ogrid[:row, :col]\n            dist = np.sqrt((x - cx)**2 + (y - cy)**2)\n            gain_variation = 1.0 + 0.1 * np.exp(-dist**2 / (2*radius**2))\n            gain_map *= gain_variation\n        \n        gain_noise = image * (gain_map - 1.0)\n        noise += gain_noise\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "maxwell_noise": {
        "title": "Maxwell Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Maxwell-Boltzmann distribution",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Maxwell-Boltzmann distribution",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_maxwell_noise(image, scale=25):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Maxwell-Boltzmann distribution\n    p(z) = sqrt(2/\u03c0)*(z\u00b2/a\u00b3)*e^(-z\u00b2/(2a\u00b2)) for z \u2265 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.maxwell.rvs(scale=scale, size=(row, col))\n    noise -= np.mean(noise)  # Center the noise around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "mixed_reality_noise": {
        "title": "Mixed Reality Noise",
        "category": "Newest and Future Directions",
        "description": "Formula: Display, tracking, and reprojection artifacts in AR/VR systems",
        "formula": "Display, tracking, and reprojection artifacts in AR/VR systems",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "display_artifacts": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter display_artifacts"
            },
            "tracking_jitter": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter tracking_jitter"
            },
            "reprojection": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter reprojection"
            }
        },
        "code_snippet": "def add_mixed_reality_noise(image, display_artifacts=True, tracking_jitter=True, reprojection=True):\n    \"\"\"\n    Formula: Display, tracking, and reprojection artifacts in AR/VR systems\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros((row, col))\n    \n    # Display artifacts (screen door effect, mura)\n    if display_artifacts:\n        # Screen door effect (visible pixel grid)\n        grid = np.zeros((row, col))\n        for i in range(0, row, 3):\n            grid[i:i+1, :] = -10\n        for j in range(0, col, 3):\n            grid[:, j:j+1] = -10\n        \n        # Mura (panel inconsistency)\n        mura = np.zeros((row, col))\n        for i in range(0, row, 20):\n            for j in range(0, col, 20):\n                i_end = min(i + 20, row)\n                j_end = min(j + 20, col)\n                mura[i:i_end, j:j_end] = np.random.normal(0, 3)\n        \n        noise += grid + mura\n    \n    # Tracking jitter (small random movements)\n    if tracking_jitter:\n        jitter_x = np.random.normal(0, 1, row)\n        jitter_y = np.random.normal(0, 1, row)\n        \n        jittered = np.zeros((row, col))\n        for i in range(row):\n            dx, dy = int(jitter_x[i]), int(jitter_y[i])\n            \n            for j in range(col):\n                new_j, new_i = j + dx, i + dy\n                if 0 <= new_i < row and 0 <= new_j < col:\n                    jittered[i, j] = image[new_i, new_j]\n                else:\n                    jittered[i, j] = image[i, j]\n        \n        jitter_noise = jittered - image\n        noise += jitter_noise\n    \n    # Reprojection artifacts (misalignment when head moves)\n    if reprojection:\n        # Simulate depth map (closer objects have higher values)\n        depth_map = np.ones((row, col)) * 128\n        \n        # Add some random depth variation\n        for _ in range(20):\n            cx, cy = np.random.randint(0, col), np.random.randint(0, row)\n            radius = np.random.randint(10, 50)\n            \n            # Create circle\n            y, x = np.ogrid[:row, :col]\n            mask = (x - cx)**2 + (y - cy)**2 <= radius**2\n            \n            # Assign depth\n            depth = np.random.randint(50, 200)\n            depth_map[mask] = depth\n        \n        # Apply reprojection based on depth\n        shift_x, shift_y = np.random.randint(-3, 4, 2)\n        \n        reprojected = np.zeros((row, col))\n        for i in range(row):\n            for j in range(col):\n                # Scale shift based on depth (closer objects move more)\n                depth_factor = (255 - depth_map[i, j]) / 255.0\n                dx = int(shift_x * depth_factor)\n                dy = int(shift_y * depth_factor)\n                \n                new_j, new_i = j + dx, i + dy\n                if 0 <= new_i < row and 0 <= new_j < col:\n                    reprojected[i, j] = image[new_i, new_j]\n                else:\n                    reprojected[i, j] = 0  # Missing data (black)\n        \n        reprojection_noise = reprojected - image\n        noise += reprojection_noise\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "moire_pattern": {
        "title": "Moire Pattern",
        "category": "Physical and Device-Based",
        "description": "Formula: Interference pattern from superimposed regular patterns",
        "formula": "Interference pattern from superimposed regular patterns",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "freq1": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter freq1"
            },
            "freq2": {
                "default": 11.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter freq2"
            },
            "angle1": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter angle1"
            },
            "angle2": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter angle2"
            },
            "amplitude": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            }
        },
        "code_snippet": "def add_moire_pattern(image, freq1=10, freq2=11, angle1=0, angle2=5, amplitude=20):\n    \"\"\"\n    Formula: Interference pattern from superimposed regular patterns\n    g(x,y) = f(x,y) + A*sin(2\u03c0f\u2081(x*cos(\u03b8\u2081)+y*sin(\u03b8\u2081)))*sin(2\u03c0f\u2082(x*cos(\u03b8\u2082)+y*sin(\u03b8\u2082)))\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    y, x = np.mgrid[:row, :col]\n    \n    # Convert angles to radians\n    angle1_rad = np.deg2rad(angle1)\n    angle2_rad = np.deg2rad(angle2)\n    \n    # Create pattern 1\n    pattern1 = np.sin(2 * np.pi * freq1 * (x * np.cos(angle1_rad) + y * np.sin(angle1_rad)) / col)\n    \n    # Create pattern 2\n    pattern2 = np.sin(2 * np.pi * freq2 * (x * np.cos(angle2_rad) + y * np.sin(angle2_rad)) / col)\n    \n    # Combine patterns to create moir\u00e9\n    noise = amplitude * pattern1 * pattern2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "mosaic_noise": {
        "title": "Mosaic Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Based on Voronoi diagram",
        "formula": "Based on Voronoi diagram",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "n_points": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n_points"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_mosaic_noise(image, n_points=20, intensity=40):\n    \"\"\"\n    Formula: Based on Voronoi diagram \n    Similar to Worley but using cell index instead of distance\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate random seed points\n    points = np.random.rand(n_points, 2)\n    points[:, 0] *= row\n    points[:, 1] *= col\n    \n    # Create Voronoi cells\n    xx, yy = np.meshgrid(np.arange(col), np.arange(row))\n    noise = np.zeros((row, col))\n    \n    # For each pixel, find the closest seed point\n    for i in range(n_points):\n        dist = np.sqrt((yy - points[i, 0])**2 + (xx - points[i, 1])**2)\n        \n        # For the first point, initialize the closest indices and distances\n        if i == 0:\n            closest_idx = np.zeros((row, col), dtype=int)\n            closest_dist = dist\n        else:\n            # Update if this point is closer\n            mask = dist < closest_dist\n            closest_dist[mask] = dist[mask]\n            closest_idx[mask] = i\n    \n    # Assign random values to each cell\n    cell_values = np.random.random(n_points) * 2 - 1  # Range [-1, 1]\n    for i in range(n_points):\n        noise[closest_idx == i] = cell_values[i]\n    \n    # Scale to desired intensity\n    noise = intensity * noise\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "motion_blur": {
        "title": "Motion Blur",
        "category": "Physical and Device-Based",
        "description": "Formula: Convolution with motion kernel",
        "formula": "Convolution with motion kernel",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "length": {
                "default": 15.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter length"
            },
            "angle": {
                "default": 45.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter angle"
            }
        },
        "code_snippet": "def add_motion_blur(image, length=15, angle=45):\n    \"\"\"\n    Formula: Convolution with motion kernel\n    g(x,y) = f(x,y) \u2297 M(length, angle)\n    \"\"\"\n    # Create motion blur kernel\n    kernel = np.zeros((length, length))\n    \n    # Convert angle to radians\n    angle_rad = np.deg2rad(angle)\n    \n    # Calculate x, y components of the directional vector\n    dx = np.cos(angle_rad)\n    dy = np.sin(angle_rad)\n    \n    # Create line of ones for the kernel\n    center = length // 2\n    for i in range(length):\n        x = int(center + dx * (i - center))\n        y = int(center + dy * (i - center))\n        if 0 <= x < length and 0 <= y < length:\n            kernel[y, x] = 1\n    \n    # Normalize kernel\n    kernel /= kernel.sum()\n    \n    # Apply motion blur\n    blurred = cv2.filter2D(image, -1, kernel)\n    \n    # Calculate \"noise\" (difference between original and blurred)\n    noise = blurred.astype(float) - image\n    \n    return blurred, noise\n"
    },
    "multifractal_noise": {
        "title": "Multifractal Noise",
        "category": "Mathematical and Theoretical",
        "description": "Formula: Extension of fractals with non-constant fractal dimension",
        "formula": "Extension of fractals with non-constant fractal dimension",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "octaves": {
                "default": 6.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "lacunarity": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter lacunarity"
            }
        },
        "code_snippet": "def add_multifractal_noise(image, intensity=30, octaves=6, lacunarity=2.0):\n    \"\"\"\n    Formula: Extension of fractals with non-constant fractal dimension\n    \"\"\"\n    row, col = image.shape\n    \n    # Base noise\n    noise = np.zeros((row, col))\n    \n    # Create different fractal dimensions for different regions\n    H_map = gaussian_filter(np.random.random((row, col)), sigma=30)\n    H_map = 0.2 + 0.6 * H_map  # Range from 0.2 to 0.8\n    \n    # Initialize values\n    frequency = 1.0\n    amplitude = 1.0\n    \n    for i in range(octaves):\n        # Generate noise layer\n        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)\n        \n        # Apply varying H parameter from H_map\n        octave_contribution = np.zeros((row, col))\n        for y in range(row):\n            for x in range(col):\n                # Calculate amplitude based on local H value\n                local_amplitude = amplitude * (frequency ** (-H_map[y, x]))\n                octave_contribution[y, x] = local_amplitude * noise_layer[y, x]\n        \n        # Add contribution to noise\n        noise += octave_contribution\n        \n        # Update parameters for next octave\n        frequency *= lacunarity\n        amplitude *= 0.5\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "nakagami_noise": {
        "title": "Nakagami Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: PDF: p(z) = (2*m^m)/(\u0393(m)*\u03a9^m)*z^(2m-1)*exp(-m*z\u00b2/\u03a9) for z \u2265 0",
        "formula": "PDF: p(z) = (2*m^m)/(\u0393(m)*\u03a9^m)*z^(2m-1)*exp(-m*z\u00b2/\u03a9) for z \u2265 0",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "nu": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter nu"
            },
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_nakagami_noise(image, nu=1, loc=0, scale=15):\n    \"\"\"\n    Formula: PDF: p(z) = (2*m^m)/(\u0393(m)*\u03a9^m)*z^(2m-1)*exp(-m*z\u00b2/\u03a9) for z \u2265 0\n    \"\"\"\n    row, col = image.shape\n    noise = nakagami.rvs(nu, loc=loc, scale=scale, size=(row, col))\n    noise = noise - np.mean(noise)  # Center around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "nbinom_noise": {
        "title": "Nbinom Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows NB(r,p)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows NB(r,p)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "n": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n"
            },
            "p": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter p"
            },
            "scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_nbinom_noise(image, n=10, p=0.5, scale=5):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows NB(r,p)\n    p(k) = (k+r-1 choose k)*(1-p)^r*p^k for k \u2265 0\n    \"\"\"\n    row, col = image.shape\n    noise = (stats.nbinom.rvs(n, p, size=(row, col)) - n*(1-p)/p) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "neuroimaging_noise": {
        "title": "Neuroimaging Noise",
        "category": "Specialized Scientific Fields",
        "description": "Formula: Physiological noise, head motion, and scanner drift",
        "formula": "Physiological noise, head motion, and scanner drift",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "physiological_scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter physiological_scale"
            },
            "motion_scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter motion_scale"
            },
            "scanner_drift": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter scanner_drift"
            }
        },
        "code_snippet": "def add_neuroimaging_noise(image, physiological_scale=15, motion_scale=10, scanner_drift=5):\n    \"\"\"\n    Formula: Physiological noise, head motion, and scanner drift\n    \"\"\"\n    row, col = image.shape\n    \n    # Physiological noise (cardiac, respiratory cycles)\n    t = np.linspace(0, 2*np.pi, col)\n    cardiac = np.sin(t*15)  # Faster oscillation for cardiac cycle\n    respiratory = np.sin(t*4)  # Slower oscillation for respiratory cycle\n    \n    physio_pattern = cardiac + respiratory\n    physio_pattern = physio_pattern / np.max(np.abs(physio_pattern))\n    \n    # Apply spatial variation\n    y_gradient = np.linspace(0, 1, row)[:, np.newaxis]\n    physiological = physio_pattern * y_gradient * physiological_scale\n    \n    # Head motion (random translations/rotations between volumes)\n    if np.random.random() < 0.3:  # 30% chance of motion artifact\n        shift_y = np.random.randint(-motion_scale, motion_scale+1)\n        shift_x = np.random.randint(-motion_scale, motion_scale+1)\n        \n        motion = np.zeros((row, col))\n        if shift_y != 0 or shift_x != 0:\n            # Apply shift\n            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])\n            shifted = cv2.warpAffine(image.astype(float), M, (col, row))\n            motion = shifted - image\n    else:\n        motion = np.zeros((row, col))\n    \n    # Scanner drift (slow signal change over time)\n    drift = np.linspace(0, scanner_drift, col)\n    drift_2d = np.tile(drift, (row, 1))\n    \n    # Combined noise\n    noise = physiological + motion + drift_2d\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "nonstationary_noise": {
        "title": "Nonstationary Noise",
        "category": "Combinational and Specific",
        "description": "Formula: Noise with spatially varying statistics",
        "formula": "Noise with spatially varying statistics",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity_range": {
                "default": 17.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity_range (default: (5, 30))"
            }
        },
        "code_snippet": "def add_nonstationary_noise(image, intensity_range=(5, 30)):\n    \"\"\"\n    Formula: Noise with spatially varying statistics\n    \"\"\"\n    row, col = image.shape\n    \n    # Create gradually varying intensity map\n    t = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(t, y)\n    \n    # Create varying intensity pattern\n    intensity_map = intensity_range[0] + (intensity_range[1] - intensity_range[0]) * (\n        0.5 * np.sin(xx * 6 * np.pi) * np.cos(yy * 4 * np.pi) + 0.5)\n    \n    # Create varying noise type (mix of Gaussian and salt-and-pepper)\n    gaussian_noise = np.random.normal(0, 1, (row, col))\n    salt_mask = np.random.random((row, col)) < 0.01\n    pepper_mask = np.random.random((row, col)) < 0.01\n    \n    # Combine noise types with spatially varying intensity\n    noise = gaussian_noise * intensity_map\n    noise[salt_mask] = 255\n    noise[pepper_mask] = -255\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "orange_noise": {
        "title": "Orange Noise",
        "category": "Colored Noise",
        "description": "Formula: Power spectrum: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 1",
        "formula": "Power spectrum: S(f) \\propto 1/f^\u03b1 where \u03b1 \u2248 1",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "alpha": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter alpha"
            }
        },
        "code_snippet": "def add_orange_noise(image, intensity=20, alpha=1.0):\n    \"\"\"\n    Formula: Power spectrum: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 1\n    Between red and pink noise, less bass-heavy than red noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center (avoid division by zero)\n    r = np.sqrt(xx**2 + yy**2)\n    r[r == 0] = r[r > 0].min()\n    \n    # 1/f^alpha filter\n    noise_fft = np.fft.fft2(white_noise)\n    filt = 1 / (r**alpha)\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "pareto_noise": {
        "title": "Pareto Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Pareto(\u03b1,xm)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Pareto(\u03b1,xm)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "b": {
                "default": 1.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter b"
            },
            "scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_pareto_noise(image, b=1.5, scale=5):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Pareto(\u03b1,xm)\n    p(z) = (\u03b1*xm^\u03b1)/z^(\u03b1+1) for z \u2265 xm\n    \"\"\"\n    row, col = image.shape\n    noise = (stats.pareto.rvs(b, size=(row, col)) - b/(b-1)) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "perlin_noise": {
        "title": "Perlin Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Complex procedural noise function using gradient interpolation",
        "formula": "Complex procedural noise function using gradient interpolation",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "octaves": {
                "default": 6.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "persistence": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter persistence"
            },
            "lacunarity": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter lacunarity"
            },
            "intensity": {
                "default": 50.0,
                "min": 0.0,
                "max": 200.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_perlin_noise(image, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, intensity=50):\n    \"\"\"\n    Formula: Complex procedural noise function using gradient interpolation\n    P(x,y) = \u03a3\u1d62\u208c\u2081\u1d57\u1d52 \u2099 (persistence^i) * noise(x*lacunarity^i, y*lacunarity^i)\n    \"\"\"\n    if not has_perlin:\n        # Generate alternative to Perlin noise if package not available\n        return add_simplex_noise(image, scale, octaves, persistence, lacunarity, intensity)\n        \n    row, col = image.shape\n    \n    # Initialize Perlin noise generator\n    noise_gen = PerlinNoise(octaves=octaves, seed=42)\n    \n    # Generate noise\n    noise = np.zeros((row, col))\n    for i in range(row):\n        for j in range(col):\n            noise[i][j] = noise_gen([i/scale, j/scale])\n    \n    # Scale noise to desired intensity\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "phase_noise": {
        "title": "Phase Noise",
        "category": "Specialized Fields",
        "description": "Formula: Random phase modulation of periodic signal",
        "formula": "Random phase modulation of periodic signal",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "frequency": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter frequency"
            }
        },
        "code_snippet": "def add_phase_noise(image, intensity=5, frequency=10):\n    \"\"\"\n    Formula: Random phase modulation of periodic signal\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    # Create phase noise by adding a filtered random component to the phase\n    phase_error = gaussian_filter(np.random.normal(0, 1, (row, col)), sigma=5) * intensity\n    \n    # Apply phase noise to a sinusoidal pattern\n    noise = np.sin(2 * np.pi * frequency * (xx + yy + phase_error/100))\n    \n    # Scale to appropriate intensity\n    noise = noise * intensity\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "photon_limited_noise": {
        "title": "Photon Limited Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Extremely low light imaging where photon counting is limiting factor",
        "formula": "Extremely low light imaging where photon counting is limiting factor",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "photon_factor": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter photon_factor"
            }
        },
        "code_snippet": "def add_photon_limited_noise(image, photon_factor=0.5):\n    \"\"\"\n    Formula: Extremely low light imaging where photon counting is limiting factor\n    \"\"\"\n    row, col = image.shape\n    \n    # Scale image to very low photon counts\n    max_photons = 255 * photon_factor\n    \n    # Convert to \"photon counts\" (0 to max_photons)\n    photon_counts = np.maximum(image * (max_photons/255), 0.001)\n    \n    # Generate Poisson noise based on photon counts\n    noisy_photons = np.random.poisson(photon_counts)\n    \n    # Convert back to intensity\n    noisy_img = noisy_photons * (255/max_photons)\n    \n    # Calculate the noise component\n    noise = noisy_img - image\n    \n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "pink_noise": {
        "title": "Pink Noise",
        "category": "Colored Noise",
        "description": "Formula: Power spectrum: S(f) \u221d 1/f",
        "formula": "Power spectrum: S(f) \\propto 1/f",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_pink_noise(image, intensity=25):\n    \"\"\"\n    Formula: Power spectrum: S(f) \u221d 1/f\n    Generated by filtering white noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center (avoid division by zero)\n    r = np.sqrt(xx**2 + yy**2)\n    r[r == 0] = r[r > 0].min()\n    \n    # 1/f filter in frequency domain\n    noise_fft = np.fft.fft2(white_noise)\n    filt = 1 / r\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "pixelation": {
        "title": "Pixelation",
        "category": "Real-World and Environmental",
        "description": "Formula: Reduction of resolution by averaging blocks of pixels",
        "formula": "Reduction of resolution by averaging blocks of pixels",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "block_size": {
                "default": 8.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter block_size"
            }
        },
        "code_snippet": "def add_pixelation(image, block_size=8):\n    \"\"\"\n    Formula: Reduction of resolution by averaging blocks of pixels\n    \"\"\"\n    row, col = image.shape\n    \n    # Ensure block_size is at least 2\n    block_size = max(2, block_size)\n    \n    # Calculate new dimensions\n    new_row = row // block_size\n    new_col = col // block_size\n    \n    # Resize down and then up to create pixelation\n    small = cv2.resize(image, (new_col, new_row), interpolation=cv2.INTER_AREA)\n    pixelated = cv2.resize(small, (col, row), interpolation=cv2.INTER_NEAREST)\n    \n    # Calculate noise (difference due to pixelation)\n    noise = pixelated.astype(float) - image\n    \n    return pixelated, noise\n"
    },
    "poisson_noise": {
        "title": "Poisson Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = Poisson(f(x,y)/scale) * scale",
        "formula": "g(x,y) = Poisson(f(x,y)/scale) * scale",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_poisson_noise(image, scale=1.0):\n    \"\"\"\n    Formula: g(x,y) = Poisson(f(x,y)/scale) * scale\n    P(X=k) = (\u03bb^k * e^(-\u03bb)) / k!, where \u03bb = f(x,y)/scale\n    \"\"\"\n    # Ensure image values are positive\n    img_data = np.maximum(image, 0) / scale\n    noise = np.random.poisson(img_data) * scale - img_data * scale\n    noisy_img = img_data * scale + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "polarization_noise": {
        "title": "Polarization Noise",
        "category": "Environment and Special Effects",
        "description": "Formula: Artifacts from polarization effects in imaging",
        "formula": "Artifacts from polarization effects in imaging",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "strength": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter strength"
            }
        },
        "code_snippet": "def add_polarization_noise(image, strength=20):\n    \"\"\"\n    Formula: Artifacts from polarization effects in imaging\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    # Calculate angle from center\n    center_x, center_y = 0.5, 0.5\n    dx, dy = xx - center_x, yy - center_y\n    angle = np.arctan2(dy, dx)\n    \n    # Create polarization pattern\n    pol_pattern = np.cos(2 * angle) ** 2  # Malus' law pattern\n    \n    # Apply intensity variation based on polarization\n    noise = (pol_pattern - 0.5) * strength\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "print_artifacts": {
        "title": "Print Artifacts",
        "category": "Environment and Special Effects",
        "description": "Formula: Dot patterns and artifacts from printing process",
        "formula": "Dot patterns and artifacts from printing process",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "dot_size": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter dot_size"
            },
            "pattern_scale": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter pattern_scale"
            }
        },
        "code_snippet": "def add_print_artifacts(image, dot_size=2, pattern_scale=5):\n    \"\"\"\n    Formula: Dot patterns and artifacts from printing process\n    \"\"\"\n    row, col = image.shape\n    \n    # Create halftone-like pattern\n    result = np.zeros_like(image)\n    \n    for i in range(0, row, pattern_scale):\n        for j in range(0, col, pattern_scale):\n            i_end = min(i + pattern_scale, row)\n            j_end = min(j + pattern_scale, col)\n            \n            # Get average intensity in this region\n            region = image[i:i_end, j:j_end]\n            avg_intensity = np.mean(region)\n            \n            # Calculate dot size based on intensity\n            relative_dot_size = 1.0 - avg_intensity / 255\n            current_dot_size = int(dot_size * relative_dot_size * 2) + 1\n            \n            # Draw dot\n            center_i = i + pattern_scale // 2\n            center_j = j + pattern_scale // 2\n            \n            if center_i < row and center_j < col:\n                dot_i_start = max(0, center_i - current_dot_size // 2)\n                dot_i_end = min(row, center_i + current_dot_size // 2 + 1)\n                dot_j_start = max(0, center_j - current_dot_size // 2)\n                dot_j_end = min(col, center_j + current_dot_size // 2 + 1)\n                \n                result[dot_i_start:dot_i_end, dot_j_start:dot_j_end] = 0\n    \n    # Add some random ink splatters\n    splatter_mask = np.random.random((row, col)) < 0.001\n    result[splatter_mask] = 0\n    \n    # Add paper texture\n    paper_texture = np.random.normal(240, 10, (row, col))\n    texture_weight = 0.2\n    blended = (1 - texture_weight) * result + texture_weight * paper_texture\n    \n    # Calculate noise\n    noise = blended - image\n    \n    return np.clip(blended, 0, 255).astype(np.uint8), noise\n"
    },
    "quantization_noise": {
        "title": "Quantization Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Error introduced by reducing bit depth",
        "formula": "Error introduced by reducing bit depth",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "levels": {
                "default": 16.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter levels"
            }
        },
        "code_snippet": "def add_quantization_noise(image, levels=16):\n    \"\"\"\n    Formula: Error introduced by reducing bit depth\n    g(x,y) = round(f(x,y) * levels / 255) * 255 / levels\n    \"\"\"\n    # Quantize the image\n    quantized = np.round(image * (levels / 255.0)) * (255.0 / levels)\n    quantized = quantized.astype(np.uint8)\n    \n    # Calculate noise (difference between original and quantized)\n    noise = quantized.astype(float) - image\n    \n    return quantized, noise\n"
    },
    "quantum_computing_noise": {
        "title": "Quantum Computing Noise",
        "category": "Physical and Quantum",
        "description": "Formula: Models decoherence, gate errors, and readout errors in quantum systems",
        "formula": "Models decoherence, gate errors, and readout errors in quantum systems",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "decoherence": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter decoherence"
            },
            "gate_error": {
                "default": 0.05,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter gate_error"
            },
            "readout_error": {
                "default": 0.03,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter readout_error"
            }
        },
        "code_snippet": "def add_quantum_computing_noise(image, decoherence=0.1, gate_error=0.05, readout_error=0.03):\n    \"\"\"\n    Formula: Models decoherence, gate errors, and readout errors in quantum systems\n    \"\"\"\n    row, col = image.shape\n    \n    # Create binary representation\n    binary = (image > 128).astype(np.uint8) * 255\n    \n    # Decoherence noise (random bit flips with probability proportional to decoherence)\n    decoherence_mask = np.random.random((row, col)) < decoherence\n    decoherence_noise = np.zeros((row, col))\n    decoherence_noise[decoherence_mask] = 255 - 2 * binary[decoherence_mask]\n    \n    # Gate error (correlated bit flips)\n    gate_noise = np.zeros((row, col))\n    for i in range(0, row, 8):  # Process in 8x8 blocks\n        for j in range(0, col, 8):\n            if np.random.random() < gate_error:\n                i_end = min(i + 8, row)\n                j_end = min(j + 8, col)\n                gate_noise[i:i_end, j:j_end] = np.random.choice([-50, 50])\n    \n    # Readout error (measurement errors)\n    readout_mask = np.random.random((row, col)) < readout_error\n    readout_noise = np.zeros((row, col))\n    readout_noise[readout_mask] = np.random.normal(0, 50, np.sum(readout_mask))\n    \n    # Combined noise\n    noise = decoherence_noise + gate_noise + readout_noise\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "quantum_shot_noise": {
        "title": "Quantum Shot Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: Shot noise follows Poisson(\u03bb) where \u03bb is photon count",
        "formula": "Shot noise follows Poisson(\\lambda) where \\lambda is photon count",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "photons_per_pixel": {
                "default": 100.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter photons_per_pixel"
            }
        },
        "code_snippet": "def add_quantum_shot_noise(image, photons_per_pixel=100):\n    \"\"\"\n    Formula: Shot noise follows Poisson(\u03bb) where \u03bb is photon count\n    \"\"\"\n    row, col = image.shape\n    \n    # Convert image to \"photon counts\"\n    photon_scale = photons_per_pixel / 255\n    photon_counts = np.maximum(image * photon_scale, 1)  # At least 1 photon\n    \n    # Generate Poisson noise based on photon counts\n    noisy_photons = np.random.poisson(photon_counts)\n    \n    # Convert back to intensity\n    noisy_img = noisy_photons / photon_scale\n    \n    # Calculate the noise component\n    noise = noisy_img - image\n    \n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "rain_noise": {
        "title": "Rain Noise",
        "category": "Real-World and Environmental",
        "description": "Formula: Bright streaks simulating rain",
        "formula": "Bright streaks simulating rain",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "density": {
                "default": 0.01,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter density"
            },
            "length": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter length"
            },
            "brightness": {
                "default": 200.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter brightness"
            },
            "angle": {
                "default": 70.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter angle"
            }
        },
        "code_snippet": "def add_rain_noise(image, density=0.01, length=10, brightness=200, angle=70):\n    \"\"\"\n    Formula: Bright streaks simulating rain\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Convert angle to radians\n    angle_rad = np.deg2rad(angle)\n    dx = int(np.cos(angle_rad) * length)\n    dy = int(np.sin(angle_rad) * length)\n    \n    # Number of raindrops\n    num_drops = int(row * col * density)\n    \n    for _ in range(num_drops):\n        # Random raindrop position\n        x = np.random.randint(0, col)\n        y = np.random.randint(0, row)\n        \n        # Draw raindrop streak\n        for i in range(length):\n            new_x = x + int(i * np.cos(angle_rad))\n            new_y = y + int(i * np.sin(angle_rad))\n            \n            if 0 <= new_x < col and 0 <= new_y < row:\n                # Fade intensity along streak\n                intensity = brightness * (1 - i/length)\n                noise[new_y, new_x] = intensity\n    \n    # Add rain to image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "rayleigh_noise": {
        "title": "Rayleigh Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Rayleigh distribution",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Rayleigh distribution",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 35.0,
                "min": 0.0,
                "max": 140.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_rayleigh_noise(image, scale=35):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Rayleigh distribution\n    p(z) = (z/\u03c3\u00b2)*e^(-z\u00b2/(2\u03c3\u00b2)) for z \u2265 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.rayleigh.rvs(scale=scale, size=(row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "red_noise": {
        "title": "Red Noise",
        "category": "Colored Noise",
        "description": "Formula: Power spectrum: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 2",
        "formula": "Power spectrum: S(f) \\propto 1/f^\u03b1 where \u03b1 \u2248 2",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "alpha": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter alpha"
            }
        },
        "code_snippet": "def add_red_noise(image, intensity=20, alpha=2.0):\n    \"\"\"\n    Formula: Power spectrum: S(f) \u221d 1/f^\u03b1 where \u03b1 \u2248 2\n    Similar to Brown noise but with adjustable exponent\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center (avoid division by zero)\n    r = np.sqrt(xx**2 + yy**2)\n    r[r == 0] = r[r > 0].min()\n    \n    # 1/f^alpha filter\n    noise_fft = np.fft.fft2(white_noise)\n    filt = 1 / (r**alpha)\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "rician_noise": {
        "title": "Rician Noise",
        "category": "Other Noise Types",
        "description": "Formula: PDF: p(z) = (z/\u03c3\u00b2)*exp(-(z\u00b2+s\u00b2)/(2\u03c3\u00b2))*I\u2080(zs/\u03c3\u00b2) for z \u2265 0",
        "formula": "PDF: p(z) = (z/\\sigma\u00b2)*exp(-(z\u00b2+s\u00b2)/(2\\sigma\u00b2))*I\u2080(zs/\\sigma\u00b2) for z \u2265 0",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "s": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter s"
            },
            "sigma": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter sigma"
            }
        },
        "code_snippet": "def add_rician_noise(image, s=0, sigma=15):\n    \"\"\"\n    Formula: PDF: p(z) = (z/\u03c3\u00b2)*exp(-(z\u00b2+s\u00b2)/(2\u03c3\u00b2))*I\u2080(zs/\u03c3\u00b2) for z \u2265 0\n    I\u2080 is the modified Bessel function of the first kind with order zero\n    \"\"\"\n    row, col = image.shape\n    noise = rice.rvs(s, scale=sigma, size=(row, col))\n    noise = noise - np.mean(noise)  # Center around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "ridge_noise": {
        "title": "Ridge Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Ridge = 1 - |fBm|",
        "formula": "Ridge = 1 - |fBm|",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "octaves": {
                "default": 6.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_ridge_noise(image, octaves=6, intensity=40):\n    \"\"\"\n    Formula: Ridge = 1 - |fBm|\n    Modified turbulence noise with ridge formation\n    \"\"\"\n    row, col = image.shape\n    \n    # Parameters for fBm\n    H = 1.0\n    persistence = 0.5\n    lacunarity = 2.0\n    \n    noise = np.zeros((row, col))\n    amplitude = 1.0\n    frequency = 1.0\n    \n    for i in range(octaves):\n        # Generate noise at current frequency\n        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)\n        \n        # Add ridge formula to accumulated noise\n        noise += amplitude * (1.0 - np.abs(noise_layer))\n        \n        # Update parameters for next octave\n        amplitude *= persistence ** H\n        frequency *= lacunarity\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "rolling_shutter": {
        "title": "Rolling Shutter",
        "category": "Physical and Device-Based",
        "description": "Formula: Horizontal shifting of rows based on sinusoidal pattern",
        "formula": "Horizontal shifting of rows based on sinusoidal pattern",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "amplitude": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter amplitude"
            },
            "frequency": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter frequency"
            }
        },
        "code_snippet": "def add_rolling_shutter(image, amplitude=5, frequency=1):\n    \"\"\"\n    Formula: Horizontal shifting of rows based on sinusoidal pattern\n    \"\"\"\n    row, col = image.shape\n    result = np.zeros_like(image)\n    \n    # Create row-dependent shift\n    shifts = (amplitude * np.sin(2 * np.pi * frequency * np.arange(row) / row)).astype(int)\n    \n    # Apply row-wise shifts\n    for i in range(row):\n        shift = shifts[i]\n        if shift >= 0:\n            result[i, shift:] = image[i, :(col-shift)]\n        else:\n            result[i, :col+shift] = image[i, -shift:]\n    \n    # Calculate noise (difference between original and shifted)\n    noise = result.astype(float) - image\n    \n    return result, noise\n"
    },
    "row_column_noise": {
        "title": "Row Column Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Noise affecting entire rows/columns",
        "formula": "Noise affecting entire rows/columns",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "row_sigma": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter row_sigma"
            },
            "col_sigma": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter col_sigma"
            },
            "row_prob": {
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter row_prob"
            },
            "col_prob": {
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter col_prob"
            }
        },
        "code_snippet": "def add_row_column_noise(image, row_sigma=10, col_sigma=5, row_prob=0.1, col_prob=0.05):\n    \"\"\"\n    Formula: Noise affecting entire rows/columns\n    g(x,y) = f(x,y) + R(y) + C(x)\n    \"\"\"\n    row, col = image.shape\n    \n    # Initialize noise\n    noise = np.zeros((row, col))\n    \n    # Add row noise\n    row_noise = np.zeros(row)\n    row_affected = np.random.random(row) < row_prob\n    row_noise[row_affected] = np.random.normal(0, row_sigma, np.sum(row_affected))\n    noise += row_noise[:, np.newaxis]\n    \n    # Add column noise\n    col_noise = np.zeros(col)\n    col_affected = np.random.random(col) < col_prob\n    col_noise[col_affected] = np.random.normal(0, col_sigma, np.sum(col_affected))\n    noise += col_noise[np.newaxis, :]\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "salt_and_pepper_noise": {
        "title": "Salt And Pepper Noise",
        "category": "Other Noise Types",
        "description": "Formula: For each pixel (x,y):",
        "formula": "For each pixel (x,y):",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "salt_prob": {
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter salt_prob"
            },
            "pepper_prob": {
                "default": 0.05,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter pepper_prob"
            }
        },
        "code_snippet": "def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):\n    \"\"\"\n    Formula: For each pixel (x,y):\n    g(x,y) = 255 with probability p_salt\n    g(x,y) = 0 with probability p_pepper\n    g(x,y) = f(x,y) with probability 1 - p_salt - p_pepper\n    \"\"\"\n    noisy_img = np.copy(image)\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Salt noise\n    salt_mask = np.random.random(image.shape) < salt_prob\n    noisy_img[salt_mask] = 255\n    noise[salt_mask] = 1\n    \n    # Pepper noise\n    pepper_mask = np.random.random(image.shape) < pepper_prob\n    noisy_img[pepper_mask] = 0\n    noise[pepper_mask] = -1\n    \n    return noisy_img, noise * 127\n"
    },
    "scanner_artifacts": {
        "title": "Scanner Artifacts",
        "category": "Environment and Special Effects",
        "description": "Formula: Artifacts typical in document scanners",
        "formula": "Artifacts typical in document scanners",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "line_intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter line_intensity"
            },
            "dust_density": {
                "default": 0.001,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter dust_density"
            }
        },
        "code_snippet": "def add_scanner_artifacts(image, line_intensity=30, dust_density=0.001):\n    \"\"\"\n    Formula: Artifacts typical in document scanners\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros((row, col))\n    \n    # Add horizontal scan lines\n    for i in range(0, row, np.random.randint(20, 50)):\n        if np.random.random() < 0.3:  # 30% of potential scan lines are visible\n            thickness = np.random.randint(1, 3)\n            intensity = np.random.uniform(0.5, 1.5) * line_intensity\n            if i + thickness < row:\n                noise[i:i+thickness, :] = intensity\n    \n    # Add dust spots\n    dust_mask = np.random.random((row, col)) < dust_density\n    dust_sizes = np.random.randint(1, 4, size=np.sum(dust_mask))\n    \n    dust_indices = np.where(dust_mask)\n    for i, (y, x) in enumerate(zip(*dust_indices)):\n        size = dust_sizes[i]\n        y1, y2 = max(0, y-size), min(row, y+size+1)\n        x1, x2 = max(0, x-size), min(col, x+size+1)\n        \n        # Draw dust speck with random intensity\n        intensity = np.random.randint(-150, -50)\n        for dy in range(y1, y2):\n            for dx in range(x1, x2):\n                if (dy-y)**2 + (dx-x)**2 <= size**2:\n                    noise[dy, dx] = intensity\n    \n    # Add slight page curl/shadow effect\n    y, x = np.ogrid[:row, :col]\n    distance_from_edge = np.minimum(x, col-x) / col\n    shadow = -20 * np.exp(-distance_from_edge * 10) * (np.random.random() < 0.5)\n    noise += shadow\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "scratched_film": {
        "title": "Scratched Film",
        "category": "Physical and Device-Based",
        "description": "Formula: Vertical bright lines simulating film scratches",
        "formula": "Vertical bright lines simulating film scratches",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "num_scratches": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter num_scratches"
            },
            "width_range": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter width_range (default: (1, 3))"
            },
            "intensity": {
                "default": 150.0,
                "min": 0.0,
                "max": 600.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_scratched_film(image, num_scratches=20, width_range=(1, 3), intensity=150):\n    \"\"\"\n    Formula: Vertical bright lines simulating film scratches\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    \n    for _ in range(num_scratches):\n        # Random scratch position and width\n        x = np.random.randint(0, col)\n        width = np.random.randint(width_range[0], width_range[1] + 1)\n        \n        # Random scratch brightness\n        brightness = np.random.randint(intensity//2, intensity)\n        \n        # Random scratch length (partial or full length)\n        if np.random.random() < 0.3:  # 30% chance of partial scratch\n            start_y = np.random.randint(0, row//2)\n            end_y = np.random.randint(row//2, row)\n        else:\n            start_y = 0\n            end_y = row\n        \n        # Draw scratch\n        for y in range(start_y, end_y):\n            for w in range(width):\n                if 0 <= x + w < col:\n                    noise[y, x + w] = brightness\n    \n    # Apply scratches to image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "sensor_noise": {
        "title": "Sensor Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Combination of shot noise, read noise, and dark current noise",
        "formula": "Combination of shot noise, read noise, and dark current noise",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "shot_scale": {
                "default": 2.0,
                "min": 0.0,
                "max": 8.0,
                "step": 1,
                "description": "Parameter shot_scale"
            },
            "read_sigma": {
                "default": 5.0,
                "min": 0.0,
                "max": 20.0,
                "step": 1,
                "description": "Parameter read_sigma"
            },
            "dark_scale": {
                "default": 0.2,
                "min": 0.0,
                "max": 0.8,
                "step": 0.01,
                "description": "Parameter dark_scale"
            }
        },
        "code_snippet": "def add_sensor_noise(image, shot_scale=2.0, read_sigma=5, dark_scale=0.2):\n    \"\"\"\n    Formula: Combination of shot noise, read noise, and dark current noise\n    g(x,y) = Poisson(f(x,y)/s1)*s1 + N(0,\u03c3\u00b2) + exp(d)*darkness(x,y)\n    \"\"\"\n    row, col = image.shape\n    \n    # Shot noise (photon noise)\n    img_data = np.maximum(image, 0) / shot_scale\n    shot_noise = np.random.poisson(img_data) * shot_scale - img_data * shot_scale\n    \n    # Read noise (electronics)\n    read_noise = np.random.normal(0, read_sigma, (row, col))\n    \n    # Dark current noise (temperature dependent, more in shadows)\n    darkness = 1.0 - (image / 255.0)\n    dark_current = np.random.exponential(dark_scale, (row, col)) * darkness * 20\n    \n    # Combine all noise components\n    total_noise = shot_noise + read_noise + dark_current\n    noisy_img = image + total_noise\n    \n    return np.clip(noisy_img, 0, 255).astype(np.uint8), total_noise\n"
    },
    "simplex_noise": {
        "title": "Simplex Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Improved version of Perlin noise with better computational properties",
        "formula": "Improved version of Perlin noise with better computational properties",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "octaves": {
                "default": 6.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "persistence": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter persistence"
            },
            "lacunarity": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter lacunarity"
            },
            "intensity": {
                "default": 50.0,
                "min": 0.0,
                "max": 200.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_simplex_noise(image, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, intensity=50):\n    \"\"\"\n    Formula: Improved version of Perlin noise with better computational properties\n    Similar formula but different grid structure\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate approximation using filtered noise\n    freq = 1.0 / scale\n    noise = np.zeros((row, col))\n    amplitude = 1.0\n    \n    for _ in range(octaves):\n        # Generate base noise at current frequency\n        base = gaussian_filter(np.random.randn(row, col), sigma=1.0/freq)\n        \n        # Add to accumulated noise\n        noise += amplitude * base\n        \n        # Update parameters for next octave\n        amplitude *= persistence\n        freq *= lacunarity\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "smoke": {
        "title": "Smoke",
        "category": "Real-World and Environmental",
        "description": "Formula: Semi-transparent overlay with fractal noise",
        "formula": "Semi-transparent overlay with fractal noise",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 0.5,
                "min": 0.0,
                "max": 2.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_smoke(image, intensity=0.5, scale=20):\n    \"\"\"\n    Formula: Semi-transparent overlay with fractal noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate fractal noise for smoke pattern\n    noise = np.zeros((row, col))\n    octaves = 5\n    persistence = 0.5\n    lacunarity = 2.0\n    \n    for i in range(octaves):\n        frequency = lacunarity ** i\n        amplitude = persistence ** i\n        \n        # Generate basic noise at current frequency\n        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=scale/frequency)\n        noise += amplitude * noise_layer\n    \n    # Normalize and adjust contrast\n    smoke = (noise - noise.min()) / (noise.max() - noise.min())\n    smoke = np.power(smoke, 0.3)  # Increase contrast\n    \n    # Apply smoke with variable opacity\n    opacity = intensity * smoke\n    smoky = image * (1 - opacity) + 200 * opacity\n    \n    # Calculate noise effect\n    noise_effect = smoky - image\n    \n    return np.clip(smoky, 0, 255).astype(np.uint8), noise_effect\n"
    },
    "snow_noise": {
        "title": "Snow Noise",
        "category": "Real-World and Environmental",
        "description": "Formula: Small white specks simulating snow",
        "formula": "Small white specks simulating snow",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "density": {
                "default": 0.01,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter density"
            },
            "size_range": {
                "default": 2.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter size_range (default: (1, 3))"
            },
            "brightness": {
                "default": 200.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter brightness"
            }
        },
        "code_snippet": "def add_snow_noise(image, density=0.01, size_range=(1, 3), brightness=200):\n    \"\"\"\n    Formula: Small white specks simulating snow\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    \n    # Number of snowflakes\n    num_flakes = int(row * col * density)\n    \n    for _ in range(num_flakes):\n        # Random snowflake position\n        x = np.random.randint(0, col)\n        y = np.random.randint(0, row)\n        \n        # Random snowflake size\n        size = np.random.randint(size_range[0], size_range[1] + 1)\n        \n        # Draw snowflake (small white disk)\n        for dy in range(-size, size + 1):\n            for dx in range(-size, size + 1):\n                if dx**2 + dy**2 <= size**2:\n                    new_x, new_y = x + dx, y + dy\n                    if 0 <= new_x < col and 0 <= new_y < row:\n                        noise[new_y, new_x] = brightness\n    \n    # Add snow to image\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "sparse_convolution_noise": {
        "title": "Sparse Convolution Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Convolution of sparse impulses with a kernel function",
        "formula": "Convolution of sparse impulses with a kernel function",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter intensity"
            },
            "density": {
                "default": 0.05,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter density"
            },
            "kernel_size": {
                "default": 15.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter kernel_size"
            }
        },
        "code_snippet": "def add_sparse_convolution_noise(image, intensity=30, density=0.05, kernel_size=15):\n    \"\"\"\n    Formula: Convolution of sparse impulses with a kernel function\n    \"\"\"\n    row, col = image.shape\n    \n    # Create sparse impulse image\n    impulses = np.zeros((row, col))\n    mask = np.random.random((row, col)) < density\n    impulses[mask] = np.random.normal(0, 1, size=np.sum(mask))\n    \n    # Create kernel\n    x = np.linspace(-1, 1, kernel_size)\n    y = np.linspace(-1, 1, kernel_size)\n    xx, yy = np.meshgrid(x, y)\n    r = np.sqrt(xx**2 + yy**2)\n    kernel = np.exp(-4 * r**2)\n    kernel /= kernel.sum()\n    \n    # Convolve impulses with kernel\n    noise = ndimage.convolve(impulses, kernel, mode='reflect')\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "speckle_noise": {
        "title": "Speckle Noise",
        "category": "Other Noise Types",
        "description": "Formula: g(x,y) = f(x,y) + f(x,y)*\u03b7(x,y), where \u03b7 follows N(0, var)",
        "formula": "g(x,y) = f(x,y) + f(x,y)*\\eta(x,y), where \\eta follows N(0, var)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "var": {
                "default": 0.1,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter var"
            }
        },
        "code_snippet": "def add_speckle_noise(image, var=0.1):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + f(x,y)*\u03b7(x,y), where \u03b7 follows N(0, var)\n    \"\"\"\n    row, col = image.shape\n    noise = np.random.normal(0, np.sqrt(var), (row, col))\n    noisy_img = image + image * noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise * 50  # Scale for visualization\n"
    },
    "spectroscopic_noise": {
        "title": "Spectroscopic Noise",
        "category": "Specialized Scientific Fields",
        "description": "Formula: Combination of shot noise, baseline drift, and cosmic ray spikes",
        "formula": "Combination of shot noise, baseline drift, and cosmic ray spikes",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "shot_scale": {
                "default": 1.0,
                "min": 0.0,
                "max": 4.0,
                "step": 1,
                "description": "Parameter shot_scale"
            },
            "baseline_drift": {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter baseline_drift"
            },
            "cosmic_ray_prob": {
                "default": 0.01,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "description": "Parameter cosmic_ray_prob"
            }
        },
        "code_snippet": "def add_spectroscopic_noise(image, shot_scale=1.0, baseline_drift=True, cosmic_ray_prob=0.01):\n    \"\"\"\n    Formula: Combination of shot noise, baseline drift, and cosmic ray spikes\n    \"\"\"\n    row, col = image.shape\n    \n    # Shot noise component (Poisson)\n    img_data = np.maximum(image, 0) / shot_scale\n    shot_noise = np.random.poisson(img_data) * shot_scale - img_data * shot_scale\n    \n    # Baseline drift (slow variation across columns)\n    if baseline_drift:\n        t = np.linspace(0, 6*np.pi, col)\n        drift = 10 * np.sin(t/3) + 5 * np.sin(t/10)\n        baseline = np.tile(drift, (row, 1))\n    else:\n        baseline = np.zeros((row, col))\n    \n    # Cosmic ray spikes (very sharp, intense peaks)\n    cosmic = np.zeros((row, col))\n    if cosmic_ray_prob > 0:\n        for i in range(int(row * col * cosmic_ray_prob)):\n            x = np.random.randint(0, col)\n            y = np.random.randint(0, row)\n            intensity = np.random.uniform(50, 200)\n            width = np.random.randint(1, 3)\n            \n            for dx in range(-width, width+1):\n                for dy in range(-width, width+1):\n                    nx, ny = x+dx, y+dy\n                    if 0 <= nx < col and 0 <= ny < row:\n                        dist = np.sqrt(dx**2 + dy**2)\n                        if dist <= width:\n                            cosmic[ny, nx] = intensity * (1 - dist/width)\n    \n    # Combined noise\n    noise = shot_noise + baseline + cosmic\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "structured_correlated_noise": {
        "title": "Structured Correlated Noise",
        "category": "Colored Noise",
        "description": "Formula: Noise with specific spatial correlation structure",
        "formula": "Noise with specific spatial correlation structure",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "correlation_length": {
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter correlation_length"
            }
        },
        "code_snippet": "def add_structured_correlated_noise(image, scale=20, correlation_length=10):\n    \"\"\"\n    Formula: Noise with specific spatial correlation structure\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate base white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Apply specific correlation structure\n    correlated_noise = gaussian_filter(white_noise, sigma=correlation_length)\n    \n    # Add structure by multiplying with pattern\n    x = np.linspace(0, 10, col)\n    y = np.linspace(0, 10, row)\n    xx, yy = np.meshgrid(x, y)\n    \n    pattern = np.sin(xx) * np.cos(yy) * 0.5 + 0.5\n    \n    # Apply pattern to correlated noise\n    structured_noise = correlated_noise * pattern * scale\n    \n    noisy_img = image + structured_noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), structured_noise\n"
    },
    "t_noise": {
        "title": "T Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Student's t(\u03bd)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Student's t(\u03bd)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "df": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter df"
            },
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_t_noise(image, df=3, scale=20):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Student's t(\u03bd)\n    p(z) = \u0393((\u03bd+1)/2)/(\u0393(\u03bd/2)*sqrt(\u03bd\u03c0)*(1+z\u00b2/\u03bd)^((\u03bd+1)/2))\n    \"\"\"\n    row, col = image.shape\n    noise = stats.t.rvs(df, size=(row, col)) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "thermal_noise": {
        "title": "Thermal Noise",
        "category": "Physical and Device-Based",
        "description": "Formula: Temperature-dependent Gaussian noise",
        "formula": "Temperature-dependent Gaussian noise",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "mean": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter mean"
            },
            "sigma": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter sigma"
            },
            "temp_factor": {
                "default": 0.8,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter temp_factor"
            }
        },
        "code_snippet": "def add_thermal_noise(image, mean=0, sigma=15, temp_factor=0.8):\n    \"\"\"\n    Formula: Temperature-dependent Gaussian noise\n    g(x,y) = f(x,y) + N(\u03bc,\u03c3\u00b2*T)\n    \"\"\"\n    row, col = image.shape\n    \n    # Create temperature gradient (hotter at bottom, for example)\n    temp_gradient = np.linspace(0.5, 1.0, row)[:, np.newaxis] * temp_factor\n    \n    # Adjust noise variance based on temperature\n    local_sigma = sigma * temp_gradient\n    \n    # Generate noise\n    noise = np.zeros((row, col))\n    for i in range(row):\n        noise[i, :] = np.random.normal(mean, local_sigma[i], col)\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "turbulence_noise": {
        "title": "Turbulence Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Absolute value of fBm",
        "formula": "Absolute value of fBm",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "octaves": {
                "default": 6.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_turbulence_noise(image, octaves=6, intensity=40):\n    \"\"\"\n    Formula: Absolute value of fBm\n    T(x,y) = |fBm(x,y)|\n    \"\"\"\n    row, col = image.shape\n    \n    # Parameters for fBm\n    H = 0.5\n    persistence = 0.5\n    lacunarity = 2.0\n    \n    noise = np.zeros((row, col))\n    amplitude = 1.0\n    frequency = 1.0\n    \n    for i in range(octaves):\n        # Generate noise at current frequency\n        noise_layer = gaussian_filter(np.random.randn(row, col), sigma=1.0/frequency)\n        \n        # Add absolute value to accumulated noise (turbulence formula)\n        noise += amplitude * np.abs(noise_layer)\n        \n        # Update parameters for next octave\n        amplitude *= persistence ** H\n        frequency *= lacunarity\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "uniform_noise": {
        "title": "Uniform Noise",
        "category": "Other Noise Types",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows U(low, high)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows U(low, high)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "low": {
                "default": -50.0,
                "min": -100.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter low"
            },
            "high": {
                "default": 50.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter high"
            }
        },
        "code_snippet": "def add_uniform_noise(image, low=-50, high=50):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows U(low, high)\n    p(\u03b7) = 1/(high-low) for low \u2264 \u03b7 \u2264 high\n    \"\"\"\n    row, col = image.shape\n    noise = np.random.uniform(low, high, (row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "value_noise": {
        "title": "Value Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Interpolation between random values at grid vertices",
        "formula": "Interpolation between random values at grid vertices",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "octaves": {
                "default": 4.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter octaves"
            },
            "persistence": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter persistence"
            },
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            },
            "intensity": {
                "default": 40.0,
                "min": 0.0,
                "max": 160.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_value_noise(image, octaves=4, persistence=0.5, scale=20, intensity=40):\n    \"\"\"\n    Formula: Interpolation between random values at grid vertices\n    Similar to Perlin but using value interpolation instead of gradient interpolation\n    \"\"\"\n    row, col = image.shape\n    \n    noise = np.zeros((row, col))\n    amplitude = 1.0\n    frequency = 1.0\n    \n    for _ in range(octaves):\n        # Generate random values at grid points\n        grid_size = int(scale / frequency)\n        if grid_size < 2:\n            grid_size = 2\n            \n        # Create grid of random values\n        grid = np.random.rand(grid_size, grid_size)\n        \n        # Resize to image dimensions using bilinear interpolation\n        layer = cv2.resize(grid, (col, row), interpolation=cv2.INTER_LINEAR)\n        \n        # Add to noise with current amplitude\n        noise += layer * amplitude\n        \n        # Update for next octave\n        amplitude *= persistence\n        frequency *= 2.0\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "vignetting": {
        "title": "Vignetting",
        "category": "Physical and Device-Based",
        "description": "Formula: Darkening around the edges of the image",
        "formula": "Darkening around the edges of the image",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "strength": {
                "default": 0.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter strength"
            }
        },
        "code_snippet": "def add_vignetting(image, strength=0.5):\n    \"\"\"\n    Formula: Darkening around the edges of the image\n    g(x,y) = f(x,y) * V(x,y) where V decreases with distance from center\n    \"\"\"\n    row, col = image.shape\n    \n    # Create coordinate grid\n    y, x = np.ogrid[:row, :col]\n    \n    # Calculate center\n    center_y, center_x = row / 2, col / 2\n    \n    # Calculate squared distance from center\n    dist_squared = (x - center_x)**2 + (y - center_y)**2\n    \n    # Normalize distance to [0, 1] range\n    max_dist_squared = (max(center_x, center_y))**2\n    normalized_dist = dist_squared / max_dist_squared\n    \n    # Create vignette mask\n    vignette = 1 - strength * normalized_dist\n    \n    # Apply vignette\n    vignetted = image * vignette\n    \n    # Calculate \"noise\" (difference due to vignetting)\n    noise = vignetted - image\n    \n    return vignetted.astype(np.uint8), noise\n"
    },
    "violet_noise": {
        "title": "Violet Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: Power spectrum: S(f) \u221d f\u00b2",
        "formula": "Power spectrum: S(f) \\propto f\u00b2",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 10.0,
                "min": 0.0,
                "max": 40.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_violet_noise(image, intensity=10):\n    \"\"\"\n    Formula: Power spectrum: S(f) \u221d f\u00b2\n    Generated by double-differentiation of white noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate 2D white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Create frequency domain filter\n    x = np.linspace(0, 1, col)\n    y = np.linspace(0, 1, row)\n    xx, yy = np.meshgrid(x, y)\n    xx = xx - 0.5\n    yy = yy - 0.5\n    \n    # Distance from center\n    r = np.sqrt(xx**2 + yy**2)\n    \n    # Violet noise filter (emphasize very high frequencies)\n    noise_fft = np.fft.fft2(white_noise)\n    filt = r**2\n    noise_fft_filtered = np.fft.fftshift(noise_fft) * filt\n    noise_fft_filtered = np.fft.ifftshift(noise_fft_filtered)\n    \n    # Inverse FFT\n    noise = np.real(np.fft.ifft2(noise_fft_filtered))\n    \n    # Normalize and apply intensity\n    noise = intensity * (noise - noise.min()) / (noise.max() - noise.min()) - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "vonmises_noise": {
        "title": "Vonmises Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: PDF: p(z) = [e^(\u03ba*cos(z-\u03bc))] / [2\u03c0*I\u2080(\u03ba)]",
        "formula": "PDF: p(z) = [e^(\u03ba*cos(z-\\mu))] / [2\u03c0*I\u2080(\u03ba)]",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "kappa": {
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter kappa"
            },
            "loc": {
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter loc"
            },
            "scale": {
                "default": 15.0,
                "min": 0.0,
                "max": 60.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_vonmises_noise(image, kappa=3, loc=0, scale=15):\n    \"\"\"\n    Formula: PDF: p(z) = [e^(\u03ba*cos(z-\u03bc))] / [2\u03c0*I\u2080(\u03ba)]\n    \"\"\"\n    row, col = image.shape\n    noise = vonmises.rvs(kappa, loc=loc, size=(row, col)) * scale\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "water_droplets": {
        "title": "Water Droplets",
        "category": "Real-World and Environmental",
        "description": "Formula: Refraction-like distortion in circular regions",
        "formula": "Refraction-like distortion in circular regions",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "num_droplets": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter num_droplets"
            },
            "min_size": {
                "default": 5.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter min_size"
            },
            "max_size": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter max_size"
            }
        },
        "code_snippet": "def add_water_droplets(image, num_droplets=20, min_size=5, max_size=20):\n    \"\"\"\n    Formula: Refraction-like distortion in circular regions\n    \"\"\"\n    row, col = image.shape\n    noise = np.zeros_like(image, dtype=float)\n    result = image.copy()\n    \n    for _ in range(num_droplets):\n        # Random droplet position and size\n        cx = np.random.randint(0, col)\n        cy = np.random.randint(0, row)\n        radius = np.random.randint(min_size, max_size)\n        \n        # Create a coordinate grid centered on the droplet\n        y, x = np.ogrid[:row, :col]\n        \n        # Create a circular mask for the droplet\n        mask = ((x - cx)**2 + (y - cy)**2) <= radius**2\n        \n        # Skip if droplet is too close to the edge (no pixels inside the mask)\n        if not np.any(mask):\n            continue\n        \n        # Create spherical distortion map\n        dx = x - cx\n        dy = y - cy\n        \n        # Calculate distance from center (avoid division by zero)\n        dist = np.sqrt((x - cx)**2 + (y - cy)**2)\n        dist[dist == 0] = 1  # Avoid division by zero\n        \n        # Calculate distortion factor (simulating lens effect)\n        factor = 0.5 * radius / dist\n        factor[~mask] = 0  # Only apply within the droplet\n        \n        # Calculate new coordinates\n        new_x = x - dx * factor\n        new_y = y - dy * factor\n        \n        # Ensure coordinates are within bounds\n        new_x = np.clip(new_x, 0, col-1).astype(int)\n        new_y = np.clip(new_y, 0, row-1).astype(int)\n        \n        # Apply distortion only within droplet\n        for i in range(row):\n            for j in range(col):\n                if mask[i, j]:\n                    result[i, j] = image[new_y[i, j], new_x[i, j]]\n        \n        # Calculate noise for visualization\n        noise[mask] = result[mask] - image[mask]\n        \n        # Add highlight to droplet edge\n        edge_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2) & ((x - cx)**2 + (y - cy)**2 >= (radius*0.9)**2)\n        result[edge_mask] = np.minimum(result[edge_mask] + 50, 255)\n        noise[edge_mask] += 50\n    \n    return result.astype(np.uint8), noise\n"
    },
    "wavelet_noise": {
        "title": "Wavelet Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: Based on wavelet decomposition and reconstruction",
        "formula": "Based on wavelet decomposition and reconstruction",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 30.0,
                "min": 0.0,
                "max": 120.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_wavelet_noise(image, intensity=30):\n    \"\"\"\n    Formula: Based on wavelet decomposition and reconstruction\n    Uses wavelet transforms to generate band-limited noise\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate white noise\n    white_noise = np.random.normal(0, 1, (row, col))\n    \n    # Perform wavelet transform (using 'db4' wavelet, level 3)\n    coeffs = pywt.wavedec2(white_noise, 'db4', level=3)\n    \n    # Modify coefficients (without directly modifying the tuple)\n    # Create a new list of coefficients\n    new_coeffs = [coeffs[0]]  # Keep approximation coefficients\n    \n    # Modify detail coefficients by creating new tuples\n    for i in range(1, len(coeffs)):\n        detail_coeffs = coeffs[i]\n        new_detail = (detail_coeffs[0] * 0.8, \n                     detail_coeffs[1] * 0.8, \n                     detail_coeffs[2] * 0.8)\n        new_coeffs.append(new_detail)\n    \n    # Reconstruct signal\n    noise = pywt.waverec2(new_coeffs, 'db4')\n    \n    # Crop to original size if dimensions changed\n    noise = noise[:row, :col]\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * noise - intensity/2\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "weibull_noise": {
        "title": "Weibull Noise",
        "category": "Statistical Distribution-Based",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Weibull(\u03bb,k)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows Weibull(\\lambda,k)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "a": {
                "default": 1.5,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter a"
            },
            "scale": {
                "default": 20.0,
                "min": 0.0,
                "max": 80.0,
                "step": 1,
                "description": "Parameter scale"
            }
        },
        "code_snippet": "def add_weibull_noise(image, a=1.5, scale=20):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows Weibull(\u03bb,k)\n    p(z) = (k/\u03bb)*(z/\u03bb)^(k-1)*e^(-(z/\u03bb)^k) for z \u2265 0\n    \"\"\"\n    row, col = image.shape\n    noise = stats.weibull_min.rvs(a, scale=scale, size=(row, col))\n    noise -= np.mean(noise)  # Center the noise around zero\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "white_noise": {
        "title": "White Noise",
        "category": "Colored Noise",
        "description": "Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows N(0,\u03c3\u00b2)",
        "formula": "g(x,y) = f(x,y) + \\eta(x,y), where \\eta follows N(0,\\sigma\u00b2)",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "intensity": {
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_white_noise(image, intensity=25):\n    \"\"\"\n    Formula: g(x,y) = f(x,y) + \u03b7(x,y), where \u03b7 follows N(0,\u03c3\u00b2)\n    Spectral density: S(f) = constant\n    \"\"\"\n    row, col = image.shape\n    noise = np.random.normal(0, intensity, (row, col))\n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    },
    "worley_noise": {
        "title": "Worley Noise",
        "category": "Procedural and Synthetic",
        "description": "Formula: Based on distance to nearest feature point",
        "formula": "Based on distance to nearest feature point",
        "formula_explanation": "This noise affects the image according to the formula shown above.",
        "parameters": {
            "n_points": {
                "default": 20.0,
                "min": 0.0,
                "max": 100.0,
                "step": 1,
                "description": "Parameter n_points"
            },
            "intensity": {
                "default": 50.0,
                "min": 0.0,
                "max": 200.0,
                "step": 1,
                "description": "Parameter intensity"
            }
        },
        "code_snippet": "def add_worley_noise(image, n_points=20, intensity=50):\n    \"\"\"\n    Formula: Based on distance to nearest feature point\n    W(x,y) = min(dist((x,y), (x_i,y_i))) for all feature points i\n    \"\"\"\n    row, col = image.shape\n    \n    # Generate random feature points\n    points = np.random.rand(n_points, 2)\n    points[:, 0] *= row\n    points[:, 1] *= col\n    \n    # Calculate distance to nearest feature point for each pixel\n    xx, yy = np.meshgrid(np.arange(col), np.arange(row))\n    noise = np.ones((row, col)) * np.inf\n    \n    for p in points:\n        dist = np.sqrt((yy - p[0])**2 + (xx - p[1])**2)\n        noise = np.minimum(noise, dist)\n    \n    # Normalize and scale\n    noise = (noise - noise.min()) / (noise.max() - noise.min())\n    noise = intensity * (1 - noise) - intensity/2  # Invert so cells are darker\n    \n    noisy_img = image + noise\n    return np.clip(noisy_img, 0, 255).astype(np.uint8), noise\n"
    }
}