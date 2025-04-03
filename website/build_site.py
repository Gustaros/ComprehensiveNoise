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

# Load metadata from the JSON file instead of using the hardcoded default
metadata_file = os.path.join(SITE_DIR, 'noise_metadata.json')
if os.path.exists(metadata_file):
    with open(metadata_file, 'r', encoding='utf-8') as f:
        NOISE_METADATA = json.load(f)
    print(f"Loaded metadata for {len(NOISE_METADATA)} noise types from {metadata_file}")
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
    print("Using default metadata (Gaussian noise only)")

# Function to get noise function by name
def get_noise_function(noise_name):
    # Convert noise_name to function name
    function_name = f"add_{noise_name}"
    if hasattr(noises, function_name):
        return getattr(noises, function_name)
    else:
        print(f"Warning: Function {function_name} not found in noises module")
        return None

# Generate sample images and noise visualizations
def generate_noise_samples(noise_name, params=None):
    # Default sample image
    sample_image = noises.create_sample_image(pattern="gradient")
    
    noise_func = get_noise_function(noise_name)
    if noise_func:
        if params:
            # Apply noise with specified parameters
            noisy_image, noise_pattern = noise_func(sample_image, **params)
        else:
            # Use default parameters
            noisy_image, noise_pattern = noise_func(sample_image)
            
        # Create the visualization
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(2, 3, height_ratios=[2, 1])
        
        # Original image
        ax_orig = plt.subplot(grid[0, 0])
        ax_orig.imshow(sample_image, cmap='gray')
        ax_orig.set_title('Original Image')
        ax_orig.axis('off')
        
        # Noisy image
        ax_noisy = plt.subplot(grid[0, 1])
        ax_noisy.imshow(np.clip(noisy_image, 0, 255), cmap='gray')
        ax_noisy.set_title(f'Image with Noise')
        ax_noisy.axis('off')
        
        # Noise pattern
        ax_noise = plt.subplot(grid[0, 2])
        im = ax_noise.imshow(noise_pattern, cmap='viridis')
        ax_noise.set_title('Noise Pattern')
        ax_noise.axis('off')
        
        # Add colorbar
        divider = make_axes_locatable(ax_noise)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Noise histogram
        ax_hist = plt.subplot(grid[1, :])
        flat_noise = noise_pattern.flatten()
        
        # For sparse noise, filter out zeros
        if np.count_nonzero(flat_noise) / flat_noise.size < 0.1:
            flat_noise = flat_noise[flat_noise != 0]
            if len(flat_noise) == 0:
                flat_noise = noise_pattern.flatten()
        
        # Remove outliers for better visualization
        if len(flat_noise) > 0:
            q1, q3 = np.percentile(flat_noise, [1, 99])
            bin_data = flat_noise[(flat_noise >= q1) & (flat_noise <= q3)]
            
            if len(bin_data) > 10:
                ax_hist.hist(bin_data, bins=50, alpha=0.7, color='steelblue', 
                           edgecolor='black', density=True)
                
                # Add normal distribution fit if appropriate
                try:
                    mu, std = norm.fit(bin_data)
                    x = np.linspace(min(bin_data), max(bin_data), 100)
                    p = norm.pdf(x, mu, std)
                    ax_hist.plot(x, p, 'r-', linewidth=2)
                    
                    # Add statistics
                    kurt = kurtosis(bin_data)
                    sk = skew(bin_data)
                    stats_text = f"μ = {mu:.2f}, σ = {std:.2f}\nSkewness = {sk:.2f}, Kurtosis = {kurt:.2f}"
                    ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes,
                                verticalalignment='top', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                except:
                    pass
            else:
                ax_hist.text(0.5, 0.5, "Insufficient data for histogram", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax_hist.transAxes)
        
        ax_hist.set_title('Noise Distribution Histogram')
        ax_hist.set_xlabel('Noise Value')
        ax_hist.set_ylabel('Probability Density')
        ax_hist.grid(True, alpha=0.3)
        
        plt.suptitle(f"Noise Type: {noise_name.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        # Save as file and return base64 for embedding
        img_path = os.path.join(OUTPUT_DIR, "assets", "images", f"{noise_name}_sample.png")
        with open(img_path, 'wb') as f:
            f.write(buf.read())
        
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_data}"
    
    return None

# Read HTML template
def read_template(template_name):
    with open(os.path.join(TEMPLATES_DIR, f"{template_name}.html"), 'r', encoding='utf-8') as f:
        return f.read()

# Generate individual noise page
def generate_noise_page(noise_name, metadata):
    template = read_template('noise_page')
    
    # Generate images
    sample_image_data = generate_noise_samples(noise_name)
    
    # Replace template placeholders
    page_content = template.replace('{{NOISE_TITLE}}', metadata['title'])
    page_content = page_content.replace('{{NOISE_CATEGORY}}', metadata['category'])
    page_content = page_content.replace('{{NOISE_DESCRIPTION}}', metadata['description'])
    page_content = page_content.replace('{{NOISE_FORMULA}}', metadata['formula'])
    page_content = page_content.replace('{{NOISE_FORMULA_EXPLANATION}}', metadata['formula_explanation'])
    page_content = page_content.replace('{{SAMPLE_IMAGE}}', sample_image_data or '')
    page_content = page_content.replace('{{CODE_SNIPPET}}', metadata['code_snippet'])
    
    # Generate parameter controls
    param_controls = ""
    if 'parameters' in metadata:
        for param_name, param_data in metadata['parameters'].items():
            control = f"""
            <div class="parameter-control">
                <label for="{param_name}">{param_data['description']} ({param_name}):</label>
                <input type="range" id="{param_name}" 
                       min="{param_data['min']}" max="{param_data['max']}" 
                       step="{param_data['step']}" value="{param_data['default']}">
                <span class="parameter-value">{param_data['default']}</span>
            </div>
            """
            param_controls += control
    
    page_content = page_content.replace('{{PARAMETER_CONTROLS}}', param_controls)
    
    # Generate parameter JSON for JavaScript
    params_json = json.dumps(metadata.get('parameters', {}))
    page_content = page_content.replace('{{PARAMETERS_JSON}}', params_json)
    page_content = page_content.replace('{{NOISE_NAME}}', noise_name)
    
    # Write the page
    output_path = os.path.join(NOISE_PAGES_DIR, f"{noise_name}.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(page_content)
    
    return metadata['title'], metadata['category']

# Generate category pages
def generate_category_pages(noise_by_category):
    template = read_template('category_page')
    
    for category, noises in noise_by_category.items():
        page_content = template.replace('{{CATEGORY_TITLE}}', category)
        
        # Generate noise list
        noise_list = ""
        for noise_name, noise_title in noises:
            noise_list += f"""
            <div class="noise-card">
                <h3>{noise_title}</h3>
                <a href="../noise_types/{noise_name}.html">
                    <img src="../assets/images/{noise_name}_sample.png" alt="{noise_title}">
                </a>
                <a href="../noise_types/{noise_name}.html" class="explore-btn">Explore</a>
            </div>
            """
        
        page_content = page_content.replace('{{NOISE_LIST}}', noise_list)
        
        # Write the category page
        slug = re.sub(r'[^a-zA-Z0-9]', '_', category.lower())
        output_path = os.path.join(CATEGORY_PAGES_DIR, f"{slug}.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(page_content)

# Generate index page
def generate_index_page(categories):
    template = read_template('index')
    
    # Generate category list
    category_list = ""
    for category, _ in categories.items():
        slug = re.sub(r'[^a-zA-Z0-9]', '_', category.lower())
        category_list += f"""
        <div class="category-card">
            <h2>{category}</h2>
            <a href="categories/{slug}.html" class="explore-btn">Explore Category</a>
        </div>
        """
    
    page_content = template.replace('{{CATEGORY_LIST}}', category_list)
    
    # Write the index page
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(page_content)

# Copy assets
def copy_assets():
    # Copy CSS
    css_src = os.path.join(ASSETS_DIR, "css")
    css_dst = os.path.join(OUTPUT_DIR, "assets", "css")
    for file in os.listdir(css_src):
        shutil.copy(os.path.join(css_src, file), os.path.join(css_dst, file))
    
    # Copy JS
    js_src = os.path.join(ASSETS_DIR, "js")
    js_dst = os.path.join(OUTPUT_DIR, "assets", "js")
    for file in os.listdir(js_src):
        shutil.copy(os.path.join(js_src, file), os.path.join(js_dst, file))

# Main function to build the site
def build_site():
    print("Building static site...")
    
    # Process each noise type
    noise_by_category = {}
    for noise_name, metadata in NOISE_METADATA.items():
        print(f"Processing {metadata['title']}...")
        title, category = generate_noise_page(noise_name, metadata)
        
        if category not in noise_by_category:
            noise_by_category[category] = []
        noise_by_category[category].append((noise_name, title))
    
    # Generate category pages
    print("Generating category pages...")
    generate_category_pages(noise_by_category)
    
    # Generate index page
    print("Generating index page...")
    generate_index_page(noise_by_category)
    
    # Copy assets
    print("Copying assets...")
    copy_assets()
    
    # Print summary
    total_files = 0
    if os.path.exists(OUTPUT_DIR):
        for root, dirs, files in os.walk(OUTPUT_DIR):
            total_files += len(files)
    
    print(f"Build complete. Generated {total_files} files.")
    print(f"Main page: {os.path.join(OUTPUT_DIR, 'index.html')}")
    
    # Verify key files exist
    key_files = ['index.html', os.path.join('assets', 'css', 'styles.css')]
    missing = [f for f in key_files if not os.path.exists(os.path.join(OUTPUT_DIR, f))]
    
    if missing:
        print("WARNING: Some key files are missing:")
        for f in missing:
            print(f"  - {f}")
    else:
        print("All key files are present.")

if __name__ == "__main__":
    build_site()
