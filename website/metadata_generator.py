import os
import sys
import re
import json
import inspect

# Add parent directory to path to import noise library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noises

def extract_docstring_info(docstring):
    """Extract description and formula from function docstring."""
    if not docstring:
        return {
            "description": "No description available.",
            "formula": "No formula available.",
            "formula_explanation": "No explanation available."
        }
    
    # Extract formula
    formula_match = re.search(r'Formula:\s*(.*?)(?=\n|$)', docstring, re.DOTALL)
    formula = formula_match.group(1).strip() if formula_match else "No formula available."
    
    # Clean up formula for LaTeX
    formula = formula.replace('η', r'\eta')
    formula = formula.replace('σ', r'\sigma')
    formula = formula.replace('μ', r'\mu')
    formula = formula.replace('λ', r'\lambda')
    formula = formula.replace('∝', r'\propto')
    
    # Extract description (first paragraph of docstring)
    desc_match = re.match(r'(.*?)(?=\n|$)', docstring.strip(), re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else "No description available."
    
    # Create explanation
    explanation = "This noise affects the image according to the formula shown above."
    
    return {
        "description": description,
        "formula": formula,
        "formula_explanation": explanation
    }

def extract_default_params(func):
    """Extract default parameters from function signature."""
    signature = inspect.signature(func)
    params = {}
    
    for name, param in signature.parameters.items():
        # Skip the first parameter (usually 'image')
        if name == 'image':
            continue
            
        if param.default is not param.empty:
            # Get default value
            default_value = param.default
            
            # Handle different types of default values
            try:
                # Check if it's a tuple or other complex type
                if isinstance(default_value, (tuple, list)):
                    # For tuples/lists, use the average or first element
                    if len(default_value) > 0:
                        if all(isinstance(x, (int, float)) for x in default_value):
                            # If all elements are numeric, use average
                            default_numeric = sum(default_value) / len(default_value)
                        else:
                            # Otherwise use first element if possible
                            try:
                                default_numeric = float(default_value[0])
                            except (ValueError, TypeError):
                                default_numeric = 1.0
                    else:
                        default_numeric = 1.0
                    
                    # Use a reasonable range
                    min_val = 0
                    max_val = 100
                    
                    # Special handling for tuple parameters
                    if 'size' in name or 'range' in name:
                        # For size or range parameters, show the original tuple as description
                        param_desc = f"Parameter {name} (default: {default_value})"
                    else:
                        param_desc = f"Parameter {name}"
                    
                    params[name] = {
                        "default": default_numeric,
                        "min": float(min_val),
                        "max": float(max_val),
                        "step": 1,
                        "description": param_desc
                    }
                    continue
                
                # Try to convert to float if it's a string
                if isinstance(default_value, str):
                    default_value = float(default_value)
                
                # Now we should have a numeric value we can compare
                if isinstance(default_value, (int, float)):
                    if 'sigma' in name or 'scale' in name or 'intensity' in name:
                        min_val = 0
                        max_val = default_value * 4 if default_value > 0 else 100
                    elif 'prob' in name:
                        min_val = 0
                        max_val = 1
                        default_value = float(default_value)  # Ensure it's a float
                    else:
                        # Generic range
                        min_val = -100 if default_value < 0 else 0
                        max_val = 100
                    
                    # Create parameter entry
                    params[name] = {
                        "default": float(default_value),
                        "min": float(min_val),
                        "max": float(max_val),
                        "step": 0.01 if 0 <= max_val <= 1 else 1,
                        "description": f"Parameter {name}"
                    }
                else:
                    # For non-numeric types, skip this parameter
                    print(f"Skipping non-numeric parameter {name} with value {default_value}")
            except (ValueError, TypeError) as e:
                # Skip parameters that can't be converted or compared
                print(f"Error processing parameter {name}: {e}")
    
    return params

def categorize_noise_function(func_name):
    """Categorize noise function based on its name."""
    categories = {
        "gaussian|normal|poisson|rayleigh|gamma|exponential|laplacian|cauchy|chi2|beta|weibull|logistic|t_noise|f_noise|lognormal|binomial|nbinom|hypergeom|pareto|maxwell|rice|vonmises|gumbel|levy|nakagami": "Statistical Distribution-Based",
        "white|pink|brown|blue|violet|grey|red|orange|green|black": "Colored Noise",
        "perlin|simplex|worley|fbm|value|wavelet|diamond|gabor|sparse|turbulence|mosaic|ridge|curl|caustic|cracks|flow": "Procedural and Synthetic",
        "film|sensor|thermal|fixed|row|column|banding|jpeg|quantization|demosaic|blur|vignett|bloom|chromatic|hot|dead|rolling|moire": "Physical and Device-Based",
        "rain|snow|fog|dust|flare|water|smoke|scratch|glitch|pixel": "Real-World and Environmental",
        "johnson|flicker|quantum|avalanche|gr_noise": "Physical and Quantum",
        "k_dist|photon|phase|cosmic|jitter": "Specialized Fields",
        "levy_stable|chaotic|brownian|multifractal": "Mathematical and Theoretical",
        "compression|hdr|dl_artifact": "New Technology",
        "nonstat|hetero|aniso|structure": "Combinational and Specific",
        "atmospher|electrical|scanner|print|polarization": "Environment and Special Effects",
        "spectro|gravit|astronom|neuro|quantum_comp": "Specialized Scientific Fields",
        "holo|diffract|mixed|machine|hdr_spec": "Newest and Future Directions"
    }
    
    for pattern, category in categories.items():
        if re.search(pattern, func_name, re.IGNORECASE):
            return category
    
    return "Other Noise Types"

def generate_noise_metadata():
    """Generate metadata for all noise functions in the noises module."""
    metadata = {}
    
    # Find all noise functions
    for name, func in inspect.getmembers(noises, inspect.isfunction):
        if name.startswith('add_') and name != 'add_noise':
            # Extract noise name
            noise_name = name[4:]  # Remove 'add_' prefix
            
            # Get function docstring
            docstring_info = extract_docstring_info(func.__doc__)
            
            # Get default parameters
            params = extract_default_params(func)
            
            # Get function category
            category = categorize_noise_function(name)
            
            # Get function source code
            source_code = inspect.getsource(func)
            
            # Create metadata entry
            metadata[noise_name] = {
                "title": noise_name.replace('_', ' ').title(),
                "category": category,
                "description": docstring_info["description"],
                "formula": docstring_info["formula"],
                "formula_explanation": docstring_info["formula_explanation"],
                "parameters": params,
                "code_snippet": source_code
            }
    
    return metadata

def save_metadata(metadata, output_file):
    """Save metadata dictionary to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved to {output_file}")
    
    # Also generate a Python file with the metadata
    metadata_py = f"NOISE_METADATA = {json.dumps(metadata, indent=4)}"
    with open(output_file.replace('.json', '.py'), 'w', encoding='utf-8') as f:
        f.write(metadata_py)
    
    print(f"Metadata Python code saved to {output_file.replace('.json', '.py')}")

if __name__ == "__main__":
    metadata = generate_noise_metadata()
    save_metadata(metadata, os.path.join(os.path.dirname(__file__), 'noise_metadata.json'))
