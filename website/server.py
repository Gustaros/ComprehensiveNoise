import http.server
import socketserver
import os
import sys
import json
import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import parse_qs, urlparse

# Add parent directory to path to import noise library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import noises

PORT = 8000
SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

class NoiseVisualizationHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Fix initialization for Python 3.7+
        super().__init__(*args, directory=SITE_DIR, **kwargs)
    
    def do_GET(self):
        # Debug log
        print(f"Received request for: {self.path}")
        
        url_parts = urlparse(self.path)
        
        # Fixed: Handle root path by redirecting to index.html
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
            
        # Handle API requests
        if url_parts.path.startswith('/api/'):
            if url_parts.path == '/api/generate_noise':
                # Parse query parameters
                query = parse_qs(url_parts.query)
                noise_type = query.get('type', ['gaussian_noise'])[0]
                params_str = query.get('params', ['{}'])[0]
                
                try:
                    params = json.loads(params_str)
                    self.generate_noise_visualization(noise_type, params)
                except Exception as e:
                    self.send_error(500, str(e))
            else:
                self.send_error(404, "API endpoint not found")
        else:
            try:
                # Serve static files
                super().do_GET()
            except Exception as e:
                print(f"Error serving file: {e}")
                self.send_error(500, f"Error serving file: {e}")
    
    def generate_noise_visualization(self, noise_type, params):
        # Convert function name
        func_name = f"add_{noise_type}"
        
        if hasattr(noises, func_name):
            noise_func = getattr(noises, func_name)
            
            # Create sample image
            sample_image = noises.create_sample_image(pattern="gradient")
            
            # Apply noise with parameters
            noisy_image, noise_pattern = noise_func(sample_image, **params)
            
            # Generate visualization image
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
            
            plt.tight_layout()
            
            # Save to buffer
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            plt.close(fig)
            img_buf.seek(0)
            img_data = base64.b64encode(img_buf.read()).decode('utf-8')
            
            # Generate histogram data
            flat_noise = noise_pattern.flatten()
            
            # Remove outliers for better visualization
            q1, q3 = np.percentile(flat_noise, [1, 99])
            bin_data = flat_noise[(flat_noise >= q1) & (flat_noise <= q3)]
            
            hist, bin_edges = np.histogram(bin_data, bins=30, density=True)
            
            histogram_data = {
                "labels": [float(x) for x in bin_edges[:-1]],
                "values": [float(x) for x in hist]
            }
            
            # Generate spectrum data
            # Using FFT for a simple spectrum visualization
            fft = np.abs(np.fft.rfft2(noise_pattern))
            fft_log = np.log1p(fft)
            
            # Average over angles to get 1D spectrum
            center_y, center_x = fft.shape[0] // 2, fft.shape[1] // 2
            y, x = np.ogrid[:fft.shape[0], :fft.shape[1]]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
            r_max = min(center_y, center_x)
            
            radial_prof = np.zeros(r_max)
            for i in range(1, r_max):
                radial_prof[i] = np.mean(fft_log[r == i])
            
            spectrum_data = {
                "labels": list(range(1, r_max)),
                "values": [float(x) for x in radial_prof[1:]]
            }
            
            # Create response
            response_data = {
                "image": f"data:image/png;base64,{img_data}",
                "histogram": histogram_data,
                "spectrum": spectrum_data
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_error(404, f"Noise function {func_name} not found")

def run_server():
    # Check if output directory exists and has content
    if not os.path.exists(SITE_DIR):
        print(f"Error: Site directory {SITE_DIR} doesn't exist!")
        return
        
    files = os.listdir(SITE_DIR)
    if not files:
        print(f"Warning: Site directory {SITE_DIR} is empty!")
    else:
        print(f"Found {len(files)} files in site directory")
    
    # Create server with fixed handler
    handler = NoiseVisualizationHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        print(f"Try accessing: http://localhost:{PORT}/index.html")
        print(f"Site directory: {SITE_DIR}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
