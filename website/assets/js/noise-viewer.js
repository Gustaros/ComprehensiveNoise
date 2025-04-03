// filepath: /d:/ComprehensiveNoise/website/assets/js/noise-viewer.js 
document.addEventListener('DOMContentLoaded', function() {
    // Initialize parameter controls
    initializeControls();
    
    // Initialize charts
    initializeCharts();
    
    // Add event listeners
    document.getElementById('update-visualization').addEventListener('click', updateVisualization);
    
    // Add input event listeners to all sliders
    const sliders = document.querySelectorAll('.parameter-control input');
    sliders.forEach(slider => {
        slider.addEventListener('input', function() {
            // Update the displayed value
            const valueDisplay = this.nextElementSibling;
            valueDisplay.textContent = this.value;
        });
    });
});

// Initialize parameter controls with default values
function initializeControls() {
    for (const [paramName, paramData] of Object.entries(noiseParameters)) {
        const slider = document.getElementById(paramName);
        if (slider) {
            slider.value = paramData.default;
            // Update the displayed value
            const valueDisplay = slider.nextElementSibling;
            if (valueDisplay) {
                valueDisplay.textContent = paramData.default;
            }
        }
    }
}

// Chart objects
let histogramChart = null;
let spectrumChart = null;

// Initialize chart placeholders
function initializeCharts() {
    // Check if Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('Chart.js is not loaded!');
        return;
    }

    try {
        // Histogram chart
        const histogramCtx = document.getElementById('noise-histogram');
        if (histogramCtx) {
            histogramChart = new Chart(histogramCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: Array.from({length: 30}, (_, i) => i),
                    datasets: [{
                        label: 'Frequency',
                        data: Array.from({length: 30}, () => Math.random() * 100),
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Noise Value Distribution'
                        }
                    }
                }
            });
        }
        
        // Spectrum chart
        const spectrumCtx = document.getElementById('noise-spectrum');
        if (spectrumCtx) {
            spectrumChart = new Chart(spectrumCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array.from({length: 50}, (_, i) => i),
                    datasets: [{
                        label: 'Power',
                        data: Array.from({length: 50}, (_, i) => 100 / (i+1)),
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            type: 'logarithmic'
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Frequency Spectrum'
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

// Update visualization based on current parameters
function updateVisualization() {
    // Collect current parameter values
    const params = {};
    for (const paramName in noiseParameters) {
        const slider = document.getElementById(paramName);
        if (slider) {
            // Convert to appropriate type (assuming number for now)
            params[paramName] = parseFloat(slider.value);
        }
    }
    
    // Show loading state
    const button = document.getElementById('update-visualization');
    const originalText = button.textContent;
    button.textContent = 'Loading...';
    button.disabled = true;
    
    try {
        // Simulate response for static site demo
        simulateResponse(params);
    } catch (error) {
        console.error('Error updating visualization:', error);
    }
    
    // Reset button
    button.textContent = originalText;
    button.disabled = false;
}

// Simulate response for static site without actual backend
function simulateResponse(params) {
    // Generate random histogram data
    const histogramData = generateRandomHistogram();
    updateHistogramChart(histogramData);
    
    // Generate random spectrum data
    const spectrumData = generateRandomSpectrum();
    updateSpectrumChart(spectrumData);
    
    // For static demo, we'll just display the parameters in the console
    console.log('Parameters:', params);
    
    // Add a note about static demo
    const note = document.createElement('div');
    note.className = 'static-demo-note';
    note.innerHTML = `
        <p style="background-color: #ffe0b2; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <strong>Note:</strong> This is a static demo. In a real implementation, this would connect 
            to a backend service to generate updated visualizations. Current parameters: 
            ${Object.entries(params).map(([key, value]) => `${key}=${value}`).join(', ')}
        </p>
    `;
    
    // Remove any existing notes
    const existingNote = document.querySelector('.static-demo-note');
    if (existingNote) {
        existingNote.remove();
    }
    
    const visualizationOutput = document.querySelector('.visualization-output');
    if (visualizationOutput) {
        visualizationOutput.appendChild(note);
    }
}

// Update histogram chart with new data
function updateHistogramChart(data) {
    if (histogramChart) {
        histogramChart.data.labels = data.labels;
        histogramChart.data.datasets[0].data = data.values;
        histogramChart.update();
    }
}

// Update spectrum chart with new data
function updateSpectrumChart(data) {
    if (spectrumChart) {
        spectrumChart.data.labels = data.labels;
        spectrumChart.data.datasets[0].data = data.values;
        spectrumChart.update();
    }
}

// Generate random histogram data for demo
function generateRandomHistogram() {
    const numBins = 30;
    const labels = Array.from({length: numBins}, (_, i) => i);
    
    // Generate normal distribution values
    const mean = numBins / 2;
    const stdDev = numBins / 6;
    const values = Array.from({length: numBins}, (_, i) => {
        const x = i;
        return Math.exp(-0.5 * Math.pow((x - mean) / stdDev, 2)) / (stdDev * Math.sqrt(2 * Math.PI));
    });
    
    // Scale values
    const maxValue = Math.max(...values);
    const scaledValues = values.map(v => v / maxValue * 100);
    
    return {
        labels: labels,
        values: scaledValues
    };
}

// Generate random spectrum data for demo
function generateRandomSpectrum() {
    const numPoints = 50;
    const labels = Array.from({length: numPoints}, (_, i) => i);
    
    // Generate 1/f spectrum (pink noise)
    const values = Array.from({length: numPoints}, (_, i) => {
        const freq = i + 1;
        return 10 / Math.sqrt(freq) * (0.5 + 0.5 * Math.random());
    });
    
    return {
        labels: labels,
        values: values
    };
}
