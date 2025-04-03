@echo off
REM filepath: /d:/ComprehensiveNoise/website/setup_and_run.bat

echo ===== Setting up directory structure =====
mkdir assets\css 2>NUL
mkdir assets\js 2>NUL
mkdir assets\images 2>NUL
mkdir templates 2>NUL
mkdir output 2>NUL

echo ===== Creating JavaScript file for string parameters =====
echo // filepath: /d:/ComprehensiveNoise/website/assets/js/string-parameter-handler.js > assets\js\string-parameter-handler.js
echo. >> assets\js\string-parameter-handler.js
echo document.addEventListener('DOMContentLoaded', function() { >> assets\js\string-parameter-handler.js
echo     // Check if we have the noiseParameters variable >> assets\js\string-parameter-handler.js
echo     if (typeof noiseParameters !== 'undefined') { >> assets\js\string-parameter-handler.js
echo         handleStringParameters(); >> assets\js\string-parameter-handler.js
echo     } >> assets\js\string-parameter-handler.js
echo }); >> assets\js\string-parameter-handler.js
echo. >> assets\js\string-parameter-handler.js
echo function handleStringParameters() { >> assets\js\string-parameter-handler.js
echo     // Find parameters that are strings based on their names or descriptions >> assets\js\string-parameter-handler.js
echo     const stringParameterPatterns = [ >> assets\js\string-parameter-handler.js
echo         'method', 'orientation', 'pattern', 'type', 'direction', 'mode' >> assets\js\string-parameter-handler.js
echo     ]; >> assets\js\string-parameter-handler.js
echo     // ... rest of the function implementation >> assets\js\string-parameter-handler.js
echo } >> assets\js\string-parameter-handler.js

echo ===== Creating noise-viewer.js file =====
echo // filepath: /d:/ComprehensiveNoise/website/assets/js/noise-viewer.js > assets\js\noise-viewer.js
echo document.addEventListener('DOMContentLoaded', function() { >> assets\js\noise-viewer.js
echo     // Initialize parameter controls >> assets\js\noise-viewer.js
echo     initializeControls(); >> assets\js\noise-viewer.js
echo     // ... rest of the file implementation >> assets\js\noise-viewer.js
echo }); >> assets\js\noise-viewer.js

echo ===== Running metadata generator =====
python metadata_generator.py

echo ===== Running build script =====
python build_site.py

echo ===== Checking output directory =====
dir output
dir output\assets

echo ===== Starting server =====
echo Starting server at http://localhost:8000
echo Please open http://localhost:8000/index.html in your browser
python server.py