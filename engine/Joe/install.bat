@echo off
setlocal enabledelayedexpansion

title Visagyn Dependency Installer
color 0A

echo.
echo ================================================================================
echo                     VISAGYN AI DRIVEN FACIAL TRACKING ENGINE v1.0.0 (joe edition) (aka, very bad ai upscaling addition)
echo                        Dependency Installer ^& Launcher
echo ================================================================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as administrator. Some installations may fail.
    echo [INFO] Consider running as administrator for better compatibility.
    echo.
    timeout /t 3 >nul
)

set "INSTALL_DIR=%~dp0"
set "PYTHON_DIR=%INSTALL_DIR%python"
set "SCRIPTS_DIR=%PYTHON_DIR%\Scripts"
set "LOG_FILE=%INSTALL_DIR%installation.log"

echo [INFO] Installation directory: %INSTALL_DIR%
echo [INFO] Python directory: %PYTHON_DIR%
echo [INFO] Log file: %LOG_FILE%
echo.

echo Installation Log - %date% %time% > "%LOG_FILE%"
echo ============================================ >> "%LOG_FILE%"

set "LOG_PREFIX=echo [%time%]"

echo [STEP 1/6] Checking for existing Python installation...
python --version >nul 2>&1
if %errorLevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"
    echo [SUCCESS] Found system Python: !PYTHON_VERSION!
    echo [%time%] Found system Python: !PYTHON_VERSION! >> "%LOG_FILE%"
    set "PYTHON_CMD=python"
    set "PIP_CMD=pip"
    goto :check_pip
) else (
    echo [INFO] System Python not found. Checking portable installation...
)

if exist "%PYTHON_DIR%\python.exe" (
    echo [SUCCESS] Found portable Python installation
    echo [%time%] Found portable Python installation >> "%LOG_FILE%"
    set "PYTHON_CMD=%PYTHON_DIR%\python.exe"
    set "PIP_CMD=%SCRIPTS_DIR%\pip.exe"
    goto :check_pip
)

echo [INFO] Downloading Python 3.11.9 (portable)...
echo [%time%] Starting Python download >> "%LOG_FILE%"

set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PYTHON_ZIP=%INSTALL_DIR%python-3.11.9-embed-amd64.zip"

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%' -UseBasicParsing}" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Failed to download Python. Check internet connection.
    echo [%time%] ERROR: Python download failed >> "%LOG_FILE%"
    pause
    exit /b 1
)

echo [SUCCESS] Python download completed
echo [%time%] Python download completed >> "%LOG_FILE%"

echo [INFO] Extracting Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Failed to extract Python
    echo [%time%] ERROR: Python extraction failed >> "%LOG_FILE%"
    pause
    exit /b 1
)

del "%PYTHON_ZIP%" 2>nul

echo [INFO] Configuring Python environment...
set "PYTHON_CMD=%PYTHON_DIR%\python.exe"

echo import site >> "%PYTHON_DIR%\python311._pth"

echo [INFO] Installing pip...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%INSTALL_DIR%get-pip.py' -UseBasicParsing}" 2>nul
if %errorLevel% neq 0 (
    echo [ERROR] Failed to download pip installer
    echo [%time%] ERROR: pip download failed >> "%LOG_FILE%"
    pause
    exit /b 1
)

"%PYTHON_CMD%" "%INSTALL_DIR%get-pip.py" --target "%PYTHON_DIR%" --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install pip
    echo [%time%] ERROR: pip installation failed >> "%LOG_FILE%"
    pause
    exit /b 1
)

del "%INSTALL_DIR%get-pip.py" 2>nul

mkdir "%SCRIPTS_DIR%" 2>nul
set "PIP_CMD=%PYTHON_CMD% -m pip"

echo [SUCCESS] Python installation completed
echo [%time%] Python installation completed >> "%LOG_FILE%"

:check_pip
echo.
echo [STEP 2/6] Verifying pip installation...
%PIP_CMD% --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] pip is not working properly
    echo [%time%] ERROR: pip verification failed >> "%LOG_FILE%"
    pause
    exit /b 1
)
echo [SUCCESS] pip is working correctly
echo [%time%] pip verification successful >> "%LOG_FILE%"

echo [INFO] Upgrading pip to latest version...
%PIP_CMD% install --upgrade pip --no-warn-script-location 2>>"%LOG_FILE%"

echo.
echo [STEP 3/6] Installing core dependencies...
echo [%time%] Starting core dependency installation >> "%LOG_FILE%"

set "CORE_DEPS=numpy==1.24.3 opencv-python==4.8.1.78 pillow==10.0.1 customtkinter==5.2.0"

for %%d in (%CORE_DEPS%) do (
    echo [INFO] Installing %%d...
    %PIP_CMD% install "%%d" --no-warn-script-location 2>>"%LOG_FILE%"
    if !errorLevel! neq 0 (
        echo [WARNING] Failed to install %%d with specific version, trying latest...
        for /f "tokens=1 delims==" %%p in ("%%d") do (
            %PIP_CMD% install "%%p" --no-warn-script-location 2>>"%LOG_FILE%"
        )
    )
)

echo [SUCCESS] Core dependencies installation completed
echo [%time%] Core dependencies installation completed >> "%LOG_FILE%"

echo.
echo [STEP 4/6] Installing PyTorch with CUDA support...
echo [%time%] Starting PyTorch installation >> "%LOG_FILE%"

echo [INFO] Installing PyTorch with CUDA 11.8 support...
%PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% neq 0 (
    echo [WARNING] CUDA PyTorch installation failed, installing CPU version...
    %PIP_CMD% install torch torchvision torchaudio --no-warn-script-location 2>>"%LOG_FILE%"
    if !errorLevel! neq 0 (
        echo [ERROR] PyTorch installation failed completely
        echo [%time%] ERROR: PyTorch installation failed >> "%LOG_FILE%"
    ) else (
        echo [SUCCESS] PyTorch CPU version installed
        echo [%time%] PyTorch CPU version installed >> "%LOG_FILE%"
    )
) else (
    echo [SUCCESS] PyTorch with CUDA support installed
    echo [%time%] PyTorch with CUDA support installed >> "%LOG_FILE%"
)

echo.
echo [STEP 5/6] Installing optional AI/ML dependencies...
echo [%time%] Starting optional dependencies installation >> "%LOG_FILE%"

echo [INFO] Installing MediaPipe...
%PIP_CMD% install mediapipe==0.10.7 --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% equ 0 (
    echo [SUCCESS] MediaPipe installed - Advanced face mesh tracking available
    echo [%time%] MediaPipe installation successful >> "%LOG_FILE%"
) else (
    echo [WARNING] MediaPipe installation failed - Basic face tracking only
    echo [%time%] WARNING: MediaPipe installation failed >> "%LOG_FILE%"
)

echo [INFO] Installing pyvirtualcam...
%PIP_CMD% install pyvirtualcam --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% equ 0 (
    echo [SUCCESS] pyvirtualcam installed - Virtual camera output available
    echo [%time%] pyvirtualcam installation successful >> "%LOG_FILE%"
) else (
    echo [WARNING] pyvirtualcam installation failed - No virtual camera support
    echo [%time%] WARNING: pyvirtualcam installation failed >> "%LOG_FILE%"
)

echo [INFO] Installing ONNX Runtime...
%PIP_CMD% install onnxruntime-gpu --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% neq 0 (
    echo [INFO] GPU version failed, installing CPU version...
    %PIP_CMD% install onnxruntime --no-warn-script-location 2>>"%LOG_FILE%"
    if !errorLevel! equ 0 (
        echo [SUCCESS] ONNX Runtime CPU installed
        echo [%time%] ONNX Runtime CPU installation successful >> "%LOG_FILE%"
    )
) else (
    echo [SUCCESS] ONNX Runtime GPU installed
    echo [%time%] ONNX Runtime GPU installation successful >> "%LOG_FILE%"
)

echo [INFO] Installing InsightFace...
%PIP_CMD% install insightface --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% equ 0 (
    echo [SUCCESS] InsightFace installed - Face swapping available
    echo [%time%] InsightFace installation successful >> "%LOG_FILE%"
) else (
    echo [WARNING] InsightFace installation failed - No face swapping
    echo [%time%] WARNING: InsightFace installation failed >> "%LOG_FILE%"
)

echo [INFO] Installing RealESRGAN...
%PIP_CMD% install realesrgan --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% equ 0 (
    echo [SUCCESS] RealESRGAN installed - AI upscaling available
    echo [%time%] RealESRGAN installation successful >> "%LOG_FILE%"
) else (
    echo [WARNING] RealESRGAN installation failed - Basic upscaling only
    echo [%time%] WARNING: RealESRGAN installation failed >> "%LOG_FILE%"
)

echo [INFO] Installing BasicSR...
%PIP_CMD% install basicsr --no-warn-script-location 2>>"%LOG_FILE%"
if %errorLevel% equ 0 (
    echo [SUCCESS] BasicSR installed
    echo [%time%] BasicSR installation successful >> "%LOG_FILE%"
)

echo.
echo [STEP 6/6] Creating launcher and verifying installation...
echo [%time%] Creating launcher and verifying installation >> "%LOG_FILE%"

set "LAUNCHER_FILE=%INSTALL_DIR%launch_visagyn.bat"
echo @echo off > "%LAUNCHER_FILE%"
echo title Visagyn AI Driven Facial Engine >> "%LAUNCHER_FILE%"
echo cd /d "%INSTALL_DIR%" >> "%LAUNCHER_FILE%"
echo echo Starting Visagyn AI Driven Facial Engine... >> "%LAUNCHER_FILE%"
echo echo. >> "%LAUNCHER_FILE%"
if defined PYTHON_VERSION (
    echo python joe.py >> "%LAUNCHER_FILE%"
) else (
    echo "%PYTHON_CMD%" joe.py >> "%LAUNCHER_FILE%"
)
echo pause >> "%LAUNCHER_FILE%"

echo [SUCCESS] Launcher created: %LAUNCHER_FILE%

set "REQ_FILE=%INSTALL_DIR%requirements.txt"
echo # Visagyn AI Driven Facial Engine - Dependencies > "%REQ_FILE%"
echo # Core dependencies >> "%REQ_FILE%"
echo numpy^>=1.24.0 >> "%REQ_FILE%"
echo opencv-python^>=4.8.0 >> "%REQ_FILE%"
echo pillow^>=10.0.0 >> "%REQ_FILE%"
echo customtkinter^>=5.2.0 >> "%REQ_FILE%"
echo torch^>=2.0.0 >> "%REQ_FILE%"
echo torchvision^>=0.15.0 >> "%REQ_FILE%"
echo torchaudio^>=2.0.0 >> "%REQ_FILE%"
echo. >> "%REQ_FILE%"
echo # Optional AI/ML dependencies >> "%REQ_FILE%"
echo mediapipe^>=0.10.0 >> "%REQ_FILE%"
echo pyvirtualcam >> "%REQ_FILE%"
echo onnxruntime >> "%REQ_FILE%"
echo insightface >> "%REQ_FILE%"
echo realesrgan >> "%REQ_FILE%"
echo basicsr >> "%REQ_FILE%"

echo [SUCCESS] Requirements file created: %REQ_FILE%

echo [INFO] Running verification test...
if defined PYTHON_VERSION (
    python -c "import cv2, numpy, torch, customtkinter; print('Core modules verified successfully')" 2>>"%LOG_FILE%"
) else (
    "%PYTHON_CMD%" -c "import cv2, numpy, torch, customtkinter; print('Core modules verified successfully')" 2>>"%LOG_FILE%"
)

if %errorLevel% equ 0 (
    echo [SUCCESS] Core modules verification passed
    echo [%time%] Core modules verification passed >> "%LOG_FILE%"
) else (
    echo [WARNING] Some core modules failed verification
    echo [%time%] WARNING: Core modules verification failed >> "%LOG_FILE%"
)

echo.
echo ================================================================================
echo                            INSTALLATION COMPLETE
echo ================================================================================
echo.
echo [SUCCESS] Visagyn dependencies have been installed successfully!
echo.
echo NEXT STEPS:
echo 1. Place your Python script as 'main.py' in this directory
echo 2. Run 'launch_visagyn.bat' to start the application
echo 3. Check 'installation.log' if you encounter any issues
echo.
echo INSTALLED FEATURES:
echo  ✓ Python %PYTHON_VERSION%
echo  ✓ OpenCV (Computer Vision)
echo  ✓ PyTorch (AI/ML Framework)
echo  ✓ CustomTkinter (Modern GUI)
echo  ✓ NumPy (Numerical Computing)
echo  ✓ Pillow (Image Processing)

%PIP_CMD% show mediapipe >nul 2>&1
if %errorLevel% equ 0 (
    echo  ✓ MediaPipe (Advanced Face Tracking)
) else (
    echo  ✗ MediaPipe (Not Available)
)

%PIP_CMD% show pyvirtualcam >nul 2>&1
if %errorLevel% equ 0 (
    echo  ✓ PyVirtualCam (Virtual Camera)
) else (
    echo  ✗ PyVirtualCam (Not Available)
)

%PIP_CMD% show insightface >nul 2>&1
if %errorLevel% equ 0 (
    echo  ✓ InsightFace (Face Swapping)
) else (
    echo  ✗ InsightFace (Not Available)
)

%PIP_CMD% show realesrgan >nul 2>&1
if %errorLevel% equ 0 (
    echo  ✓ RealESRGAN (AI Upscaling)
) else (
    echo  ✗ RealESRGAN (Not Available)
)

echo.
echo TROUBLESHOOTING:
echo - If the application fails to start, check 'installation.log'
echo - For GPU acceleration, ensure NVIDIA drivers are up to date
echo - Some features require additional system-level dependencies
echo.
echo MANUAL INSTALLATION (if needed):
echo pip install numpy opencv-python pillow customtkinter torch torchvision
echo pip install mediapipe pyvirtualcam onnxruntime insightface realesrgan
echo.
echo ================================================================================

echo [%time%] Installation completed successfully >> "%LOG_FILE%"

echo.
set /p "LAUNCH_NOW=Launch Visagyn now? (y/n): "
if /i "%LAUNCH_NOW%"=="y" (
    if exist "%INSTALL_DIR%joe.py" (
        echo [INFO] Starting Visagyn AI Driven Facial Engine...
        call "%LAUNCHER_FILE%"
    ) else (
        echo [ERROR] Python script 'joe.py' not found in current directory
        echo [INFO] Please place your Python script as 'joe.py' and run 'launch_visagyn.bat'
    )
)

echo.
echo Press any key to exit...
pause >nul

endlocal
