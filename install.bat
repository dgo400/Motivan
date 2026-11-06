@echo off
rem Installs Motivan on Windows with a double click: finds conda, creates the
rem 'vame' environment if needed, installs this folder into it and adds a desktop
rem shortcut to the window. Run it again to update after pulling new code.
rem   install.bat        normal install
rem   install.bat dev    editable install with the test tools (pytest, ruff)
setlocal

set "ENV_NAME=vame"
set "PYTHON_VERSION=3.12"
set "SHORTCUT_NAME=Motivan"
set "REPO_DIR=%~dp0"
rem PyTorch with CUDA, for PCs with an NVIDIA GPU (pip's default PyTorch on Windows is CPU-only).
rem cu132 = CUDA 12.x: works with any recent NVIDIA driver.
set "TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu132"

echo.
echo === Motivan installer ===
echo.

rem 1. find conda: the shell's own conda first, then the usual install folders
if defined CONDA_EXE if exist "%CONDA_EXE%" goto conda_found
for %%B in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~B\Scripts\conda.exe" (
        set "CONDA_EXE=%%~B\Scripts\conda.exe"
        goto conda_found
    )
)
echo Error: conda was not found.
echo Install Miniconda from https://www.anaconda.com/download/success and run this file again.
goto fail

:conda_found
echo Using conda: %CONDA_EXE%

rem 2. create the environment only if it does not exist yet
"%CONDA_EXE%" run -n %ENV_NAME% python --version >nul 2>&1
if not errorlevel 1 goto env_ready
echo Creating the '%ENV_NAME%' environment with Python %PYTHON_VERSION% (a few minutes)...
rem conda-forge only: avoids the Terms of Service prompt of the default channels
"%CONDA_EXE%" create -y -n %ENV_NAME% python=%PYTHON_VERSION% -c conda-forge --override-channels
if errorlevel 1 goto fail

:env_ready
rem the environment's own python, so nothing needs 'conda activate'
set "PREFIX_FILE=%TEMP%\Motivan-prefix.txt"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(sys.prefix)" > "%PREFIX_FILE%"
set /p ENV_PREFIX=<"%PREFIX_FILE%"
del "%PREFIX_FILE%" >nul 2>&1
set "ENV_PYTHON=%ENV_PREFIX%\python.exe"
if not exist "%ENV_PYTHON%" (
    echo Error: could not find the Python of the '%ENV_NAME%' environment.
    goto fail
)
echo Environment: %ENV_PREFIX%

rem 3. PyTorch: with CUDA if this PC has an NVIDIA GPU, so VAME trains on it.
rem    Installed before Motivan: VAME then finds PyTorch already there and keeps it.
echo.
"%ENV_PYTHON%" -m pip install --upgrade pip
where nvidia-smi >nul 2>&1
if errorlevel 1 goto torch_ready
echo NVIDIA GPU found.
rem already a CUDA build (e.g. when updating)? then nothing to do
"%ENV_PYTHON%" -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 goto torch_ready
echo Installing PyTorch with CUDA (a large download: several minutes)...
"%ENV_PYTHON%" -m pip uninstall -y torch >nul 2>&1
"%ENV_PYTHON%" -m pip install --no-warn-script-location torch --index-url %TORCH_CUDA_INDEX%
if errorlevel 1 goto fail

:torch_ready
rem 4. install this folder (pulls vame-py, and torch if not there yet: can take a while)
echo.
echo Installing Motivan and its dependencies...
pushd "%REPO_DIR%"
if /i "%~1"=="dev" (
    "%ENV_PYTHON%" -m pip install --no-warn-script-location -e ".[dev]"
) else (
    "%ENV_PYTHON%" -m pip install --no-warn-script-location .
)
set "PIP_RESULT=%ERRORLEVEL%"
popd
if not "%PIP_RESULT%"=="0" goto fail

rem 5. desktop shortcut to the Motivan-gui command (opens the window without a console);
rem    the shortcut of older installs ('VAME motifs') is removed
echo.
echo Creating the desktop shortcut '%SHORTCUT_NAME%'...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$desktop = [Environment]::GetFolderPath('Desktop');" ^
    "Remove-Item -LiteralPath (Join-Path $desktop 'VAME motifs.lnk') -ErrorAction SilentlyContinue;" ^
    "$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop '%SHORTCUT_NAME%.lnk'));" ^
    "$shortcut.TargetPath = '%ENV_PREFIX%\Scripts\Motivan-gui.exe';" ^
    "$shortcut.WorkingDirectory = $env:USERPROFILE;" ^
    "$shortcut.IconLocation = '%ENV_PREFIX%\pythonw.exe,0';" ^
    "$shortcut.Save()"
if errorlevel 1 echo Warning: the shortcut could not be created; start the window with start-gui.bat instead.

rem 6. tell where the models will train
echo.
"%ENV_PYTHON%" -c "from vame_motifs.pipeline import describe_device; print('Training device:', describe_device())"

echo.
echo === Done ===
echo Open the window with the '%SHORTCUT_NAME%' icon on the desktop, or with start-gui.bat.
echo From a terminal: conda activate %ENV_NAME% ^&^& Motivan --help
echo.
pause
exit /b 0

:fail
echo.
echo Installation failed: see the messages above.
echo.
pause
exit /b 1
