@echo off
rem Installs vame-motifs on Windows with a double click: finds conda, creates the
rem 'vame' environment if needed, installs this folder into it and adds a desktop
rem shortcut to the window. Run it again to update after pulling new code.
rem   install.bat        normal install
rem   install.bat dev    editable install with the test tools (pytest, ruff)
setlocal

set "ENV_NAME=vame"
set "PYTHON_VERSION=3.12"
set "SHORTCUT_NAME=VAME motifs"
set "REPO_DIR=%~dp0"

echo.
echo === vame-motifs installer ===
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
set "PREFIX_FILE=%TEMP%\vame-motifs-prefix.txt"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(sys.prefix)" > "%PREFIX_FILE%"
set /p ENV_PREFIX=<"%PREFIX_FILE%"
del "%PREFIX_FILE%" >nul 2>&1
set "ENV_PYTHON=%ENV_PREFIX%\python.exe"
if not exist "%ENV_PYTHON%" (
    echo Error: could not find the Python of the '%ENV_NAME%' environment.
    goto fail
)
echo Environment: %ENV_PREFIX%

rem 3. install this folder (pulls vame-py and torch the first time: can take a while)
echo.
echo Installing vame-motifs and its dependencies...
pushd "%REPO_DIR%"
"%ENV_PYTHON%" -m pip install --upgrade pip
if /i "%~1"=="dev" (
    "%ENV_PYTHON%" -m pip install --no-warn-script-location -e ".[dev]"
) else (
    "%ENV_PYTHON%" -m pip install --no-warn-script-location .
)
set "PIP_RESULT=%ERRORLEVEL%"
popd
if not "%PIP_RESULT%"=="0" goto fail

rem 4. desktop shortcut: pythonw starts the window without a console
echo.
echo Creating the desktop shortcut '%SHORTCUT_NAME%'...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$desktop = [Environment]::GetFolderPath('Desktop');" ^
    "$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop '%SHORTCUT_NAME%.lnk'));" ^
    "$shortcut.TargetPath = '%ENV_PREFIX%\pythonw.exe';" ^
    "$shortcut.Arguments = '-m vame_motifs.gui';" ^
    "$shortcut.WorkingDirectory = $env:USERPROFILE;" ^
    "$shortcut.IconLocation = '%ENV_PREFIX%\pythonw.exe,0';" ^
    "$shortcut.Save()"
if errorlevel 1 echo Warning: the shortcut could not be created; start the window with start-gui.bat instead.

echo.
echo === Done ===
echo Open the window with the '%SHORTCUT_NAME%' icon on the desktop, or with start-gui.bat.
echo From a terminal: conda activate %ENV_NAME% ^&^& vame-motifs --help
echo.
pause
exit /b 0

:fail
echo.
echo Installation failed: see the messages above.
echo.
pause
exit /b 1
