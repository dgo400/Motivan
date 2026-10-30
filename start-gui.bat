@echo off
rem Opens the vame-motifs window from the 'vame' conda environment, without activating it.
rem Pass an experiment file to open it: start-gui.bat experiment.yaml
setlocal

set "ENV_NAME=vame"

if defined CONDA_EXE if exist "%CONDA_EXE%" goto conda_found
for %%B in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if exist "%%~B\Scripts\conda.exe" (
        set "CONDA_EXE=%%~B\Scripts\conda.exe"
        goto conda_found
    )
)
echo Error: conda was not found. Run install.bat first.
pause
exit /b 1

:conda_found
set "PREFIX_FILE=%TEMP%\vame-motifs-prefix.txt"
"%CONDA_EXE%" run -n %ENV_NAME% python -c "import sys; print(sys.prefix)" > "%PREFIX_FILE%"
set /p ENV_PREFIX=<"%PREFIX_FILE%"
del "%PREFIX_FILE%" >nul 2>&1
if not exist "%ENV_PREFIX%\pythonw.exe" (
    echo Error: the '%ENV_NAME%' environment was not found. Run install.bat first.
    pause
    exit /b 1
)

rem 'start' returns at once, so this console closes and only the window stays open
start "" "%ENV_PREFIX%\pythonw.exe" -m vame_motifs.gui %*
