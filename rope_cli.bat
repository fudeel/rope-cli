@echo off
REM run_rope_cli.bat
REM Windows batch script for running Rope CLI

echo ========================================
echo     Rope Deepfake CLI Runner
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

REM Set default paths (modify these as needed)
set DEFAULT_VIDEO_DIR=.\videos
set DEFAULT_FACES_DIR=.\faces
set DEFAULT_OUTPUT_DIR=.\output

REM Check if rope_cli.py exists
if not exist "rope_cli.py" (
    echo ERROR: rope_cli.py not found in current directory
    echo Please ensure rope_cli.py is in the same folder as this script
    pause
    exit /b 1
)

echo Available options:
echo 1. Quick process (use default folders)
echo 2. Custom process (specify paths)
echo 3. Batch process all videos
echo 4. Find faces only (no processing)
echo 5. Exit
echo.

set /p choice="Select option (1-5): "

if "%choice%"=="1" goto quick_process
if "%choice%"=="2" goto custom_process
if "%choice%"=="3" goto batch_process
if "%choice%"=="4" goto find_faces
if "%choice%"=="5" goto end

echo Invalid choice!
pause
goto end

:quick_process
echo.
echo Using default directories:
echo   Videos: %DEFAULT_VIDEO_DIR%
echo   Faces: %DEFAULT_FACES_DIR%
echo   Output: %DEFAULT_OUTPUT_DIR%
echo.

if not exist "%DEFAULT_VIDEO_DIR%" mkdir "%DEFAULT_VIDEO_DIR%"
if not exist "%DEFAULT_FACES_DIR%" mkdir "%DEFAULT_FACES_DIR%"
if not exist "%DEFAULT_OUTPUT_DIR%" mkdir "%DEFAULT_OUTPUT_DIR%"

echo Place your video files in: %DEFAULT_VIDEO_DIR%
echo Place your face images in: %DEFAULT_FACES_DIR%
echo.
pause

REM Process first video found
for %%f in (%DEFAULT_VIDEO_DIR%\*.mp4 %DEFAULT_VIDEO_DIR%\*.avi %DEFAULT_VIDEO_DIR%\*.mov) do (
    echo Processing: %%f
    python rope_cli.py -v "%%f" -f "%DEFAULT_FACES_DIR%" -o "%DEFAULT_OUTPUT_DIR%" -q 18 -t 2
    goto done_quick
)
echo No video files found in %DEFAULT_VIDEO_DIR%
pause
:done_quick
goto end

:custom_process
echo.
set /p video_path="Enter video file path: "
set /p faces_dir="Enter faces directory path: "
set /p output_dir="Enter output directory path: "
set /p quality="Enter quality (1-51, default 18): "
set /p threads="Enter number of threads (default 2): "

if "%quality%"=="" set quality=18
if "%threads%"=="" set threads=2

echo.
echo Processing with:
echo   Video: %video_path%
echo   Faces: %faces_dir%
echo   Output: %output_dir%
echo   Quality: %quality%
echo   Threads: %threads%
echo.

python rope_cli.py -v "%video_path%" -f "%faces_dir%" -o "%output_dir%" -q %quality% -t %threads%
pause
goto end

:batch_process
echo.
set /p video_dir="Enter directory containing videos: "
set /p faces_dir="Enter faces directory path: "
set /p output_dir="Enter output directory path: "

echo.
echo Processing all videos in: %video_dir%
echo.

for %%f in (%video_dir%\*.mp4 %video_dir%\*.avi %video_dir%\*.mov) do (
    echo Processing: %%~nxf
    python rope_cli.py -v "%%f" -f "%faces_dir%" -o "%output_dir%" -q 18 -t 2
    echo.
)

echo Batch processing complete!
pause
goto end

:find_faces
echo.
set /p video_path="Enter video file path to analyze: "

echo.
echo Finding faces in: %video_path%
echo.

python rope_cli.py -v "%video_path%" -f ".\temp" -o ".\temp" --find-faces-only
pause
goto end

:end
echo.
echo Goodbye!
exit /b 0