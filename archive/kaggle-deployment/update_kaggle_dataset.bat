@echo off
REM Update Kaggle Dataset with Relation Prediction Fixes
REM Run this script to create a new zip file with your updated code

echo === PSN-2 Kaggle Dataset Updater ===
echo.
echo This will create psn2_kaggle_full_v2.zip with the relation prediction fixes.
echo.

REM Create temporary directory
set TEMP_DIR=psn2_kaggle_temp
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"
mkdir "%TEMP_DIR%"

echo Copying files...

REM Copy all necessary files
xcopy /E /I /Q psn2 "%TEMP_DIR%\psn2"
xcopy /E /I /Q configs "%TEMP_DIR%\configs"
xcopy /E /I /Q data "%TEMP_DIR%\data"
xcopy /E /I /Q scripts "%TEMP_DIR%\scripts"
copy train.py "%TEMP_DIR%\"
copy train_sequential.py "%TEMP_DIR%\"
copy evaluate.py "%TEMP_DIR%\"
copy README.md "%TEMP_DIR%\"
copy PRD.md "%TEMP_DIR%\"
copy CHANGELOG.md "%TEMP_DIR%\"
copy RELATION_PREDICTION_FIX.md "%TEMP_DIR%\"
copy SEQUENTIAL_TRAINING_GUIDE.md "%TEMP_DIR%\"

echo Cleaning up...
REM Remove unnecessary files
for /d /r "%TEMP_DIR%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r "%TEMP_DIR%" %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r "%TEMP_DIR%" %%d in (.git) do @if exist "%%d" rd /s /q "%%d"
del /s /q "%TEMP_DIR%\*.pyc" 2>nul

echo Creating zip file...
REM Use PowerShell to create zip (built into Windows 10+)
powershell -command "Compress-Archive -Path '%TEMP_DIR%\*' -DestinationPath 'psn2_kaggle_full_v2.zip' -Force"

REM Cleanup
rmdir /s /q "%TEMP_DIR%"

echo.
echo Done! Created psn2_kaggle_full_v2.zip
echo.
echo Next steps:
echo 1. Go to kaggle.com/datasets
echo 2. Find your 'psn2-kaggle' dataset
echo 3. Click 'New Version'
echo 4. Upload psn2_kaggle_full_v2.zip
echo 5. Add version notes: 'Relation prediction fixes - bond formation and masked entity context'
echo 6. Click 'Create'
echo.
echo Your existing Kaggle notebooks will automatically use the new version!
pause
