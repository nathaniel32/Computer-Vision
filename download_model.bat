@echo off
set URL=https://github.com/nathaniel32/Computer-Vision/releases/download/v1/best_model.zip
set ZIP=segmentation_model_tmp.zip
set DEST=.

if not exist "%DEST%" (
    mkdir "%DEST%"
)

curl -L -o "%ZIP%" "%URL%"
powershell -Command "Expand-Archive '%ZIP%' '%DEST%' -Force"
del "%ZIP%"