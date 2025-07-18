@echo off
pyinstaller --onefile --windowed main.py --name "TC_AI_Prediction_Tool" ^
  --add-data "Images;Images" ^
  --add-data "Font;Font" ^
  --icon "icon.ico"