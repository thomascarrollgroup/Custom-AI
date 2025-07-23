@echo off
pyinstaller --name "Custom AI Prediction" --onefile --windowed --icon=icon.ico  --add-data "Images;Images"  --add-data "Font;Font" --add-data ".env;." --paths core --paths ui --hidden-import sklearn._cyutility --hidden-import sklearn.utils._cython_blas --hidden-import sklearn.utils._cython_utils main.py
