import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_writable_path(relative_path):
    base_path = os.path.expanduser("~")
    return os.path.join(base_path, "TC_AI_Prediction_Tool", relative_path)