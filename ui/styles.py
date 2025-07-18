from core.resource_path import resource_path

BUTTON_STYLESHEET = """
QPushButton {
  background-color: #333333;
  border: 2px solid #333333;
  border-radius: 15px;
  color: #FFFFFF;
  font-size: 18px;
  font-weight: 500;
  padding: 16px 24px;
  text-align: center;
  font-family: 'Inter', Arial, sans-serif;
}
QPushButton:hover {
  background-color: #333A3F;
  border-color: #333A3F;
}
QPushButton:pressed {
  background-color: #333A3F;
  border-color: #333A3F;
}
"""

TAB_STYLESHEET = """ 
QTabBar::tab {
  font-size: 16px;
  padding: 10px 28px 10px 18px;
  min-width: 155px;
  min-height: 17px;
  border-top-left-radius: 5px 5px;
  border-top-right-radius: 5px 5px;
  border: .5px solid black;
  background-color: white;  
  font-family: 'Inter', Arial, sans-serif;
}
QTabBar::tab:last {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #eaffea, stop:1 #d4f5d4);
  color: #219150;
  font-weight: bold;
  font-size: 18px;
  border: 2px solid #219150;
  margin-left: 0px;
  margin-right: 0px;
  min-width: 340px;
  min-height: 32px;
  letter-spacing: 1px;
}
QTabBar::tab:selected {
  background-color: #121212;  
  color: white; 
  border: .5px solid #121212;
}
QTabBar::tab:disabled {
  background-color: #b0b0b0;
  color: #666666;
}
QTabBar::tab:hover {
  background-color: #eaffea; 
  color:#219150;
  border: 2px solid #219150;
}
QTabWidget::pane {
  background: white;
  border-top: 2px solid #121212;
}
"""

ARROW_IMAGE_PATH = resource_path("Images/down-arrow.png")
ESCAPED_PATH = ARROW_IMAGE_PATH.replace("\\", "/")

COMBOBOX_STYLESHEET = f"""
QComboBox {{
    background-color: white;
    border: 1px solid #dcdcdc;
    border-radius: 15px;
    padding: 5px 20px;
    font-size: 18px;
    color: #333;
    min-width: 200px;
    max-width: 300px;
    font-family: 'Inter', Arial, sans-serif;
}}
QComboBox::down-arrow {{
    image: url("{ESCAPED_PATH}"); 
    background-color: transparent;
    border: none;
    width: 13px;
    height: 13px;
    padding-right: 30px;
}}
QComboBox::drop-down {{
    border: 0px;
}}
QComboBox:hover {{
    border: 1px solid #0078d4;
}}
QComboBox:focus {{
    border: 1px solid #0078d4;
    padding: 4px 19px;
}}
"""

PREDICTION_ENABLED_STYLESHEET = """
QPushButton {
    background-color: #27ae60;
    border: 2px solid #27ae60;
    border-radius: 15px;
    color: #fff;
    font-size: 18px;
    font-weight: 500;
    padding: 16px 24px;
    text-align: center;
    font-family: 'Inter', Arial, sans-serif;
}
QPushButton:hover {
    background-color: #219150;
    border-color: #219150;
}
QPushButton:pressed {
    background-color: #219150;
    border-color: #219150;
}
"""

PREDICTION_DISABLED_STYLESHEET = """
QPushButton {
    background-color: #e0e0e0;
    border: 2px solid #cccccc;
    border-radius: 15px;
    color: #aaaaaa;
    font-size: 18px;
    font-weight: 500;
    padding: 16px 24px;
    text-align: center;
    font-family: 'Inter', Arial, sans-serif;
}
"""