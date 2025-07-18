from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QMessageBox, QListWidget, QAbstractItemView, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QInputDialog, QProgressDialog, QHeaderView, QScrollArea, QDialog,
    QLineEdit, QStackedLayout, QTextEdit, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont, QFontDatabase, QPixmap, QPainter, QIcon
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ui.styles import BUTTON_STYLESHEET, TAB_STYLESHEET, COMBOBOX_STYLESHEET, PREDICTION_ENABLED_STYLESHEET, PREDICTION_DISABLED_STYLESHEET
from ui.dialogs import ColumnSelectDialog
from core.logging import log_error_to_neon
from core.resource_path import resource_path, get_writable_path
from core.data import load_csv
from core.model import auto_train_and_evaluate_models, save_model
from core.preprocess import auto_preprocess_data
from core.model_worker import ModelTrainingWorker
import sys
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

class BespokePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.user_name = None
        self.prediction_help_state = "default"
        self.setWindowTitle("TC AI Prediction Tool")
        self.setWindowIcon(QIcon(resource_path("Images/icon.png")))
        self.df = None
        self.selected_features = []
        self.background_image = QPixmap(resource_path("Images/wire-svg.svg"))

        self.stacked_layout = QStackedLayout(self)
        self.init_login_page()
        self.main_widget = QWidget()
        self.init_main_ui(self.main_widget)
        self.stacked_layout.addWidget(self.login_widget)
        self.stacked_layout.addWidget(self.main_widget)
        self.setLayout(self.stacked_layout)
        self.stacked_layout.setCurrentWidget(self.login_widget)

    def init_main_ui(self, main_widget):
        main_widget.setAttribute(Qt.WA_StyledBackground, True)
        main_widget.setStyleSheet("background: transparent;")
        self.init_fonts()
        self.init_palette()
        self.init_tabs()
        self.init_home_tab()
        self.init_select_tab()
        self.init_quality_tab()
        self.init_preprocess_tab()
        self.init_prediction_tab()
        main_layout = QVBoxLayout(main_widget)
        help_row = QHBoxLayout()
        help_row.addStretch()
        self.global_help_btn = QPushButton("Help")
        self.global_help_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.global_help_btn.clicked.connect(self.show_context_help)
        help_row.addWidget(self.global_help_btn)
        main_layout.addLayout(help_row)
        main_layout.addWidget(self.tabs)
        self.set_tab_margin(80)
        main_widget.setLayout(main_layout)
        self.set_tab_enabled(1, False)
        self.set_tab_enabled(2, False)
        self.set_tab_enabled(3, False)
        self.set_tab_enabled(4, True)

    def init_login_page(self):
        self.login_widget = QWidget()
        self.login_widget.setAttribute(Qt.WA_StyledBackground, True)
        self.login_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(self.login_widget)
        layout.setAlignment(Qt.AlignCenter)
        icon_path = resource_path("Images/icon.png")
        icon_label = QLabel()
        pixmap = QPixmap(icon_path)
        pixmap = pixmap.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label, alignment=Qt.AlignCenter)
        label = QLabel("Enter your name to continue:")
        label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(label, alignment=Qt.AlignCenter)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Your name")
        self.name_input.setMinimumWidth(250)
        self.name_input.setMaximumWidth(350)
        self.name_input.setStyleSheet("font-size: 18px; padding: 8px;")
        layout.addWidget(self.name_input, alignment=Qt.AlignCenter)
        self.login_btn = QPushButton("Continue")
        self.login_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn, alignment=Qt.AlignCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        if hasattr(self, "stacked_layout") and self.stacked_layout.currentWidget() == self.login_widget:
            painter.setOpacity(1.0)
            painter.drawPixmap(self.rect(), self.background_image)
        elif hasattr(self, "tabs") and self.tabs.currentWidget() == self.home_tab:
            painter.setOpacity(1.0)
            painter.drawPixmap(self.rect(), self.background_image)
        else:
            painter.setOpacity(0.08)
            painter.drawPixmap(self.rect(), self.background_image)
        super().paintEvent(event)

    def handle_login(self):
        name = self.name_input.text().strip()
        if name:
            self.user_name = name
            self.stacked_layout.setCurrentWidget(self.main_widget)
        else:
            QMessageBox.warning(self, "Input Required", "Please enter your name to continue.")

    def show_context_help(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.show_help("home")
        elif idx == 1:
            self.show_help("select")
        elif idx == 2:
            self.show_help("quality")
        elif idx == 3:
            self.show_help("preprocess")
        elif idx == 4:
            self.show_help("metrics")
        else:
            self.show_help("home")

    def init_fonts(self):
        font_db = QFontDatabase()
        caveat_font_id = font_db.addApplicationFont(resource_path("Font/Caveat/Caveat-VariableFont_wght.ttf"))
        inter_font_id = font_db.addApplicationFont(resource_path("Font/Inter/Inter-VariableFont_opsz,wght.ttf"))
        self.caveat_font_family = font_db.applicationFontFamilies(caveat_font_id)[0] if caveat_font_id != -1 else "Arial"
        self.inter_font_family = font_db.applicationFontFamilies(inter_font_id)[0] if inter_font_id != -1 else "Arial"
        self.setFont(QFont(self.inter_font_family, 15))

    def init_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#fafbfc"))
        palette.setColor(QPalette.WindowText, QColor("#121212"))
        palette.setColor(QPalette.Base, QColor("#ffffff"))
        palette.setColor(QPalette.AlternateBase, QColor("#f5f5f5"))
        palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
        palette.setColor(QPalette.ToolTipText, QColor("#121212"))
        palette.setColor(QPalette.Text, QColor("#121212"))
        palette.setColor(QPalette.Button, QColor("#f5f5f5"))
        palette.setColor(QPalette.ButtonText, QColor("#121212"))
        palette.setColor(QPalette.BrightText, QColor("#e53935"))
        palette.setColor(QPalette.Highlight, QColor("#e1f1ff"))
        palette.setColor(QPalette.HighlightedText, QColor("#0078d4"))
        self.setPalette(palette)
        self.setStyleSheet(f"""
            QWidget {{ background-color: #fafbfc; color: #121212; font-size: 15px; font-family: '{self.inter_font_family}', Arial, sans-serif; }}
            QLabel {{ color: #121212; font-family: '{self.inter_font_family}', Arial, sans-serif; }}
            QScrollArea {{ background: #fafbfc; }}
            QTableWidget {{ background-color: #fff; color: #121212; border-radius: 4px; font-family: '{self.inter_font_family}', Arial, sans-serif; }}
            QHeaderView::section {{ background-color: #f5f5f5; color: #121212; font-family: '{self.inter_font_family}', Arial, sans-serif; }}
        """)

    def init_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setStyleSheet(TAB_STYLESHEET)
        self.home_tab = QWidget()
        self.select_tab = QWidget()
        self.quality_tab = QWidget()
        self.preprocess_tab = QWidget()
        self.prediction_tab = QWidget()
        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.select_tab, "Select Features")
        self.tabs.addTab(self.quality_tab, "Data Quality")
        self.tabs.addTab(self.preprocess_tab, "Preprocess")
        self.tabs.addTab(self.prediction_tab, "Prediction")

    def set_tab_enabled(self, index, enabled):
        self.tabs.setTabEnabled(index, enabled)

    def set_tab_margin(self, margin):
        self.tabs.setStyleSheet(TAB_STYLESHEET + f"QTabWidget::tab-bar {{ margin-top: {margin}px; }}")

    def init_home_tab(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(
            f"<span style='font-family:\"{self.inter_font_family}\"; font-size:28px; font-weight:bold; color:#121212;'>"
            "TC</span>"
            f"<span style='font-family:\"{self.caveat_font_family}\"; font-size:60px; color: red;'> AI Prediction Tool</span>"
        )
        label.setAlignment(Qt.AlignCenter)
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setAlignment(Qt.AlignCenter)
        open_button = QPushButton("Create a new AI Application")
        open_button.setStyleSheet(BUTTON_STYLESHEET)
        open_button.clicked.connect(self.open_csv)
        button_layout.addWidget(open_button)
        self.generate_prediction_btn = QPushButton("Generate Predictions")
        self.generate_prediction_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.generate_prediction_btn.clicked.connect(self.goto_prediction_tab)
        button_layout.addWidget(self.generate_prediction_btn)
        layout.addWidget(label, alignment=Qt.AlignCenter)
        layout.addWidget(button_container, alignment=Qt.AlignCenter)
        self.home_tab.setLayout(layout)

    def init_select_tab(self):
        layout = QVBoxLayout()
        self.prediction_type_label = QLabel("Select prediction type:")
        self.prediction_type_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.prediction_type_combo = QComboBox()
        self.prediction_type_combo.setStyleSheet(COMBOBOX_STYLESHEET)
        self.prediction_type_combo.addItems(["Yes/No (Classification)", "Number (Regression)"])
        layout.addWidget(self.prediction_type_label)
        layout.addWidget(self.prediction_type_combo)
        self.target_label = QLabel("Select target column:")
        self.target_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.target_combo = QComboBox()
        self.target_combo.setStyleSheet(COMBOBOX_STYLESHEET)
        layout.addWidget(self.target_label)
        layout.addWidget(self.target_combo)
        self.features_label = QLabel("Select feature columns:")
        self.features_label.setStyleSheet("font-weight: bold; font-size: 18px;")
        self.features_list = QListWidget()
        self.features_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.features_label)
        layout.addWidget(self.features_list)
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(BUTTON_STYLESHEET)
        back_btn.clicked.connect(self.goto_home_tab)
        home_btn = QPushButton("Home")
        home_btn.setStyleSheet(BUTTON_STYLESHEET)
        home_btn.clicked.connect(self.goto_home_tab)
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet(BUTTON_STYLESHEET)
        next_btn.clicked.connect(self.goto_quality_tab)
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(home_btn)
        nav_layout.addWidget(next_btn)
        layout.addLayout(nav_layout)
        self.select_tab.setLayout(layout)

    @property
    def prediction_type(self):
        return self.prediction_type_combo.currentText()
    
    def run_data_analyst(self):
        question = self.analyst_question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Required", "Please enter a question about your data.")
            return
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        self.analyst_output.setText("Thinking...")
        QApplication.processEvents()

        try:
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")

            system_prompt = (
                "You are an expert Python data analyst.\n"
                "A CSV file has been loaded into a pandas DataFrame named `df`.\n"
                "Write Python code that answers the user's question using `df`.\n"
                "If the answer requires a graph, generate it using matplotlib (and seaborn if needed).\n"
                "Call plt.show() to display any plots.\n"
                "Assign any non-plot output to a variable called `result`.\n"
                "Do not include print statements or explanations. Just return the code."
            )

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "temperature": 0.3
                },
                timeout=30
            )

            response_json = response.json()

            if response.status_code != 200:
                error_msg = response_json.get("error", {}).get("message", str(response_json))
                self.analyst_output.setText(f"<span style='color:red;'>Groq API error: {error_msg}</span>")
                return

            if "choices" not in response_json:
                self.analyst_output.setText(f"<span style='color:red;'>Unexpected API response: {response_json}</span>")
                return

            code = response_json["choices"][0]["message"]["content"]
            code = code.strip("```python\n").strip("```").strip()

            local_vars = {"df": self.df.copy(), "plt": plt}

            def custom_show():
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close()
                self.analyst_output.append(f'<img src="data:image/png;base64,{img_b64}"/>')

            plt.show = custom_show

            try:
                exec(code, {}, local_vars)
                result = local_vars.get("result", None)
                if result is not None:
                    self.analyst_output.append(f"<pre>{str(result)}</pre>")
            except Exception as e:
                self.analyst_output.append(f"<span style='color:red;'>Error executing code: {e}</span>")

        except Exception as e:
            self.analyst_output.setText(f"Error: {e}")

    def init_quality_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)
        title = QLabel("AI Data Analyst")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        self.analyst_question_input = QLineEdit()
        self.analyst_question_input.setPlaceholderText("Ask a question about your data (e.g. 'Show me a histogram of Age')")
        self.analyst_question_input.setStyleSheet("font-size: 16px; padding: 8px;")
        layout.addWidget(self.analyst_question_input)

        self.analyst_run_btn = QPushButton("Get Analysis")
        self.analyst_run_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.analyst_run_btn.clicked.connect(self.run_data_analyst)
        layout.addWidget(self.analyst_run_btn)

        self.analyst_output = QTextEdit()
        self.analyst_output.setReadOnly(True)
        self.analyst_output.setStyleSheet("font-size: 15px; background: #fff;")
        layout.addWidget(self.analyst_output, stretch=1)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(15)
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(BUTTON_STYLESHEET)
        back_btn.clicked.connect(self.goto_select_tab)
        home_btn = QPushButton("Home")
        home_btn.setStyleSheet(BUTTON_STYLESHEET)
        home_btn.clicked.connect(self.goto_home_tab)
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet(BUTTON_STYLESHEET)
        next_btn.clicked.connect(self.goto_preprocess_tab)
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(home_btn)
        nav_layout.addWidget(next_btn)
        layout.addLayout(nav_layout)
        self.quality_tab.setLayout(layout)

    def init_preprocess_tab(self):
        layout = QVBoxLayout()
        self.preprocess_label = QLabel("Preview of selected features and target column:")
        self.preprocess_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.preprocess_label)
        self.preprocess_preview_table = QTableWidget()
        self.preprocess_preview_table.setMinimumWidth(700)
        self.preprocess_preview_table.setMinimumHeight(300)
        self.preprocess_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preprocess_preview_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.preprocess_preview_table)
        self.apply_preprocessing_btn = QPushButton("Apply Preprocessing (Automatic)")
        self.apply_preprocessing_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.apply_preprocessing_btn.clicked.connect(self.preprocess_selected)
        layout.addWidget(self.apply_preprocessing_btn)
        nav_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet(BUTTON_STYLESHEET)
        back_btn.clicked.connect(self.goto_quality_tab)
        home_btn = QPushButton("Home")
        home_btn.setStyleSheet(BUTTON_STYLESHEET)
        home_btn.clicked.connect(self.goto_home_tab)
        nav_layout.addWidget(back_btn)
        nav_layout.addWidget(home_btn)
        layout.addLayout(nav_layout)
        self.preprocess_tab.setLayout(layout)

    def populate_preprocess_preview(self):
        features = [item.text() for item in self.features_list.selectedItems()]
        target = self.target_combo.currentText()
        all_columns = features.copy()
        if target and target not in all_columns:
            all_columns.append(target)
        preview_df = self.df[all_columns].head(10)
        self.preprocess_preview_table.setRowCount(len(preview_df))
        self.preprocess_preview_table.setColumnCount(len(all_columns))
        self.preprocess_preview_table.setHorizontalHeaderLabels(all_columns)
        for i in range(len(preview_df)):
            for j, col in enumerate(all_columns):
                val = str(preview_df.iloc[i, j])
                self.preprocess_preview_table.setItem(i, j, QTableWidgetItem(val))
        self.preprocess_preview_table.resizeColumnsToContents()

    def goto_preprocess_tab(self):
        self.set_tab_enabled(3, True)
        self.tabs.setCurrentIndex(3)
        self.set_tab_margin(0)
        self.populate_preprocess_preview()

    def preprocess_selected(self):
        selected_features = [item.text() for item in self.features_list.selectedItems()]
        target_column = self.target_combo.currentText()
        if not selected_features or not target_column:
            QMessageBox.warning(self, "Warning", "Please select features and target.")
            return
        df = self.df[selected_features + [target_column]].copy()
        
        threshold = int(0.5 * len(df.columns))
        initial_row_count = len(df)
        df_cleaned = df[df.isnull().sum(axis=1) <= threshold].copy()
        removed_rows = initial_row_count - len(df_cleaned)
        if removed_rows > 0:
            QMessageBox.information(self, "Rows Removed", f"{removed_rows} rows were removed due to having more than 50% missing values.")
  
        for col in df_cleaned.columns:
            missing_flag = f"{col}_missing"
            if pd.api.types.is_numeric_dtype(df_cleaned[col]) or pd.api.types.is_float_dtype(df_cleaned[col]):
                df_cleaned[missing_flag] = df_cleaned[col].isnull().astype(int)
                df_cleaned[col] = df_cleaned[col].fillna(-1)
            else:
                df_cleaned[missing_flag] = df_cleaned[col].isnull().astype(int)
                df_cleaned[col] = df_cleaned[col].fillna('missing')
        try:
            processed_data, encoders, processed_cols = auto_preprocess_data(df_cleaned, target_column)
            self.processed_data = processed_data
            self.encoders = encoders
            self.selected_features_after_preproc = [col for col in processed_cols if col != target_column]
            self.target_column_after_preproc = target_column
            self.apply_preprocessing_btn.hide()
            self.preprocess_label.setText("Preprocessing complete! Proceeding to model training...")
            QTimer.singleShot(1200, self.show_model_training_panel)
        except Exception as e:
            log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Error", f"Failed to preprocess data:\n{e}")

    def show_model_training_panel(self):
        self.start_model_training()

    def start_model_training(self):
        encoder_name, ok = QInputDialog.getText(
            self, "Save Encoders",
            "Enter a name for your encoder file (no extension)\nThis saves how your data needs to be preprocessed\n⚠️THIS WILL BE NEEDED WHEN TESTING!!⚠️"
        )
        if ok and encoder_name:
            enc_dir = get_writable_path(os.path.join("encoders", ""))
            if not os.path.exists(enc_dir):
                os.makedirs(enc_dir)
            file_path = get_writable_path(os.path.join("encoders", f"{encoder_name}.pkl"))
            with open(file_path, "wb") as f:
                pickle.dump({
                    'encoders': self.encoders,
                    'features': self.selected_features_after_preproc,
                    'target': self.target_column_after_preproc,
                    'initial_features': getattr(self, 'initial_selected_features', self.selected_features_after_preproc)
                }, f)
            QMessageBox.information(
                self,
                "Success",
                f"Preprocessing complete and pipeline saved!\n\nFile: {file_path}\n\n"
                "Remember the file name, you'll need it when generating predictions."
            )
        x = self.processed_data[self.selected_features_after_preproc]
        y = self.processed_data[self.target_column_after_preproc]
        prediction_type = "classification" if self.prediction_type.startswith("Yes/No") else "regression"
        self.progress_dialog = QProgressDialog("Training models...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Training in Progress")
        self.progress_dialog.setWindowModality(True)
        self.progress_dialog.setValue(0)
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.show()
        self.worker = ModelTrainingWorker(x, y, prediction_type=prediction_type, train_func=auto_train_and_evaluate_models)
        self.worker.progress.connect(self.update_training_progress)
        self.worker.finished.connect(self.training_finished)
        self.worker.start()

    def update_training_progress(self, percent, msg):
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(f"Training models... {msg}")

    def training_finished(self, results, trained_models):
        self.progress_dialog.close()
        self.show_model_selection(results, trained_models)

    def show_model_selection(self, results, trained_models):
        self.preprocess_label.hide()
        self.apply_preprocessing_btn.hide()
        layout = self.preprocess_tab.layout()
        if not hasattr(self, 'model_select_label'):
            self.model_select_label = QLabel("Choose which model you'd like to save for your future predictions for this application.")
            self.model_select_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
            layout.insertWidget(0, self.model_select_label)
        else:
            self.model_select_label.setText("Choose which model you'd like to save for your future predictions for this application.")
            self.model_select_label.show()
        if not hasattr(self, 'model_results_table'):
            self.model_results_table = QTableWidget()
            self.model_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.model_results_table.verticalHeader().setVisible(False)
            self.model_results_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.model_results_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.model_results_table.cellClicked.connect(self.update_selected_model_index)
            layout.insertWidget(1, self.model_results_table)

        prediction_type = "classification" if self.prediction_type.startswith("Yes/No") else "regression"

        if prediction_type == "classification":
            self.model_results_table.setColumnCount(4)
            self.model_results_table.setHorizontalHeaderLabels(['Model', 'Accuracy (%)', 'F1 Score (%)', 'AUC (%)'])
        else:
            self.model_results_table.setColumnCount(3)
            self.model_results_table.setHorizontalHeaderLabels(['Model', 'R² Score', 'MSE'])

        self.model_results_table.setRowCount(len(results)) 
        if prediction_type == "classification":
            for i, res in enumerate(results):
                self.model_results_table.setItem(i, 0, QTableWidgetItem(res['name']))
                self.model_results_table.setItem(i, 1, QTableWidgetItem(f"{res['accuracy']*100:.2f}"))
                self.model_results_table.setItem(i, 2, QTableWidgetItem(f"{res['f1']*100:.2f}"))
                auc_str = f"{res['auc']*100:.2f}" if res['auc'] is not None else "-"
                self.model_results_table.setItem(i, 3, QTableWidgetItem(auc_str))
        else:
            for i, res in enumerate(results):
                self.model_results_table.setItem(i, 0, QTableWidgetItem(res['name']))
                self.model_results_table.setItem(i, 1, QTableWidgetItem(f"{res['r2']:.4f}"))
                self.model_results_table.setItem(i, 2, QTableWidgetItem(f"{res['mse']:.4f}"))

        self.model_results_table.show()
        if not hasattr(self, 'save_model_btn'):
            self.save_model_btn = QPushButton("Save Selected Model")
            self.save_model_btn.setStyleSheet(BUTTON_STYLESHEET)
            self.save_model_btn.clicked.connect(self.save_selected_model)
            nav_layout_index = layout.count() - 1
            layout.insertWidget(nav_layout_index, self.save_model_btn)
        self.trained_models = trained_models
        self.results = results
        self.selected_model_index = 0
        if len(results) > 0:
            self.model_results_table.selectRow(0)

    def update_selected_model_index(self, row, col):
        self.selected_model_index = row

    def save_selected_model(self):
        idx = self.selected_model_index
        res = self.results[idx]
        model = self.trained_models[res['name']]
        model_name, ok = QInputDialog.getText(self, "Save Model", "Enter a name for your model file (no extension):")
        if ok and model_name:
            file_path = save_model(model, model_name, get_writable_path)
            QMessageBox.information(
                self,
                "Success",
                f"Model saved!\n\nFile: {file_path}\n\n"
                "Now you can use your saved model to generate predictions on new data for this application."
            )
            self.goto_prediction_tab()

    def restart_app(self):
        QTimer.singleShot(100, self._do_restart)

    def _do_restart(self):
        self.close()
        self.__class__._new_window = BespokePredictionApp()
        self.__class__._new_window.show()

    def init_prediction_tab(self):
        layout = QVBoxLayout()
        self.pred_summary_label = QLabel()
        layout.addWidget(self.pred_summary_label)
        self.pred_upload_test_btn = QPushButton("Upload Test File")
        self.pred_upload_test_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.pred_upload_test_btn.clicked.connect(self.pred_upload_test_file)
        layout.addWidget(self.pred_upload_test_btn)
        self.pred_load_encoder_btn = QPushButton("Load Preprocessor")
        self.pred_load_encoder_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_load_encoder_btn.clicked.connect(self.pred_select_encoder_file)
        layout.addWidget(self.pred_load_encoder_btn)
        self.pred_load_model_btn = QPushButton("Load Model")
        self.pred_load_model_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_load_model_btn.clicked.connect(self.pred_select_model_file)
        layout.addWidget(self.pred_load_model_btn)
        self.pred_save_btn = QPushButton("Download Predictions as CSV")
        self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_save_btn.clicked.connect(self.pred_save_predictions)
        layout.addWidget(self.pred_save_btn)
        self.pred_table = QTableWidget()
        layout.addWidget(self.pred_table)
        self.prediction_tab.setLayout(layout)

    def reset_prediction_tab(self):
        self.prediction_help_state = "default"
        self.pred_summary_label.setText("")
        self.pred_upload_test_btn.setEnabled(True)
        self.pred_upload_test_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.pred_load_encoder_btn.setEnabled(False)
        self.pred_load_encoder_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_load_model_btn.setEnabled(False)
        self.pred_load_model_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_save_btn.setEnabled(False)
        self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_table.clear()
        self.pred_table.setRowCount(0)
        self.pred_table.setColumnCount(0)
        self.prediction_test_df = None
        self.prediction_processed_x = None
        self.prediction_encoders = None
        self.prediction_features = None
        self.prediction_target = None

    def pred_upload_test_file(self):
        test_file, _ = QFileDialog.getOpenFileName(self, "Open Test CSV File", "", "CSV Files (*.csv)")
        if not test_file:
            return
        try:
            test_df = pd.read_csv(test_file)
            self.prediction_test_df = test_df
        except Exception as e:
            log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Error", f"Failed to load test file:\n{e}")
            return

        summary = f"<b>File:</b> {test_file}<br>"
        summary += f"<b>Shape:</b> {test_df.shape[0]} rows, {test_df.shape[1]} columns<br><br>"
        used_cols = list(test_df.columns)
        summary += "<b>Preview:</b><br>"
        summary += test_df[used_cols].head(5).to_html(index=False)
        self.pred_summary_label.setText(summary)
        self.pred_load_encoder_btn.setEnabled(True)
        self.pred_load_encoder_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
        self.pred_load_model_btn.setEnabled(False)
        self.pred_load_model_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_save_btn.setEnabled(False)
        self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.prediction_processed_x = None

    def pred_select_encoder_file(self):
        from core.preprocess import preprocess_test_data
        enc_dir = get_writable_path(os.path.join("encoders", ""))
        encoder_file, _ = QFileDialog.getOpenFileName(self, "Select Encoder File", enc_dir, "Pickle Files (*.pkl)")
        if not encoder_file:
            return
        try:
            with open(encoder_file, "rb") as f:
                enc_data = pickle.load(f)
            encoders = enc_data['encoders']
            features = enc_data['features']
            target = enc_data['target']
            initial_features = enc_data.get('initial_features', features)
            test_df = self.prediction_test_df.copy()

            threshold = int(0.5 * len(test_df.columns))
            initial_row_count = len(test_df)
            test_df_cleaned = test_df[test_df.isnull().sum(axis=1) <= threshold].copy()
            removed_rows = initial_row_count - len(test_df_cleaned)
            if removed_rows > 0:
                QMessageBox.information(self, "Rows Removed", f"{removed_rows} rows were removed from your test data due to having more than 50% missing values.")

            processed_test_df = preprocess_test_data(test_df_cleaned, encoders, features, target_col=target)

            self.prediction_features = features
            self.prediction_target = target
            self.prediction_encoders = encoders
            self.prediction_processed_x = processed_test_df
            QMessageBox.information(self, "Success", "Test data preprocessed successfully!\nYou can now load a model and generate predictions.")
            self.pred_load_model_btn.setEnabled(True)
            self.pred_load_model_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
            self.pred_save_btn.setEnabled(False)
            self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        except Exception as e:
            log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Error", f"Failed to preprocess test data:\n{e}")

    def pred_select_model_file(self):
        model_dir = get_writable_path(os.path.join("models", ""))
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", model_dir, "Pickle Files (*.pkl)")
        if not model_file:
            return
        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Error", f"Failed to load model file:\n{e}")
            return
        try:
            prediction_type = "classification" if self.prediction_type.startswith("Yes/No") else "regression"
            preds = model.predict(self.prediction_processed_x)

            result_df = self.prediction_test_df.copy()

            if prediction_type == "classification":
                target_encoder = None
                class_names = None
                if self.prediction_encoders and self.prediction_target in self.prediction_encoders:
                    target_encoder = self.prediction_encoders[self.prediction_target]
                if target_encoder and hasattr(target_encoder, "inverse_transform"):
                    try:
                        preds = target_encoder.inverse_transform(preds)
                        if hasattr(target_encoder, "classes_"):
                            class_names = target_encoder.classes_
                    except Exception:
                        pass
            result_df["Prediction"] = preds

            if prediction_type == "classification" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(self.prediction_processed_x)
                if class_names is None and hasattr(model, "classes_"):
                    class_names = model.classes_
                if class_names is not None:
                    if len(class_names) == 2:
                        pos_class = class_names[1]
                        result_df[f"Probability_{pos_class} (%)"] = (proba[:, 1] * 100).round(2)
                    else:
                        for idx, cname in enumerate(class_names):
                            result_df[f"Probability_{cname} (%)"] = (proba[:, idx] * 100).round(2)
                else:
                    for i in range(proba.shape[1]):
                        result_df[f"Probability_Class_{i} (%)"] = (proba[:, i] * 100).round(2)

            self.prediction_result_df = result_df
            self.show_predictions_table(result_df.head(100))
            self.pred_save_btn.setEnabled(True)
            self.pred_save_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
        except Exception as e:
            log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Error", f"Failed to generate predictions:\n{e}")

    def show_predictions_table(self, df):
        self.prediction_help_state = "predictions_made"
        layout = self.prediction_tab.layout()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            elif item.layout() is not None:
                while item.layout().count():
                    subitem = item.layout().takeAt(0)
                    subwidget = subitem.widget()
                    if subwidget is not None:
                        subwidget.setParent(None)
        extra_col_widget = QWidget()
        extra_col_widget.setObjectName("extra_col_widget")
        extra_layout = QHBoxLayout(extra_col_widget)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(10)
        extra_label = QLabel("Add columns from your test data:")
        extra_layout.addWidget(extra_label)

        available_cols = []
        if hasattr(self, "prediction_test_df") and self.prediction_test_df is not None:
            already_shown = set(df.columns)
            available_cols = [col for col in self.prediction_test_df.columns if col not in already_shown]

        select_btn = QPushButton("Select Columns...")
        select_btn.setStyleSheet(BUTTON_STYLESHEET)
        def open_col_dialog():
            dialog = ColumnSelectDialog(available_cols, self)
            if dialog.exec_() == QDialog.Accepted:
                selected_cols = dialog.selected_columns()
                for col in selected_cols:
                    if col in self.prediction_test_df.columns:
                        self.prediction_result_df[col] = self.prediction_test_df[col].values
                self.show_predictions_table(self.prediction_result_df.head(100))
        select_btn.clicked.connect(open_col_dialog)
        extra_layout.addWidget(select_btn)
        extra_layout.addStretch()
        layout.addWidget(extra_col_widget)

        table_scroll = QScrollArea()
        table_scroll.setWidgetResizable(True)
        table_scroll.setStyleSheet("QScrollArea {background: #fafbfc; border-radius: 10px;}")
        self.pred_table = QTableWidget()
        self.pred_table.setRowCount(len(df))
        self.pred_table.setColumnCount(len(df.columns))
        self.pred_table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        for i in range(len(df)):
            for j in range(len(df.columns)):
                val = str(df.iloc[i, j])
                self.pred_table.setItem(i, j, QTableWidgetItem(val))
        self.pred_table.resizeColumnsToContents()
        table_scroll.setWidget(self.pred_table)
        layout.addWidget(table_scroll, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 10, 0, 10)
        btn_layout.addStretch()
        self.pred_save_btn = QPushButton("Download Predictions as CSV")
        self.pred_save_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
        self.pred_save_btn.clicked.connect(self.pred_save_predictions)
        btn_layout.addWidget(self.pred_save_btn)
        home_btn = QPushButton("Home")
        home_btn.setStyleSheet(BUTTON_STYLESHEET)
        home_btn.clicked.connect(self.restart_app)
        btn_layout.addWidget(home_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def add_extra_column_to_predictions(self):
        if not hasattr(self, "prediction_test_df") or self.prediction_test_df is None:
            return
        selected_items = self.extra_col_list.selectedItems()
        selected_cols = [item.text() for item in selected_items]
        for col in selected_cols:
            if col in self.prediction_test_df.columns:
                self.prediction_result_df[col] = self.prediction_test_df[col].values
        self.show_predictions_table(self.prediction_result_df.head(100))

    def pred_save_predictions(self):
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Predictions As", "", "CSV Files (*.csv)")
        if out_path:
            self.prediction_result_df.to_csv(out_path, index=False)
            QMessageBox.information(self, "Success", f"Predictions saved to:\n{out_path}")

    def show_help(self, page):
        QMessageBox.information(self, "Help", self.get_help_content(page))

    def get_help_content(self, page):
        if page == "home":
            return (
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                "Quick Instructions:</span><br>"
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                "1. If you are generating predictions for a new application you need to first upload training data.<br>"
                "2. Select the features and target column.<br>"
                "3. Review data quality and fix any issues.<br>"
                "4. Preprocessing is now automatic.<br>"
                "5. Train the model and save the encoder.<br>"
                "6. Then return to home to upload the saved encoder and saved model to generate the predictions.</span>"
            )
        elif page == "select":
            return (
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                "Feature Selection Help:</span><br>"
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                "Select the columns you want to use to make predictions. Choose columns that can affect the outcome. "
                "Avoid using unique IDs like Contact Ref or Invoice Ref, as these do not help in prediction. "
                "Also, select the column you want to predict as your target column.</span>"
            )
        elif page == "quality":
            return (
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                "Data Quality Help:</span><br>"
                f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                "This page shows you the quality of your data. Good quality data is important for accurate predictions. "
                "If your data has missing values or your target column is imbalanced (one value appears much more than others), "
                "the results may not be reliable. Please review and fix any issues before proceeding.</span>"
            )
        elif page == "preprocess":
            if hasattr(self, 'model_results_table') and self.model_results_table.isVisible():
                return (
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                    "Model Evaluation Help:</span><br>"
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                    "<b>Accuracy</b>: The percentage of correct predictions.<br>"
                    "<b>F1</b>: A balance between how many correct positive results and how many were missed.<br>"
                    "<b>AUC</b>: Shows how well the model separates different outcomes.<br>"
                    "Higher values are better. After saving your model, return to the Home page to generate predictions for your data.</span>"
                )
            else:
                return (
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                    "Preprocessing Help:</span><br>"
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                    "Preprocessing is now automatic. The system will intelligently scale numeric columns and encode categorical columns.<br>"
                    "After reviewing your selected features and target, click 'Apply Preprocessing' to continue.</span>"
                )
        elif page == "metrics" or page == "prediction":
            if getattr(self, "prediction_help_state", "default") == "predictions_made":
                return (
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                    "Prediction Results Help:</span><br>"
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                    "These are the predictions for all the records you entered. "
                    "Some values may be missing if predictions could not be made for them, for example due to missing values in required columns. "
                    "You can also add other columns from your original test data to the results using the button above.</span>"
                )
            else:
                return (
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:22px; font-weight:bold;'>"
                    "Prediction Help:</span><br>"
                    f"<span style='font-family:\"{self.inter_font_family}\"; font-size:20px;'>"
                    "Upload your test data, load your saved preprocessor and model, and generate predictions. "
                    "You can add extra columns from your test data to the results as needed.</span>"
                )
        else:
            return "No help available for this page."

    def goto_home_tab(self, confirm=True):
        if confirm:
            reply = QMessageBox.question(
                self,
                "Confirm Reset",
                "Are you sure you want to return to Home? This will reset the tool and clear all progress.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        self.restart_app()

    def goto_select_tab(self):
        self.tabs.setCurrentIndex(1)
        self.set_tab_margin(0)
        self.set_tab_enabled(2, False)
        self.set_tab_enabled(3, False)
        self.selected_features = []

    def goto_quality_tab(self):
        self.set_tab_margin(0)
        self.selected_features = [item.text() for item in self.features_list.selectedItems()]
        if not self.selected_features:
            QMessageBox.warning(self, "Warning", "Please select at least one feature column.")
            return
        if not self.target_combo.currentText():
            QMessageBox.warning(self, "Warning", "Please select a target column.")
            return
        self.set_tab_enabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.set_tab_enabled(3, False)

    def goto_preprocess_tab(self):
        self.set_tab_enabled(3, True)
        self.tabs.setCurrentIndex(3)
        self.set_tab_margin(0)
        self.populate_preprocess_preview()

    def goto_prediction_tab(self):
        self.tabs.setCurrentWidget(self.prediction_tab)
        self.set_tab_margin(0)
        self.reset_prediction_tab()

    def open_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_name:
            try:
                self.df = pd.read_csv(file_name)
                columns = list(self.df.columns)
                self.features_list.clear()
                self.target_combo.clear()
                for col in columns:
                    self.features_list.addItem(col)
                    self.target_combo.addItem(col)
                self.set_tab_enabled(1, True)
                self.set_tab_enabled(2, False)
                self.set_tab_enabled(3, False)
                QMessageBox.information(self, "Success", "File loaded! Now select features and target.")
                self.goto_select_tab()
            except Exception as e:
                log_error_to_neon(getattr(self, "user_name", None), type(e), e, sys.exc_info()[2])
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    