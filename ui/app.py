from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QMessageBox, QListWidget, QAbstractItemView, QComboBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QInputDialog, QProgressDialog, QHeaderView, QScrollArea, QDialog,
    QLineEdit, QStackedLayout, QTextEdit, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont, QFontDatabase, QPixmap, QPainter, QIcon
from core.config import Config
import requests
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ui.styles import BUTTON_STYLESHEET, TAB_STYLESHEET, COMBOBOX_STYLESHEET, PREDICTION_ENABLED_STYLESHEET, PREDICTION_DISABLED_STYLESHEET
from ui.dialogs import ColumnSelectDialog
from core.error_logger import log_error_to_file, get_error_logger
from ui.error_dialog import show_error_dialog
from core.resource_path import resource_path, get_writable_path
from core.model import auto_train_and_evaluate_models, save_model
from core.preprocess import auto_preprocess_data
from core.model_worker import ModelTrainingWorker
import sys
from dotenv import load_dotenv
import io
import base64
import pandas as pd
import matplotlib

from core.validators import validate_dataframe

# CRITICAL FIX: Set matplotlib backend before any other imports
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import signal

class BespokePredictionApp(QWidget):
    def __init__(self):
        """
        Initialises the application.

        Sets up the application window, its title and icon, and creates the layout
        for the login page and main page. The login page is shown by default.

        :return: None
        """
        super().__init__()
        self.user_name = None
        self.prediction_help_state = "default"
        self.setWindowTitle("TC AI Prediction Tool")
        self.setWindowIcon(QIcon(resource_path("Images/icon.png")))
        self.df = None
        self.selected_features = []
        self.background_image = QPixmap(resource_path("Images/wire-svg.svg"))
        
        # Initialize error logging with configurable admin email
        self.admin_email = Config.admin.ADMIN_EMAIL  # Configure this as needed
        self.error_logger = get_error_logger(self.admin_email)

        self.stacked_layout = QStackedLayout(self)
        self.init_login_page()
        self.main_widget = QWidget()
        self.init_main_ui(self.main_widget)
        self.stacked_layout.addWidget(self.login_widget)
        self.stacked_layout.addWidget(self.main_widget)
        self.setLayout(self.stacked_layout)
        self.stacked_layout.setCurrentWidget(self.login_widget)

    def init_main_ui(self, main_widget):
        """
        Initialises the main application UI.

        Sets up the main application layout, including the tabs and a help button.

        :param main_widget: The main application widget.
        :type main_widget: QWidget
        :return: None
        """
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
        """
        Initializes the login page UI.

        Sets up the layout and style for the login page, including an icon, 
        an input field for the user to enter their name, and a button to proceed. 
        The login button is connected to the handle_login method to handle user input.

        :return: None
        """
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
        """
        Custom paint event to draw the background image with varying opacity based on the current widget.

        This method overrides the default paint event to render a background image with full opacity 
        when the current widget is the login or home tab, and reduced opacity otherwise.

        :param event: The paint event object containing details about the repaint request.
        :type event: QPaintEvent
        """
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

    def handle_error_with_popup(self, exc_type: type, exc_value: Exception, tb):
        """Handle errors by logging to file and showing popup dialog."""
        try:
            # Log error to file
            error_details = log_error_to_file(
                self.user_name or "Unknown", 
                exc_type, 
                exc_value, 
                tb, 
                self.admin_email
            )
            
            # Show error popup dialog
            show_error_dialog(
                error_details,
                self.error_logger.get_log_file_path(),
                self.admin_email,
                self
            )
        except Exception as e:
            # Fallback error handling
            QMessageBox.critical(
                self, 
                "Critical Error", 
                f"A critical error occurred:\n{str(exc_value)}\n\nAdditional error in error handling: {str(e)}"
            )

    def handle_login(self):
        """
        Handle the login button click by validating the input name and switching to the main widget if valid.

        :return: None
        """
        name = self.name_input.text().strip()
        if name:
            self.user_name = name
            self.stacked_layout.setCurrentWidget(self.main_widget)
        else:
            QMessageBox.warning(self, "Input Required", "Please enter your name to continue.")

    def show_context_help(self):
        """
        Show context-sensitive help based on the current tab index.

        :return: None
        """
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
        """
        Initialize the fonts used in the application by loading the Caveat and Inter variable fonts and setting the
        default font to Inter.

        :return: None
        """
        font_db = QFontDatabase()
        caveat_font_id = font_db.addApplicationFont(resource_path("Font/Caveat/Caveat-VariableFont_wght.ttf"))
        inter_font_id = font_db.addApplicationFont(resource_path("Font/Inter/Inter-VariableFont_opsz,wght.ttf"))
        self.caveat_font_family = font_db.applicationFontFamilies(caveat_font_id)[0] if caveat_font_id != -1 else "Arial"
        self.inter_font_family = font_db.applicationFontFamilies(inter_font_id)[0] if inter_font_id != -1 else "Arial"
        self.setFont(QFont(self.inter_font_family, 15))

    def init_palette(self):
        """
        Initialize the application's color palette.

        Set the application's palette to a custom color scheme.

        Also set a global stylesheet for all widgets to use the Inter font family and a font size of 15px. Labels and
        table headers are also given a font family of Inter.

        :return: None
        """
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
        """
        Initialize the application's tabs.

        Creates a QTabWidget and sets up 5 tabs: Home, Select Features, Data Quality, Preprocess, and Prediction.
        Each tab is initially set to an empty QWidget, and the tabs are added to the QTabWidget in the correct order.

        :return: None
        """
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
        """
        Initializes the home tab UI.

        Sets up the layout and style for the home tab, including the application title, 
        a button to open a CSV file, and a button to generate predictions. The open button 
        is connected to the open_csv method to handle user input, and the generate predictions 
        button is connected to the goto_prediction_tab method to handle user input.

        :return: None
        """
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
        """
        Initializes the select tab UI.

        Sets up the layout and style for the select tab, allowing the user to choose
        the prediction type, target column, and feature columns. The navigation buttons
        allow the user to go back to the home tab, or proceed to the quality tab.

        :return: None
        """
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
    
    def run_data_quality_analysis(self):
        """
        Generate comprehensive business-friendly data quality report with interactive visualizations.
        """
        if self.df is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        try:
            from core.data_quality import InteractiveBusinessDataQuality
            
            self.analyst_output.setText("Analyzing your data quality... This may take a moment.")
            QApplication.processEvents()

            # Create business-friendly data quality analyzer
            self.data_quality = InteractiveBusinessDataQuality(self.df)
            
            # Display business-friendly summary report
            summary_report = self.data_quality.get_business_summary()
            self.analyst_output.setPlainText(summary_report)
            
            # Add chart selection and export buttons
            self._add_chart_selection_buttons()
            self._add_export_button()
            
        except Exception as e:
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            self.analyst_output.setText(f"Error during data quality analysis: {str(e)}")
    
    def _add_chart_selection_buttons(self):
        """Add interactive chart selection buttons to the quality tab."""
        # Clear existing chart buttons if any
        if hasattr(self, 'chart_buttons_widget'):
            self.chart_buttons_widget.setParent(None)
        
        # Create chart selection widget
        self.chart_buttons_widget = QWidget()
        chart_layout = QHBoxLayout(self.chart_buttons_widget)
        chart_layout.setContentsMargins(0, 10, 0, 10)
        
        chart_label = QLabel("View Charts:")
        chart_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        chart_layout.addWidget(chart_label)
        
        # Get available chart options
        chart_options = self.data_quality.get_chart_options()
        
        for chart_key, chart_name in chart_options.items():
            if chart_key != 'overview':  # Skip overview as it's text-based
                btn = QPushButton(chart_name)
                btn.setStyleSheet(BUTTON_STYLESHEET)
                btn.clicked.connect(lambda checked, key=chart_key: self._show_chart(key))
                chart_layout.addWidget(btn)
        
        chart_layout.addStretch()
        
        # Add to quality tab layout
        quality_layout = self.quality_tab.layout()
        quality_layout.insertWidget(quality_layout.count() - 1, self.chart_buttons_widget)
    
    def _show_chart(self, chart_type: str):
        """Display selected chart type."""
        try:
            if not hasattr(self, 'data_quality'):
                QMessageBox.warning(self, "No Analysis", "Please run data quality analysis first.")
                return
            
            # Create the chart
            fig = self.data_quality.create_chart(chart_type)
            if fig is None:
                QMessageBox.warning(self, "Chart Error", f"Could not create {chart_type} chart.")
                return
            
            # Create a new dialog to display the chart
            from PyQt5.QtWidgets import QDialog, QVBoxLayout
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Data Quality - {self.data_quality.get_chart_options()[chart_type]}")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Add matplotlib canvas
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Add close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet(BUTTON_STYLESHEET)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec_()
            
        except Exception as e:
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Chart Error", f"Error displaying chart: {str(e)}")
    
    def _add_export_button(self):
        """Add export functionality button to the quality tab."""
        # Create export button widget
        if hasattr(self, 'export_button_widget'):
            self.export_button_widget.setParent(None)
        
        self.export_button_widget = QWidget()
        export_layout = QHBoxLayout(self.export_button_widget)
        export_layout.setContentsMargins(0, 10, 0, 10)
        
        export_label = QLabel("Export Report:")
        export_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        export_layout.addWidget(export_label)
        
        # Export to Excel button
        export_btn = QPushButton("Save Detailed Report (Excel)")
        export_btn.setStyleSheet(BUTTON_STYLESHEET)
        export_btn.clicked.connect(self._export_data_quality_report)
        export_layout.addWidget(export_btn)
        
        export_layout.addStretch()
        
        # Add to quality tab layout
        quality_layout = self.quality_tab.layout()
        quality_layout.insertWidget(quality_layout.count() - 1, self.export_button_widget)
    
    def _export_data_quality_report(self):
        """Export comprehensive data quality report with highlighted issues."""
        try:
            if not hasattr(self, 'data_quality'):
                QMessageBox.warning(self, "No Analysis", "Please run data quality analysis first.")
                return
            
            # Get save location from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Data Quality Report",
                f"data_quality_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "Excel files (*.xlsx);;All files (*.*)"
            )
            
            if file_path:
                # Show progress
                progress = QProgressDialog("Generating comprehensive report...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                progress.setValue(25)
                QApplication.processEvents()
                
                # Export the report
                success = self.data_quality.export_business_report(file_path)
                
                progress.setValue(100)
                progress.close()
                
                if success:
                    # Show success message with details
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Export Successful")
                    msg.setText("Data Quality Report exported successfully!")
                    msg.setInformativeText(
                        f"Report saved to: {file_path}\n\n"
                        "The Excel file contains:\n"
                        "• Executive Summary with key metrics\n"
                        "• Your data with issues highlighted in color\n"
                        "• Detailed analysis and recommendations\n\n"
                        "Legend:\n"
                        "🔴 Red cells = Missing data\n"
                        "🟡 Yellow cells = Unusual values\n"
                        "🔵 Blue rows = Duplicate records"
                    )
                    msg.exec_()
                else:
                    QMessageBox.warning(self, "Export Failed", 
                                      "Failed to export the report. Please check file permissions and try again.")
                
        except Exception as e:
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.warning(self, "Export Error", f"Error exporting report: {str(e)}")

    def init_quality_tab(self):
        """
        Initializes the quality tab UI.

        Sets up the layout and style for the quality tab with data quality analysis
        functionality that provides comprehensive reporting and interactive visualizations.

        :return: None
        """
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)
        
        title = QLabel("Data Quality Analysis")
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # Description
        description = QLabel("Get a clear, business-friendly assessment of your data's health and reliability. We'll identify missing information, inconsistencies, and unusual patterns that could affect your analysis - all explained in plain language with actionable recommendations.")
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 14px; color: #666; margin-bottom: 15px;")
        layout.addWidget(description)

        # Analysis button
        self.analyst_run_btn = QPushButton("Check My Data Health")
        self.analyst_run_btn.setStyleSheet(BUTTON_STYLESHEET)
        self.analyst_run_btn.clicked.connect(self.run_data_quality_analysis)
        layout.addWidget(self.analyst_run_btn)

        # Output area for text report
        self.analyst_output = QTextEdit()
        self.analyst_output.setReadOnly(True)
        self.analyst_output.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 13px; 
                background: #f8f9fa; 
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.analyst_output.setPlaceholderText("Click 'Check My Data Health' to get a business-friendly assessment of your data quality with clear recommendations and export options...")
        layout.addWidget(self.analyst_output, stretch=1)

        # Navigation buttons
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
        """Preprocess selected features and target with enhanced error handling."""
        try:
            from core.validators import validate_features, validate_target_column
            from core.errors import PreprocessingError, ValidationError
            from core.logging_setup import get_logger
            
            logger = get_logger(__name__, getattr(self, "user_name", None))
            logger.info("Starting data preprocessing")
            
            selected_features = [item.text() for item in self.features_list.selectedItems()]
            target_column = self.target_combo.currentText()
            
            if not selected_features or not target_column:
                QMessageBox.warning(self, "Warning", "Please select features and target.")
                return
            
            # Validate selections
            validate_features(self.df, selected_features)
            validate_target_column(self.df, target_column)
            
            # Use view instead of copy for initial selection to save memory
            df_subset = self.df[selected_features + [target_column]]
            
            # Check for sufficient data
            if len(df_subset) < 10:
                raise ValidationError("Dataset too small for training (minimum 10 rows required)")
            
            threshold = int(0.5 * len(df_subset.columns))
            initial_row_count = len(df_subset)
            
            # Only create copy when we need to modify the data
            missing_mask = df_subset.isnull().sum(axis=1) <= threshold
            df_cleaned = df_subset[missing_mask].copy()
            removed_rows = initial_row_count - len(df_cleaned)
            
            if len(df_cleaned) < 10:
                raise ValidationError("Too many rows with missing values. Dataset too small after cleaning.")
            
            if removed_rows > 0:
                QMessageBox.information(self, "Rows Removed", f"{removed_rows} rows were removed due to having more than 50% missing values.")
      
            for col in df_cleaned.columns:
                missing_flag = f"{col}_missing"
                if pd.api.types.is_numeric_dtype(df_cleaned[col]) or pd.api.types.is_float_dtype(df_cleaned[col]):
                    df_cleaned[missing_flag] = df_cleaned[col].isnull().astype(int)
                    df_cleaned[col] = df_cleaned[col].fillna(-1)
                else:
                    df_cleaned[missing_flag] = df_cleaned[col].isnull().astype(int)
                    
                    if pd.api.types.is_categorical_dtype(df_cleaned[col]):
                        if 'missing' not in df_cleaned[col].cat.categories:
                            df_cleaned[col] = df_cleaned[col].cat.add_categories(['missing'])
                    
                    df_cleaned[col] = df_cleaned[col].fillna('missing')
            
            processed_data, encoders, processed_cols, feature_to_base = auto_preprocess_data(df_cleaned, target_column)
            self.feature_to_base = feature_to_base

            # Save processed_cols as usual
            self.selected_features_after_preproc = [col for col in processed_cols if col != target_column]
            # Validate preprocessing results
            if processed_data.empty:
                raise PreprocessingError("Preprocessing resulted in empty dataset")
            
            if len(processed_cols) == 0:
                raise PreprocessingError("No features available after preprocessing")
            
            self.processed_data = processed_data
            self.encoders = encoders
            self.selected_features_after_preproc = [col for col in processed_cols if col != target_column]
            self.target_column_after_preproc = target_column
            self.apply_preprocessing_btn.hide()
            self.preprocess_label.setText("Preprocessing complete! Proceeding to model training...")
            
            logger.info(f"Preprocessing completed successfully. Features: {len(self.selected_features_after_preproc)}, Rows: {len(processed_data)}")
            QTimer.singleShot(1200, self.show_model_training_panel)
            
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", f"{e.message}\n\n{e.details or ''}")
        except PreprocessingError as e:
            QMessageBox.critical(self, "Preprocessing Error", f"{e.message}\n\n{e.details or ''}")
        except Exception as e:
            from core.errors import handle_error
            import logging
            handle_error(e, "data_preprocessing", logging.getLogger(__name__))
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred during preprocessing:\n{str(e)}")

    def show_model_training_panel(self):
        self.start_model_training()

    def start_model_training(self):
        """
        Start model training with enhanced error handling and validation.
        """
        try:
            from core.validators import validate_model_name
            from core.errors import ModelTrainingError, ValidationError
            from core.logging_setup import get_logger
            
            logger = get_logger(__name__, getattr(self, "user_name", None))
            
            # CRITICAL FIX: Ensure matplotlib is properly configured for threading
            import matplotlib
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            plt.ioff()  # Turn off interactive mode
            
            encoder_name, ok = QInputDialog.getText(
                self, "Save Encoders",
                "Enter a name for your encoder file (no extension)\nThis saves how your data needs to be preprocessed\n⚠️THIS WILL BE NEEDED WHEN TESTING!!⚠️"
            )
            
            if ok and encoder_name:
                try:
                    # Validate encoder name
                    encoder_name = validate_model_name(encoder_name)
                    
                    enc_dir = get_writable_path(os.path.join("encoders", ""))
                    if not os.path.exists(enc_dir):
                        os.makedirs(enc_dir)
                    
                    file_path = get_writable_path(os.path.join("encoders", f"{encoder_name}.pkl"))
                    
                    # Check if file already exists
                    if os.path.exists(file_path):
                        reply = QMessageBox.question(
                            self, "File Exists", 
                            f"Encoder file '{encoder_name}.pkl' already exists. Overwrite?",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply != QMessageBox.Yes:
                            return
                    
                    with open(file_path, "wb") as f:
                        pickle.dump({
                            'encoders': self.encoders,
                            'features': self.selected_features_after_preproc,
                            'target': self.target_column_after_preproc,
                            'feature_to_base': self.feature_to_base,
                            'initial_features': getattr(self, 'initial_selected_features', self.selected_features_after_preproc)
                        }, f)
                    
                    logger.info(f"Encoder pipeline saved successfully: {file_path}")
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Preprocessing complete and pipeline saved!\n\nFile: {file_path}\n\n"
                        "Remember the file name, you'll need it when generating predictions."
                    )
                    
                except ValidationError as e:
                    QMessageBox.critical(self, "Invalid Name", f"{e.message}")
                    return
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Failed to save encoder file:\n{str(e)}")
                    return
            
            # Validate training data
            if not hasattr(self, 'processed_data') or self.processed_data is None:
                raise ModelTrainingError("No processed data available for training")
            
            if not self.selected_features_after_preproc:
                raise ModelTrainingError("No features available for training")
            
            x = self.processed_data[self.selected_features_after_preproc]
            y = self.processed_data[self.target_column_after_preproc]
            
            # Validate training data quality
            if x.empty or y.empty:
                raise ModelTrainingError("Training data is empty")
            
            if len(x) != len(y):
                raise ModelTrainingError("Feature and target data length mismatch")
            
            if len(x) < 10:
                raise ModelTrainingError("Insufficient data for training (minimum 10 samples required)")
            
            # Check for target variable issues
            unique_targets = y.nunique()
            if unique_targets == 1:
                raise ModelTrainingError("Target variable has only one unique value - cannot train model")
            
            prediction_type = "classification" if self.prediction_type.startswith("Yes/No") else "regression"
            
            # For classification, ensure we have enough samples per class
            if prediction_type == "classification" and unique_targets > 1:
                class_counts = y.value_counts()
                min_class_count = class_counts.min()
                if min_class_count < 2:
                    raise ModelTrainingError(f"Insufficient samples for classification. Smallest class has only {min_class_count} sample(s)")
            
            logger.info(f"Starting model training. Type: {prediction_type}, Features: {len(self.selected_features_after_preproc)}, Samples: {len(x)}")
            
            # CRITICAL FIX: Create progress dialog with proper parent
            self.progress_dialog = QProgressDialog("Training models...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowTitle("Training in Progress")
            self.progress_dialog.setWindowModality(Qt.ApplicationModal)  # Use application modal
            self.progress_dialog.setValue(0)
            self.progress_dialog.setMinimumWidth(400)
            self.progress_dialog.show()
            
            # Connect cancel button
            self.progress_dialog.canceled.connect(self._cancel_training)
            
            # CRITICAL FIX: Create worker with proper thread isolation
            self.worker = ModelTrainingWorker(x, y, prediction_type=prediction_type, train_func=auto_train_and_evaluate_models)
            self.worker.progress.connect(self.update_training_progress)
            self.worker.finished.connect(self.training_finished)
            self.worker.error_occurred.connect(self.training_error)  # Use error_occurred, not error
            
            # CRITICAL FIX: Ensure worker is properly started
            self.worker.start()

            
        except ModelTrainingError as e:
            QMessageBox.critical(self, "Training Error", f"{e.message}\n\n{e.details or ''}")
        except Exception as e:
            from core.errors import handle_error
            import logging
            handle_error(e, "model_training_setup", logging.getLogger(__name__))
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred:\n{str(e)}")
    
    def _cancel_training(self):
        """Handle training cancellation."""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            from core.logging_setup import get_logger
            logger = get_logger(__name__, getattr(self, "user_name", None))
            logger.info("Model training cancelled by user")
    
    def training_error(self, error_msg):
        """Handle training errors from worker thread."""
        self.progress_dialog.close()
        QMessageBox.critical(self, "Training Failed", f"Model training failed:\n{error_msg}")
        from core.logging_setup import get_logger
        logger = get_logger(__name__, getattr(self, "user_name", None))
        logger.error(f"Model training failed: {error_msg}")

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
            self.model_results_table.setHorizontalHeaderLabels(['Model', 'R² Score', 'MAE'])

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
                self.model_results_table.setItem(i, 2, QTableWidgetItem(f"{res['mae']:.4f}"))

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
        """
        Save the currently selected model to a file.

        Prompts the user to enter a name for the model file. If a valid name is provided,
        the model is saved to the specified path, and a confirmation message is shown.
        After saving, navigates to the prediction tab to allow the user to use the saved model
        for generating predictions on new data.
        """

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
        """
        Resets the prediction tab to its default state.

        This method clears all data and UI elements related to predictions. It
        resets the help state, clears summary labels, disables certain buttons,
        and resets the prediction table. This prepares the prediction tab for
        fresh input and prevents any residual data from previous usage.
        """

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
        """
        Load test data file using secure data loader.
        """
        test_file, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Test Data File", 
            "", 
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if not test_file:
            return
        
        try:
            from core.data_loader import load_data_file
            from core.errors import DataLoadingError, ValidationError, SecurityError
            
            test_df = load_data_file(test_file)
            self.prediction_test_df = test_df
            
        except SecurityError as e:
            QMessageBox.critical(self, "Security Error", f"{e.message}\n\n{e.details or ''}")
            return
        except DataLoadingError as e:
            QMessageBox.critical(self, "File Loading Error", f"{e.message}\n\n{e.details or ''}")
            return
        except ValidationError as e:
            QMessageBox.critical(self, "Data Validation Error", f"{e.message}\n\n{e.details or ''}")
            return
        except Exception as e:
            from core.errors import handle_error
            import logging
            handle_error(e, "test_file_loading", logging.getLogger(__name__))
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred:\n{str(e)}")
            return

        summary = f"<b>File:</b> {test_file}<br>"
        summary = f"<b>Shape:</b> {test_df.shape[0]} rows, {test_df.shape[1]} columns<br><br>"
        used_cols = list(test_df.columns)
        summary = "<b>Preview:</b><br>"
        summary = test_df[used_cols].head(5).to_html(index=False)
        self.pred_summary_label.setText(summary)
        self.pred_load_encoder_btn.setEnabled(True)
        self.pred_load_encoder_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
        self.pred_load_model_btn.setEnabled(False)
        self.pred_load_model_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.pred_save_btn.setEnabled(False)
        self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
        self.prediction_processed_x = None

    def pred_select_encoder_file(self):
        """Load encoder file and preprocess test data with enhanced error handling."""
        try:
            from core.preprocess import preprocess_test_data
            from core.errors import PreprocessingError, ValidationError
            from core.logging_setup import get_logger
            
            logger = get_logger(__name__, getattr(self, "user_name", None))
            
            enc_dir = get_writable_path(os.path.join("encoders", ""))
            encoder_file, _ = QFileDialog.getOpenFileName(self, "Select Encoder File", enc_dir, "Pickle Files (*.pkl)")
            if not encoder_file:
                return
            
            # Validate test data exists
            if not hasattr(self, 'prediction_test_df') or self.prediction_test_df is None:
                QMessageBox.warning(self, "No Test Data", "Please upload test data first.")
                return
            
            logger.info(f"Loading encoder file: {encoder_file}")
            
            # Validate file size before loading
            file_size = os.path.getsize(encoder_file)
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                QMessageBox.critical(self, "File Too Large", "Encoder file is too large (>50MB). This may indicate corruption.")
                return
            
            # Load encoder data with validation
            with open(encoder_file, "rb") as f:
                enc_data = pickle.load(f)
            
            # Validate encoder data structure
            required_keys = ['encoders', 'features', 'target']
            missing_keys = [key for key in required_keys if key not in enc_data]
            if missing_keys:
                raise ValidationError(f"Invalid encoder file. Missing keys: {', '.join(missing_keys)}")
            
            target = enc_data['target']
            features = enc_data['features']
            initial_features = enc_data.get('initial_features', features)
            encoders = enc_data['encoders']
            feature_to_base = enc_data.get('feature_to_base', {})
            
            # Validate encoder data
            if not encoders or not features:
                raise ValidationError("Encoder file contains empty encoders or features")
            
            if not isinstance(features, list) or not isinstance(target, str):
                raise ValidationError("Invalid data types in encoder file")
            
            test_df = self.prediction_test_df.copy()

            for col in initial_features:
                if col.endswith('_missing') and col not in test_df.columns:
                    orig_col = col[:-8]  # remove '_missing'
                    if orig_col in test_df.columns:
                        test_df[col] = test_df[orig_col].isnull()
                    else:
                        # If original column missing, fill entire column as True (all missing)
                        test_df[col] = True
            
            # Validate test data
            if test_df.empty:
                raise ValidationError("Test data is empty")
            
            # Check if required columns exist in test data using fuzzy matching
            from core.preprocess import validate_test_data_columns
            is_valid, missing_cols, mapped_cols = validate_test_data_columns(test_df, encoders, feature_to_base)
            
            if not is_valid:
                # Get required columns for better error message
                from core.preprocess import get_required_columns_for_test_data
                required_cols = get_required_columns_for_test_data(encoders, feature_to_base)
                
                error_msg = f"Test data is missing required columns: {', '.join(missing_cols)}"
                details_msg = f"Required columns: {', '.join(required_cols)}\nAvailable columns: {', '.join(test_df.columns.tolist())}"
                
                if mapped_cols:
                    details_msg += f"\n\nColumn mappings found: {', '.join([f'{orig} -> {mapped}' for orig, mapped in mapped_cols])}"
                
                raise ValidationError(error_msg, details_msg)
            
            # Log column mappings if any
            if mapped_cols:
                logger.info(f"Column mappings: {mapped_cols}")
            
            threshold = int(0.5 * len(test_df.columns))
            initial_row_count = len(test_df)
            test_df_cleaned = test_df[test_df.isnull().sum(axis=1) <= threshold].copy()
            removed_rows = initial_row_count - len(test_df_cleaned)
            
            if len(test_df_cleaned) == 0:
                raise ValidationError("All test data rows were removed due to missing values")
            
            if removed_rows > 0:
                QMessageBox.information(self, "Rows Removed", f"{removed_rows} rows were removed from your test data due to having more than 50% missing values.")
            
            # Preprocess test data
            processed_test_df = preprocess_test_data(
                test_df_cleaned,
                encoders,
                features,
                feature_to_base,
                target_col=target
            )
            
            if processed_test_df.empty:
                raise PreprocessingError("Preprocessing resulted in empty test data")
            
            # Store preprocessing results
            self.prediction_features = features
            self.prediction_target = target
            self.prediction_encoders = encoders
            self.prediction_processed_x = processed_test_df
            
            logger.info(f"Test data preprocessed successfully. Shape: {processed_test_df.shape}")
            
            QMessageBox.information(self, "Success", f"Test data preprocessed successfully!\nProcessed {len(processed_test_df)} rows with {len(features)} features.\nYou can now load a model and generate predictions.")
            
            self.pred_load_model_btn.setEnabled(True)
            self.pred_load_model_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
            self.pred_save_btn.setEnabled(False)
            self.pred_save_btn.setStyleSheet(PREDICTION_DISABLED_STYLESHEET)
            
        except FileNotFoundError:
            QMessageBox.critical(self, "File Error", "The selected encoder file was not found.")
        except (pickle.UnpicklingError, EOFError):
            QMessageBox.critical(self, "Corrupt File", "Could not load the encoder file. It may be corrupted or incompatible.")
        except PermissionError:
            QMessageBox.critical(self, "Permission Error", "Permission denied when accessing the encoder file.")
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", f"{e.message}\n\n{e.details or ''}")
        except PreprocessingError as e:
            QMessageBox.critical(self, "Preprocessing Error", f"{e.message}\n\n{e.details or ''}")
        except Exception as e:
            from core.errors import handle_error
            import logging
            handle_error(e, "encoder_loading", logging.getLogger(__name__))
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred:\n{str(e)}")

    def pred_select_model_file(self):
        """Load model file and generate predictions with enhanced error handling."""
        try:
            from core.errors import PredictionError, ValidationError
            from core.logging_setup import get_logger
            
            logger = get_logger(__name__, getattr(self, "user_name", None))
            
            model_dir = get_writable_path(os.path.join("models", ""))
            model_file, _ = QFileDialog.getOpenFileName(self, "Select Model File", model_dir, "Pickle Files (*.pkl)")
            if not model_file:
                return
            
            # Validate prerequisites
            if not hasattr(self, 'prediction_processed_x') or self.prediction_processed_x is None:
                QMessageBox.warning(self, "No Processed Data", "Please load and process test data with encoders first.")
                return
            
            logger.info(f"Loading model file: {model_file}")
            
            # Validate file size before loading
            file_size = os.path.getsize(model_file)
            if file_size > 100 * 1024 * 1024:  # 100MB limit for models
                QMessageBox.critical(self, "File Too Large", "Model file is too large (>100MB). This may indicate corruption.")
                return
            
            # Load model with validation
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            
            # Validate model object
            if not hasattr(model, 'predict'):
                raise ValidationError("Loaded object is not a valid machine learning model (missing predict method)")
            
            logger.info("Model loaded successfully, starting predictions")

            # Generate predictions with validation
            prediction_type = "classification" if self.prediction_type.startswith("Yes/No") else "regression"
            
            # Validate input data shape
            expected_features = len(self.prediction_features)
            actual_features = self.prediction_processed_x.shape[1]
            if actual_features != expected_features:
                raise PredictionError(
                    f"Feature count mismatch. Expected {expected_features}, got {actual_features}",
                    "The model and processed data have incompatible feature dimensions"
                )
            
            # Generate predictions
            preds = model.predict(self.prediction_processed_x)
            
            # Validate predictions
            if preds is None or len(preds) == 0:
                raise PredictionError("Model returned empty predictions")
            
            if len(preds) != len(self.prediction_processed_x):
                raise PredictionError("Prediction count doesn't match input data count")
            
            result_df = self.prediction_test_df.copy()

            class_names = None
            if prediction_type == "classification":
                target_encoder = self.prediction_encoders.get(self.prediction_target)
                if target_encoder and hasattr(target_encoder, "inverse_transform"):
                    try:
                        # Validate predictions are within expected range
                        if hasattr(target_encoder, "classes_"):
                            max_class = len(target_encoder.classes_) - 1
                            if any(pred < 0 or pred > max_class for pred in preds):
                                logger.warning("Some predictions are outside expected class range")
                        
                        preds = target_encoder.inverse_transform(preds)
                        if hasattr(target_encoder, "classes_"):
                            class_names = target_encoder.classes_
                    except Exception as e:
                        logger.warning(f"Failed to inverse transform predictions: {e}")

            result_df["Prediction"] = preds

            # Add probability columns for classification
            if prediction_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(self.prediction_processed_x)
                    
                    # Validate probabilities
                    if proba is not None and len(proba) > 0:
                        if class_names is None and hasattr(model, "classes_"):
                            class_names = model.classes_

                        if class_names is not None:
                            if len(class_names) == 2:
                                pos_class = class_names[1]
                                if proba.shape[1] == 2:
                                    result_df[f"Probability_{pos_class} (%)"] = (proba[:, 1] * 100).round(2)
                                elif proba.shape[1] == 1:
                                    result_df[f"Probability_{pos_class} (%)"] = (proba[:, 0] * 100).round(2)
                            else:
                                for idx, cname in enumerate(class_names):
                                    if idx < proba.shape[1]:
                                        result_df[f"Probability_{cname} (%)"] = (proba[:, idx] * 100).round(2)
                        else:
                            for i in range(proba.shape[1]):
                                result_df[f"Probability_Class_{i} (%)"] = (proba[:, i] * 100).round(2)
                except Exception as e:
                    logger.warning(f"Failed to generate prediction probabilities: {e}")

            self.prediction_result_df = result_df
            
            logger.info(f"Predictions generated successfully for {len(result_df)} samples")
            
            self.show_predictions_table(result_df.head(100))
            self.pred_save_btn.setEnabled(True)
            self.pred_save_btn.setStyleSheet(PREDICTION_ENABLED_STYLESHEET)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Predictions generated successfully!\n{len(result_df)} predictions created.\nShowing first 100 rows in the table."
            )

        except FileNotFoundError:
            QMessageBox.critical(self, "File Error", "The selected model file was not found.")
        except (pickle.UnpicklingError, EOFError):
            QMessageBox.critical(self, "Corrupt File", "Could not load the model file. It may be corrupted or incompatible.")
        except PermissionError:
            QMessageBox.critical(self, "Permission Error", "Permission denied when accessing the model file.")
        except ValidationError as e:
            QMessageBox.critical(self, "Validation Error", f"{e.message}\n\n{e.details or ''}")
        except PredictionError as e:
            QMessageBox.critical(self, "Prediction Error", f"{e.message}\n\n{e.details or ''}")
        except Exception as e:
            from core.errors import handle_error
            import logging
            handle_error(e, "model_prediction", logging.getLogger(__name__))
            self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
            QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred during prediction:\n{str(e)}")


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
        """Load data file using secure data loader with comprehensive error handling."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Data File", 
            "", 
            "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if file_name:
            try:
                from core.data_loader import load_data_file
                from core.errors import DataLoadingError, ValidationError, SecurityError
                from core.async_operations import optimize_dataframe_memory
                
                # Use the secure data loader
                raw_df = load_data_file(file_name)
                
                # Optimize memory usage for large datasets
                self.df = optimize_dataframe_memory(raw_df)
                
                columns = list(self.df.columns)
                self.features_list.clear()
                self.target_combo.clear()
                for col in columns:
                    self.features_list.addItem(col)
                    self.target_combo.addItem(col)
                
                self.set_tab_enabled(1, True)
                self.set_tab_enabled(2, False)
                self.set_tab_enabled(3, False)
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"File loaded successfully!\n"
                    f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n"
                    f"Now select features and target column."
                )
                self.goto_select_tab()
                
            except SecurityError as e:
                QMessageBox.critical(self, "Security Error", f"{e.message}\n\n{e.details or ''}")
            except DataLoadingError as e:
                QMessageBox.critical(self, "File Loading Error", f"{e.message}\n\n{e.details or ''}")
            except ValidationError as e:
                QMessageBox.critical(self, "Data Validation Error", f"{e.message}\n\n{e.details or ''}")
            except Exception as e:
                from core.errors import handle_error
                import logging
                handle_error(e, "file_loading", logging.getLogger(__name__))
                self.handle_error_with_popup(type(e), e, sys.exc_info()[2])
                QMessageBox.critical(self, "Unexpected Error", f"An unexpected error occurred:\n{str(e)}")
