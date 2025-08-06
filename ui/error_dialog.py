import os
import subprocess
import platform
import urllib.parse
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from typing import Dict, Any

from core.config import Config


class ErrorDialog(QDialog):
    """Error dialog with email functionality for sending error logs to admin."""
    
    def __init__(self, error_details: Dict[str, Any], log_file_path: str, 
                 admin_email: str = "admin@company.com", parent=None):
        super().__init__(parent)
        self.error_details = error_details
        self.log_file_path = log_file_path
        self.admin_email = admin_email
        self.init_ui()
    
    def init_ui(self):
        """Initialize the error dialog UI."""
        self.setWindowTitle("Application Error")
        self.setWindowIcon(QIcon("Images/icon.png") if os.path.exists("Images/icon.png") else QIcon())
        self.setModal(True)
        self.setMinimumSize(500, 400)
        self.setMaximumSize(700, 600)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Error icon and title
        title_layout = QHBoxLayout()
        title_label = QLabel("An Error Occurred")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #d32f2f;
                padding: 10px 0;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Error summary
        summary_label = QLabel("The application encountered an unexpected error. "
                              "You can help improve the application by sending this error report to the administrator.")
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("font-size: 14px; color: #555; margin-bottom: 10px;")
        layout.addWidget(summary_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)
        
        # Error details section
        details_label = QLabel("Error Details:")
        details_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 10px;")
        layout.addWidget(details_label)
        
        # Error details text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        self.details_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        # Format error details for display
        error_text = self._format_error_details()
        self.details_text.setPlainText(error_text)
        layout.addWidget(self.details_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Send Email button
        self.email_button = QPushButton("Send Error Report")
        self.email_button.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0d47a1;
            }
        """)
        self.email_button.clicked.connect(self.send_error_email)
        button_layout.addWidget(self.email_button)
        
        # Close button
        self.close_button = QPushButton("Close")
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
        """)
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Set default focus to close button
        self.close_button.setFocus()
    
    def _format_error_details(self) -> str:
        """Format error details for display."""
        details = []
        details.append(f"Time: {self.error_details.get('timestamp', 'Unknown')}")
        details.append(f"User: {self.error_details.get('user_name', 'Unknown')}")
        details.append(f"Error Type: {self.error_details.get('error_type', 'Unknown')}")
        details.append(f"Error Message: {self.error_details.get('error_message', 'Unknown')}")
        details.append(f"File: {self.error_details.get('file_name', 'Unknown')}")
        details.append(f"Line: {self.error_details.get('line_number', 'Unknown')}")
        
        return "\n".join(details)
    
    def send_error_email(self):
        """Open default email client with error report."""
        try:
            # Create email subject and body
            subject = f"Error Report - TC AI Prediction Tool - {self.error_details.get('error_type', 'Unknown Error')}"
            
            body_lines = [
                "Dear Administrator,",
                "",
                "An error occurred in the TC AI Prediction Tool. Please find the error details below:",
                "",
                "--- ERROR DETAILS ---",
                f"Time: {self.error_details.get('timestamp', 'Unknown')}",
                f"User: {self.error_details.get('user_name', 'Unknown')}",
                f"Error Type: {self.error_details.get('error_type', 'Unknown')}",
                f"Error Message: {self.error_details.get('error_message', 'Unknown')}",
                f"File: {self.error_details.get('file_name', 'Unknown')}",
                f"Line Number: {self.error_details.get('line_number', 'Unknown')}",
                "",
                "--- FULL TRACEBACK ---"
            ]
            
            # Add traceback if available
            if 'full_traceback' in self.error_details and self.error_details['full_traceback']:
                body_lines.extend(self.error_details['full_traceback'])
            
            body_lines.extend([
                "",
                "Please find the complete error log file attached to this email.",
                "",
                "Best regards,",
                f"{self.error_details.get('user_name', 'User')}"
            ])
            
            body = "\n".join(body_lines)
            
            # URL encode the email components
            subject_encoded = urllib.parse.quote(subject)
            body_encoded = urllib.parse.quote(body)
            
            # Create mailto URL
            mailto_url = f"mailto:{self.admin_email}?subject={subject_encoded}&body={body_encoded}"
            
            # Try to open email client
            success = self._open_email_client(mailto_url)
            
            if success:
                # Show instructions for attaching log file
                QMessageBox.information(
                    self,
                    "Email Client Opened",
                    f"Your email client has been opened with the error report.\n\n"
                    f"Please attach the error log file located at:\n{self.log_file_path}\n\n"
                    f"Then send the email to help improve the application."
                )
            else:
                # Fallback: show manual instructions
                self._show_manual_email_instructions(subject, body)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Email Error",
                f"Failed to open email client:\n{str(e)}\n\n"
                f"Please manually send an email to:\n{self.admin_email}\n\n"
                f"Include the error log file:\n{self.log_file_path}"
            )
    
    def _open_email_client(self, mailto_url: str) -> bool:
        """Try to open the default email client."""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                os.startfile(mailto_url)
            elif system == "darwin":  # macOS
                subprocess.run(["open", mailto_url], check=True)
            else:  # Linux and others
                subprocess.run(["xdg-open", mailto_url], check=True)
            
            return True
            
        except Exception:
            return False
    
    def _show_manual_email_instructions(self, subject: str, body: str):
        """Show manual email instructions if automatic opening fails."""
        instructions = (
            f"Please manually create an email with these details:\n\n"
            f"To: {self.admin_email}\n"
            f"Subject: {subject}\n\n"
            f"Body:\n{body[:500]}{'...' if len(body) > 500 else ''}\n\n"
            f"Attach the error log file:\n{self.log_file_path}"
        )
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Manual Email Instructions")
        msg_box.setText("Could not open email client automatically.")
        msg_box.setDetailedText(instructions)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()


def show_error_dialog(error_details: Dict[str, Any], log_file_path: str, 
                     admin_email: str = Config.admin.ADMIN_EMAIL, parent=None):
    """
    Show error dialog with email functionality.
    
    Args:
        error_details: Error details dictionary from error logger
        log_file_path: Path to the error log file
        admin_email: Administrator email address
        parent: Parent widget
    """
    dialog = ErrorDialog(error_details, log_file_path, admin_email, parent)
    dialog.exec_()