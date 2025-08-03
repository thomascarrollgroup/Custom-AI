from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QAbstractItemView, QHBoxLayout, QPushButton

class ColumnSelectDialog(QDialog):
    def __init__(self, columns, parent=None):
        """
        Initialize the dialog with a list of columns to select from.

        :param columns: List of column names to select from.
        :type columns: list[str]
        :param parent: Parent widget (optional).
        :type parent: QWidget
        """
        super().__init__(parent)
        self.setWindowTitle("Select Columns")
        self.setMinimumWidth(350)
        layout = QVBoxLayout(self)
        label = QLabel("Select columns to add:")
        layout.addWidget(label)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_widget.addItems(columns)
        layout.addWidget(self.list_widget)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def selected_columns(self):
        return [item.text() for item in self.list_widget.selectedItems()]