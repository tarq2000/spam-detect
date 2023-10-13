import sys
import docx2txt
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class WordParagraphCounter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self): 
        self.setWindowTitle('Find Spam in Word File')
        self.setGeometry(100, 100, 800, 600) # Size of the main window

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)

        self.upload_button = QPushButton('Upload Word File', central_widget)
        central_layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.upload_file)

        self.result_text = QTextEdit(central_widget)
        central_layout.addWidget(self.result_text)
        self.result_text.setReadOnly(True)
        self.result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(central_widget)
        
        self.result_text2 = QTextEdit(central_widget)
        self.result_text2.setFixedHeight(50)  
        central_layout.addWidget(self.result_text2)
        self.result_text2.setReadOnly(True)

        # Load training data from the Excel file
        self.df = pd.read_excel('spamdata.xlsx', header=None, skiprows=1, names=['label', 'message'], dtype={'label': str}, engine='openpyxl')
        self.X_train = self.df['message'].str.strip()
        self.y_train = self.df['label'].str.strip()

        # Create a TfidfVectorizer for text vectorization
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Train a Random Forest classifier for finding spam and ham
        X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_classifier.fit(X_train_vectorized, self.y_train)
        
    def upload_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Word File', '', 'Word Files (*.docx);;All Files (*)', options=options)
    
        if file_path:
            paragraphs = docx2txt.process(file_path)
            self.result_text.setPlainText(paragraphs)

            # Preprocess the paragraph text
            paragraph_features = self.vectorizer.transform([paragraphs])

            # Predict whether the paragraph is spam or ham
            prediction = self.rf_classifier.predict(paragraph_features)[0]

            if prediction == 'ham':
                prediction_result = "My message is ham"
            else:
                prediction_result = "My message is spam"

            self.result_text2.setPlainText(prediction_result)

def main():
    app = QApplication(sys.argv)
    window = WordParagraphCounter()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
