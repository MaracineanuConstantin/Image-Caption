import sys
import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
import cv2
import torch
import time

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from model import EncoderCNN, DecoderRNN
from utils import load_checkpoint
import numpy as np
import pickle
from torchvision import transforms
from build_vocab import Vocabulary
from PIL import Image

class ImageCaptionGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_path = 'data/vocab.pkl'
        self.embed_size = 256
        self.hidden_size = 512
        self.num_layers = 1 
        self.learning_rate = 0.001
        self.image_path = None
        self.model = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]
        )

        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        
    def initialize_model(self, filename):
        self.start_epoch, self.encoder, self.decoder, self.validation_loss, self.epochs_since_last_improvement = load_checkpoint(self.model, self.embed_size,
                                                                                        self.hidden_size, self.vocab, self.num_layers, self.learning_rate)
        
        print(f'Loaded checkpoint: model-{filename} epoch-{self.start_epoch}')
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.encoder.eval()
        self.decoder.eval()


    def generate_caption(self, image_path):
        image = self.load_image(image_path, self.transform)
        image_tensor = image.to(self.device)

        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()


        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)

        return sentence


    def generate_frame_caption(self, frame):
        pil_image = Image.fromarray(frame)

        image = self.load_image(pil_image, self.transform)
        image_tensor = image.to(self.device)

        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)

        return sentence

    
    def load_image(self, image_path, transform=None):
        if (type(image_path)==Image.Image):
            image = image_path.convert('RGB')
        else:    
            image = Image.open(image_path).convert('RGB')

        image = image.resize([224, 224], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image
        

class ConsoleRedirector(QtCore.QObject):
    text_written = QtCore.pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

class ImageLoaderWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.setGeometry(400, 200, 1000, 800)
        self.setContentsMargins(20, 20, 20, 20)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)


        # Buttons to load model, image, livestream
        self.loadModelButton = QPushButton("Load Model",self.central_widget)
        self.loadModelButton.setGeometry(QtCore.QRect(20, 50, 170, 80))
        self.loadModelButton.setFont(QFont('Times', 12))
        self.loadModelButton.clicked.connect(self.load_model)

        self.loadImageButton = QPushButton("Load Image", self.central_widget)
        self.loadImageButton.setGeometry(QtCore.QRect(20, 170, 170, 80))
        self.loadImageButton.setFont(QFont('Times', 12))
        self.loadImageButton.clicked.connect(self.load_image)
        
        self.livestreamButton = QPushButton("Livestream", self.central_widget)
        self.livestreamButton.setGeometry(QtCore.QRect(20, 290, 170, 80))
        self.livestreamButton.setFont(QFont('Times', 12))
        self.livestreamButton.clicked.connect(self.start_livestream)

        # Label to display loaded image
        self.image_label = QLabel(self.central_widget)
        self.image_label.setGeometry(220, 50, 700, 450)


        # Output console
        self.output_console = QTextEdit(self.central_widget)
        self.output_console.setGeometry(0, 500, 1000, 300)
        self.output_console.setFont(QFont('Times', 10))
        self.output_console.setStyleSheet("padding: 15px")

        
        self.output_console.setReadOnly(True)


        # Redirect console output to QTextEdit
        self.console_redirector = ConsoleRedirector()
        sys.stdout = self.console_redirector
        self.console_redirector.text_written.connect(self.onUpdateText) 

        # Variable to hold the loaded image
        self.loaded_image = None

        self.image_caption_generate = ImageCaptionGenerator()


        
    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Image", "png/", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        if filename:
            pixmap = QPixmap(filename)
            self.loaded_image = filename
            self.display_image(pixmap)
            if self.image_caption_generate.model:
                self.generate_caption(self.loaded_image)
        

    def load_model(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load model", "experiments/", "Model Files (*.pt *ckpt *pth *pth.tar)", options=options)
        if filename:
            self.image_caption_generate.model = filename
            self.image_caption_generate.initialize_model(filename)


    def display_image(self, pixmap):
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(pixmap)


    def onUpdateText(self, text):
        self.output_console.moveCursor(QtGui.QTextCursor.End)
        self.output_console.insertPlainText(text)


    def generate_caption(self, image_path):
        caption = self.image_caption_generate.generate_caption(image_path)
        print(caption)


    def start_livestream(self):
        vid = cv2.VideoCapture(0)

        while(True):
            time.sleep(1)
            ret, frame = vid.read()
            cv2.imshow('frame', frame)
            
            caption = self.image_caption_generate.generate_frame_caption(frame)

            print(caption)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLoaderWindow()
    window.show()

    sys.exit(app.exec_())
