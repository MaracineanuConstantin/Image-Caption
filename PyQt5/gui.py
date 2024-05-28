import sys
import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase, QImage
import cv2
import torch
import time
import csv
import pandas as pd
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
import pyttsx3
from threading import Thread as Thread


class ImageCaptionGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_path = 'data/vocab2017.pkl'
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
        self.text = ''
        self.toggleSpeak = True
        self.reader_object = pd.read_csv("test_images.csv")
        self.loaded_image = None
        self.runLivestream = False
        self.runVideostream = False

        self.setWindowTitle("GUI")
        self.setGeometry(400, 100, 1050, 900)
        self.setContentsMargins(20, 20, 20, 20)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)


        # Buttons to load model, image, livestream from camera, videostream from video, clear the chat, clear the image 
        self.loadModelButton = QPushButton("Load Model",self.central_widget)
        self.loadModelButton.setGeometry(QtCore.QRect(20, 30, 170, 60))
        self.loadModelButton.setFont(QFont('Times', 11))
        self.loadModelButton.clicked.connect(self.thread1)

        self.loadImageButton = QPushButton("Load Image", self.central_widget)
        self.loadImageButton.setGeometry(QtCore.QRect(20, 110, 170, 60))
        self.loadImageButton.setFont(QFont('Times', 11))
        self.loadImageButton.clicked.connect(self.thread2)

        self.livestreamButton = QPushButton("Livestream", self.central_widget)
        self.livestreamButton.setGeometry(QtCore.QRect(20, 190, 170, 60))
        self.livestreamButton.setFont(QFont('Times', 11))
        self.livestreamButton.clicked.connect(self.thread3)


        self.videostreamButton = QPushButton("Videostream", self.central_widget)
        self.videostreamButton.setGeometry(QtCore.QRect(20, 270, 170, 60))
        self.videostreamButton.setFont(QFont('Times', 11))
        self.videostreamButton.clicked.connect(self.thread4)


        self.clearButton = QPushButton("Clear chat", self.central_widget)
        self.clearButton.setGeometry(QtCore.QRect(20, 350, 170, 40))
        self.clearButton.setFont(QFont("Times", 11))
        self.clearButton.clicked.connect(self.clear_chat)

        self.clearImage = QPushButton("Clear image", self.central_widget)
        self.clearImage.setGeometry(QtCore.QRect(20, 410, 170, 40))
        self.clearImage.setFont(QFont("Times", 11))
        self.clearImage.clicked.connect(self.clear_image)

        self.toggleSpeakButton = QPushButton("Text To Speech ON", self.central_widget)
        self.toggleSpeakButton.setGeometry(QtCore.QRect(20, 470, 170, 40))
        self.toggleSpeakButton.setFont(QFont("Times", 11))
        self.toggleSpeakButton.clicked.connect(self.toggle_speak)


        # Label to display loaded image
        self.image_label = QLabel(self.central_widget)
        self.image_label.setGeometry(250, 30, 700, 480)

        self.engine = pyttsx3.init()

        # Output console
        self.output_console = QTextEdit(self.central_widget)
        self.output_console.setGeometry(0, 550, 1000, 300)
        self.output_console.setFont(QFont('Times', 10))
        self.output_console.setStyleSheet("padding: 15px")
        self.output_console.ensureCursorVisible()
        self.output_console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.output_console.setReadOnly(True)


        # Redirect console output to QTextEdit
        self.console_redirector = ConsoleRedirector()
        sys.stdout = self.console_redirector
        self.console_redirector.text_written.connect(self.onUpdateText)
        # Variable to hold the loaded image
        self.image_caption_generate = ImageCaptionGenerator()

    
    def clear_chat(self):
        self.output_console.clear()
    

    def clear_image(self):
        self.image_label.clear()
    

    def toggle_speak(self):
        self.toggleSpeak = not self.toggleSpeak

        if self.toggleSpeak:
            self.toggleSpeakButton.setText("Text To Speech ON")
        else:
            self.toggleSpeakButton.setText("Text To Speech OFF")


    def load_image(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Image", "experiments/SCSS", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4)",
                                                  options=options)

        if filename:
            pixmap = QPixmap(filename)
            self.loaded_image = os.path.basename(filename)
            self.display_image(pixmap)
            

            for index, row in self.reader_object.iterrows():
                if(self.loaded_image == row[0]):
                    print(f'Reference caption: {row[4]}')
                        # self.speak(f'Reference caption: {row[4]}')
                        
            
            if self.image_caption_generate.model:
                self.generate_caption(filename)
        

    def load_model(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load model", "experiments/SCSS", "Model Files (*.pt *ckpt *pth *pth.tar)", options=options)
        if filename:
            self.image_caption_generate.model = filename
            self.image_caption_generate.initialize_model(filename)
            # self.speak(f'Loaded checkpoint: model-{filename} epoch-{self.image_caption_generate.start_epoch}')


    def numpy_to_image(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        return pixmap


    def display_image(self, pixmap):
        self.image_label.setScaledContents(True)
        self.image_label.setPixmap(pixmap)
        


    def onUpdateText(self, text):
        self.text = text
        self.output_console.moveCursor(QtGui.QTextCursor.End)
        self.output_console.insertPlainText(text)
        self.output_console.verticalScrollBar().setValue(self.output_console.verticalScrollBar().maximum())
        


    def speak(self, text):
        if self.toggleSpeak:
            self.engine.say(text)
            self.engine.runAndWait()

    def generate_caption(self, image_path):
        caption = self.image_caption_generate.generate_caption(image_path)
        print(f'Generated caption: {caption}')
        self.speak(f'Generated caption: {caption}')

    
    def start_livestream(self):
        if self.livestreamButton.text() == "Livestream":
            self.livestreamButton.setText("Stopstream")
        else:
            self.livestreamButton.setText("Livestream")

        self.runLivestream = not self.runLivestream

        vid = cv2.VideoCapture(0)
        while(self.runLivestream):
            ret, frame = vid.read()
            image_frame = self.numpy_to_image(frame)
            # cv2.imshow('frame', frame)
            self.display_image(image_frame)

            if self.image_caption_generate.model:
                caption = self.image_caption_generate.generate_frame_caption(frame)

                print(caption)
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()

        cv2.destroyAllWindows()

    def start_videostream(self):
        if self.videostreamButton.text() == 'Stopstream':
            self.runVideostream = not self.runVideostream
            self.videostreamButton.setText("Videostream")
            return

        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load video", "experiments/SCSS", "Model Files (*.mp4 *.avi *.mov *.wmw *.avchd *.webm *.flv)", options=options)


        if filename:

            if self.videostreamButton.text() == "Videostream":
                self.videostreamButton.setText("Stopstream")
            else:
                self.videostreamButton.setText("Videostream")

            vid = cv2.VideoCapture(filename)
            self.runVideostream = not self.runVideostream
            while(self.runVideostream):
                ret, frame = vid.read()
                image_frame = self.numpy_to_image(frame)
                self.display_image(image_frame)

                if self.image_caption_generate.model:
                    caption = self.image_caption_generate.generate_frame_caption(frame)

                    print(caption)
                time.sleep(0.5)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
            vid.release()

            cv2.destroyAllWindows()


    def thread1(self):
        t1 = Thread(target = self.load_model)
        t1.start()
    
    def thread2(self):
        t2 = Thread(target = self.load_image)
        t2.start()
    
    def thread3(self):
        t3 = Thread(target = self.start_livestream)
        t3.start()

    def thread4(self):
        t4 = Thread(target = self.start_videostream)
        t4.start()
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageLoaderWindow()
    window.show()

    sys.exit(app.exec_())
