# model
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

import time

# camera
import cv2
import pyrealsense2
from realsense_depth import *

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main():
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    vocab_path = "data/vocab.pkl"
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder_path = "models/encoder-12-205.ckpt"
    decoder_path = "models/decoder-12-205.ckpt"
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        time.sleep(2)
        # Save color frame as an image file
        image_path = 'temp_image.jpg'
        cv2.imwrite(image_path, color_frame)


        image = load_image(image_path, transform)
        image_tensor = image.to(device)


        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        
        # Print out the image and the generated caption
        print (sentence)
        image = Image.open(image_path)
        cv2.imshow("image ",np.asarray(image))

        key = cv2.waitKey(1)
        if key == 27:
            os.remove(image_path)
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()