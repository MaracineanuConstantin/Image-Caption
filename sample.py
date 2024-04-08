import torch
from model import EncoderCNN, DecoderRNN
from utils import load_checkpoint
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from PIL import Image
import random


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main():
        # config params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = 'data/vocab.pkl'
    val_dir = 'data/resizedval2014'
    val_caption_path = 'data/annotations/captions_val2014.json'
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 128
    learning_rate = 0.001
    image_path = 'png/biciclisti.png'

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    start_epoch, encoder, decoder, validation_loss, epochs_since_last_improvement = load_checkpoint("experiments/9/best_model_9.pth", embed_size, hidden_size, vocab, num_layers, learning_rate)

    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()


    # Prepare an image
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
    plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    main()
