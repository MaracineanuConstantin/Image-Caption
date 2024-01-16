import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pickle
from data_loader import get_loader, validation_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import AverageMeter
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver


## TODO: check 1 image dataflow
## check real img caption vs output caption
## check sample.py
## TODO: Implement sacred
## TODO: Code refactor into functions
## TODO: best_loss = first loss
## TODO: BLEU4 eval metric
## TODO: de facut media la loss la finalul unei epoci si dupa bucla de antrenat verificat best_loss


# Create a sacred experiment
ex = Experiment("train_experiment")
observers_directory = 'experiments'
ex.observers.append(FileStorageObserver(observers_directory))


@ex.config
def cfg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/'
    crop_size = 224
    vocab_path = 'data/vocab.pkl'
    train_dir = 'data/resized2014'
    val_dir = 'data/resizedval2014'
    caption_path = 'data/annotations/captions_train2014.json'
    val_caption_path = 'data/annotations/captions_val2014.json'
    log_step = 1
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 10
    batch_size = 128
    num_workers = 2
    learning_rate = 1e-3
    best_loss = None
    epochs_since_last_improvement = 0

@ex.capture
def train(device, data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step, num_epochs, log_step):
    encoder.train()
    decoder.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    for i, (images, captions, lengths) in enumerate(data_loader):
            data_time.update(time.time() - start)
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # average batch loss
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Get training statistics.
            stats = f'Training epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Batch time {batch_time.avg:.3f}, Data time {data_time.avg:.3f}, Loss {loss.item():.3f}, Perplexity {np.exp(loss.item()):.3f}'
            # Print training statistics .
            if (i+1) % log_step == 0:
                print(stats)

    ex.log_scalar('Train/Loss', losses.avg, epoch)
    print(f'Training loss: {losses.avg}')
    return losses.avg


@ex.capture
def validate(device, val_loader, encoder, decoder, criterion, epoch, total_step, num_epochs, log_step):
    encoder.eval()
    decoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            stats = f'Validation epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Batch time {batch_time.avg:.3f}, Loss {loss.item():.3f}'
            # Print training statistics .
            if (i+1) % log_step == 0:
                print(stats)

    ex.log_scalar('Validation/Loss', losses.avg, epoch)
    print(f'Validation loss: {losses.avg:.3f}')
    return losses.avg


@ex.automain
def main(device, model_path, crop_size, vocab_path, train_dir, val_dir, caption_path, val_caption_path, 
        log_step, embed_size, hidden_size, num_layers, num_epochs, batch_size, num_workers, learning_rate, _run, best_loss, 
        epochs_since_last_improvement):

    
    # Create directory for current time
    current_time = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    log_dir = os.path.join(observers_directory, _run._id)
    os.makedirs(log_dir, exist_ok=True)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    

    # Build data loader
    data_loader = get_loader(train_dir, caption_path, vocab, 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers)

    val_loader = validation_loader(val_dir, val_caption_path, vocab, transform, batch_size,
                            num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        start = time.time()
        train_loss = train(device, data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step, num_epochs)
        validation_loss = validate(device, val_loader, encoder, decoder, criterion, epoch, total_step, num_epochs)
        
        if best_loss is None:
            best_loss = validation_loss
        # Save the model checkpoints
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(decoder.state_dict(), os.path.join(
                log_dir, 'decoder-{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                log_dir, 'encoder-{}.ckpt'.format(epoch+1)))
        else:
            epochs_since_last_improvement += 1
        
        if epochs_since_last_improvement == 100:
            break

