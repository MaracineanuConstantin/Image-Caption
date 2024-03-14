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
from utils import AverageMeter, seed_everything, save_checkpoint, load_checkpoint
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from nltk.translate.bleu_score import corpus_bleu
from nlgmetricverse import NLGMetricverse, load_metric
from torch.utils.tensorboard import SummaryWriter

## de antrenat in functie de validation loss dar verificat/calculat de asemenea metricile
## OPTIONAL: de salvat imaginea la fiecare etapa din ANTRENARE
## eventual salvat sub fiecare strat numarul/numele layer-ului

# Create a sacred experiment
ex = Experiment("train_experiment")
observers_directory = 'experiments'
ex.observers.append(FileStorageObserver(observers_directory))


@ex.config
def cfg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    num_epochs = 15
    batch_size = 128
    num_workers = 0
    learning_rate = 1e-3
    best_loss = None
    epochs_since_last_improvement = 0
    start_epoch = 0
    seed = seed_everything(42)

@ex.capture
def train(device, data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step, num_epochs, log_step, writer):
    encoder.train()
    decoder.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    for i, (images, captions, lengths) in enumerate(data_loader):
            data_time.update(time.time() - start)
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # captions au padding, lengths au marimile originale (inainte sa fie umplut cu 0 pana la max length)
            # targets = valorile puse 
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
            
            # print(f'Training epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Batch time {batch_time.avg:.3f}, Data time {data_time.avg:.3f}, Loss {loss.item():.3f}, Perplexity {np.exp(loss.item()):.3f}')

    ex.log_scalar('Train/Loss', losses.avg, epoch)
    writer.add_scalar("Loss/Train", losses.avg, epoch)
    print(f'Training loss: {losses.avg}')
    return losses.avg


@ex.capture
def validate(device, val_loader, encoder, decoder, criterion, epoch, total_step, num_epochs, log_step, writer):
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
            # = crossentropyloss: -(SUM(ground truth - log(targets)) 
            losses.update(loss.item(), images.size(0))

            stats = f'Validation epoch [{epoch}/{num_epochs}], Step [{i}/{len(val_loader)}], Batch time {batch_time.avg:.3f}, Loss {loss.item():.3f}'
            # Print training statistics .
            if (i+1) % log_step == 0:
                print(stats)

    ex.log_scalar('Validation/Loss', losses.avg, epoch)
    writer.add_scalar("Loss/Val", losses.avg, epoch)
    print(f'Validation loss: {losses.avg:.3f}')
    return losses.avg



@ex.automain
def main(device, crop_size, vocab_path, train_dir, val_dir, caption_path, val_caption_path, 
        log_step, embed_size, hidden_size, num_layers, num_epochs, batch_size, num_workers, learning_rate, _run, best_loss, 
        epochs_since_last_improvement, start_epoch, seed):

    print(f'main {seed}')

    # create log directory to store the encoder & decoder
    log_dir = os.path.join(observers_directory, _run._id)

    writer = SummaryWriter(log_dir=log_dir)

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

    
    # start_epoch, encoder, decoder, validation_loss, epochs_since_last_improvement = load_checkpoint("experiments/24/best_model.pth.tar", embed_size, hidden_size, vocab, num_layers, learning_rate)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        train_loss = train(device, data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step, num_epochs, writer=writer)
        validation_loss = validate(device, val_loader, encoder, decoder, criterion, epoch, total_step, num_epochs, writer=writer)

        if best_loss is None:
            best_loss = validation_loss

        # Save the model checkpoints
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_since_last_improvement = 0
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)
        
        if epoch == (num_epochs-1):
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)

        else:
            epochs_since_last_improvement += 1
        
        if epochs_since_last_improvement == 100:
            break
    
    writer.close()
        
