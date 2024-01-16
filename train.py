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

## TODO: check 1 image dataflow
## check real img caption vs output caption
## check sample.py
## TODO: Implement sacred
## TODO: Code refactor into functions
## TODO: best_loss = first loss
## TODO: BLEU4 eval metric
## TODO: de facut media la loss la finalul unei epoci si dupa bucla de antrenat verificat best_loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    global best_loss
    best_loss = None
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    

    # Build data loader
    data_loader = get_loader(args.train_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    val_loader = validation_loader(args.val_dir, args.val_caption_path, vocab, transform, args.batch_size,
                            num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        start = time.time()
        train_loss = train(args,data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step)
        # Validate the model on the validation set
        validation_loss = validate(args, val_loader, encoder, decoder, criterion)
        if best_loss is None:
            best_loss = train_loss
        # Save the model checkpoints
        if best_loss < train_loss:
            best_loss = train_loss
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

def train(args,data_loader, encoder, decoder, criterion, optimizer, epoch, best_loss, total_step):
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
            stats = f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Batch time {batch_time.avg:.3f}, Data time {data_time.avg:.3f}, Loss {loss.item():.3f}, Perplexity {np.exp(loss.item()):.3f}'
            # Print training statistics .
            if (i+1) % args.log_step == 0:
                print(stats)
            
    return losses.avg
        
def validate(args,val_loader, encoder, decoder, criterion, epoch, total_step):
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
            # stats = f'Epoch [{epoch}/{args.num_epochs}], Step [{i}/{total_step}], Batch time {batch_time.avg:.3f}, Loss {loss.item():.3f}'

    
    print(f'Validation loss: {losses.avg:.3f}')
    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--train_dir', type=str, default='data/resized2014', help='directory for resized train images')
    parser.add_argument('--val_dir', type=str, default='data/resizedval2014', help='directory for resized val images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json', help='path for val annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # log file
    parser.add_argument('--log_file', type=str, default='training_log.txt', help='name of the log file + .txt')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)