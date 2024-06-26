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


## eventual salvat sub fiecare strat numarul/numele layer-ului

# Create a sacred experiment
ex = Experiment("train_experiment")
observers_directory = 'experiments'
ex.observers.append(FileStorageObserver(observers_directory))


@ex.config
def cfg():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_size = 160
    vocab_path = 'data/vocab2017.pkl'
    train_dir = 'data/resizedtrain2017'
    val_dir = 'data/resizedval2017'
    caption_path = 'data/annotations/captions_train2017.json'
    val_caption_path = 'data/annotations/captions_val2017.json'
    log_step = 25
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 30
    batch_size = 128
    num_workers = 0
    learning_rate = 1e-3
    best_bleu_accuracy = None
    best_validation_loss = None
    epochs_since_last_improvement = 0
    start_epoch = 0
    seed = seed_everything(42)
    metrics = [
    load_metric("bleu"),
    load_metric("rouge"),
    load_metric("meteor"),
    load_metric("cider")
    ]

@ex.capture
def train(device, data_loader, encoder, decoder, criterion, optimizer, epoch, total_step, num_epochs, log_step, writer, scorer, vocab):
    encoder.train()
    decoder.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    reference_caption = []
    generated_caption = []

    
    start = time.time()
    for i, (images, captions, lengths) in enumerate(data_loader):
            data_time.update(time.time() - start)
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # captions au padding, lengths au marimile originale (inainte sa fie umplut cu 0 pana la max length)
            # targets = valorile puse 
            # Forward, backward and optimize
            optimizer.zero_grad()
            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

            
            index_outputs = decoder.sample(features)

            generated_caption.extend(index_outputs.cpu().numpy().tolist())
            reference_caption.extend(captions.cpu().numpy().tolist())
            
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


    generated_words = [[vocab.idx2word[word] for word in indexes if vocab.idx2word[word] not in ['<start>', '<end>']] for indexes in generated_caption]
    reference_words = [[vocab.idx2word[word] for word in indexes if vocab.idx2word[word] not in ['<start>', '<end>']] for indexes in reference_caption]

    # convert to strings for metrics evaluation
    generated_words_strings = [' '.join(sentence) for sentence in generated_words]
    reference_words_strings = [' '.join(sentence) for sentence in reference_words]


    scores = scorer(predictions=generated_words_strings, references=reference_words_strings)
    print(f"Train scores are: {scores}")


    ex.log_scalar('Train/Loss/CrossEntropy', losses.avg, epoch)
    ex.log_scalar('Train/Accuracy/Bleu ', scores['bleu']['score'], epoch)
    ex.log_scalar('Train/Accuracy/Rouge1 ', scores['rouge']['rouge1'], epoch)
    ex.log_scalar('Train/Accuracy/Meteor ', scores['meteor']['score'], epoch)
    ex.log_scalar('Train/Accuracy/Cider ', scores['cider']['score'], epoch)

    writer.add_scalar('Train/Loss/CrossEntropy', losses.avg, epoch)
    writer.add_scalar('Train/Accuracy/Bleu ', scores['bleu']['score'], epoch)
    writer.add_scalar('Train/Accuracy/Rouge1 ', scores['rouge']['rouge1'], epoch)
    writer.add_scalar('Train/Accuracy/Meteor ', scores['meteor']['score'], epoch)
    writer.add_scalar('Train/Accuracy/Cider ', scores['cider']['score'], epoch)
    print(f'Training loss: {losses.avg}')
    return losses.avg


@ex.capture
def validate(device, val_loader, encoder, decoder, criterion, epoch, total_step, num_epochs, log_step, writer, vocab, scorer):
    encoder.eval()
    decoder.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()

    generated_caption = []
    reference_caption = []

    start = time.time()

    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)

            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            
            
            output_sample = decoder.sample(features)

            generated_caption.extend(output_sample.cpu().numpy().tolist())
            reference_caption.extend(captions.cpu().numpy().tolist())
            
           
            losses.update(loss.item(), images.size(0))

            stats = f'Validation epoch [{epoch}/{num_epochs}], Step [{i}/{len(val_loader)}], Batch time {batch_time.avg:.3f}, Loss {loss.item():.3f}'
            # Print training statistics .
            if (i+1) % log_step == 0:
                print(stats)

    # convert indexes to words
    generated_words = [[vocab.idx2word[word] for word in indexes if vocab.idx2word[word] not in ['<start>', '<end>']] for indexes in generated_caption]
    reference_words = [[vocab.idx2word[word] for word in indexes if vocab.idx2word[word] not in ['<start>', '<end>']] for indexes in reference_caption]

    # convert to strings for metrics evaluation
    generated_words_strings = [' '.join(sentence) for sentence in generated_words]
    reference_words_strings = [' '.join(sentence) for sentence in reference_words]

    scores = scorer(predictions=generated_words_strings, references=reference_words_strings)
    print(f"Validation scores are: {scores}")

    # bleu_accuracy = scores['bleu']['score']
    bleu_accuracy = scores['rouge']['rouge1']
    ex.log_scalar('Validation/Loss/CrossEntropy', losses.avg, epoch)
    ex.log_scalar('Validation/Accuracy/Bleu ', scores['bleu']['score'], epoch)
    ex.log_scalar('Validation/Accuracy/Rouge1 ', scores['rouge']['rouge1'], epoch)
    ex.log_scalar('Validation/Accuracy/Meteor ', scores['meteor']['score'], epoch)
    ex.log_scalar('Validation/Accuracy/Cider ', scores['cider']['score'], epoch)

    writer.add_scalar('Validation/Loss/CrossEntropy', losses.avg, epoch)
    writer.add_scalar('Validation/Accuracy/Bleu ', scores['bleu']['score'], epoch)
    writer.add_scalar('Validation/Accuracy/Rouge1 ', scores['rouge']['rouge1'], epoch)
    writer.add_scalar('Validation/Accuracy/Meteor ', scores['meteor']['score'], epoch)
    writer.add_scalar('Validation/Accuracy/Cider ', scores['cider']['score'], epoch)
    print(f'Validation loss: {losses.avg:.3f}')
    return losses.avg, bleu_accuracy



@ex.automain
def main(device, crop_size, vocab_path, train_dir, val_dir, caption_path, val_caption_path, 
        log_step, embed_size, hidden_size, num_layers, num_epochs, batch_size, num_workers, learning_rate, _run, best_validation_loss, 
        best_bleu_accuracy, epochs_since_last_improvement, start_epoch, seed, metrics):

    print(f'main {seed}')

    # create log directory to store the encoder & decoder
    log_dir = os.path.join(observers_directory, _run._id)

    writer = SummaryWriter(log_dir=log_dir)

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    

    # Build data loaders
    data_loader = get_loader(train_dir, caption_path, vocab, 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers)

    val_loader = get_loader(val_dir, val_caption_path, vocab, transform, batch_size,
                            shuffle=False, num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    # start_epoch, encoder, decoder, validation_loss, epochs_since_last_improvement = load_checkpoint("experiments/3/best_model_29.pth.tar", embed_size, hidden_size, vocab, num_layers, learning_rate)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()    # = crossentropyloss: -(SUM(ground truth * log(targets)) 
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)


    scorer = NLGMetricverse(metrics=metrics)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(start_epoch, num_epochs):
        start = time.time()

        train_loss = train(device=device, data_loader=data_loader, encoder=encoder, decoder=decoder, criterion=criterion, optimizer=optimizer, epoch=epoch, 
        total_step=total_step, num_epochs=num_epochs,log_step=log_step, writer=writer, scorer=scorer, vocab=vocab)
 
        validation_loss, bleu_accuracy = validate(device=device, val_loader=val_loader, encoder=encoder, decoder=decoder, criterion=criterion, epoch=epoch, total_step=total_step, 
        num_epochs=num_epochs, log_step=log_step, writer=writer, vocab=vocab, scorer=scorer)

        if best_bleu_accuracy is None:
            best_bleu_accuracy = bleu_accuracy

        if best_validation_loss is None:
            best_validation_loss = validation_loss
        
        # Save the model checkpoints

        if bleu_accuracy > best_bleu_accuracy:
            best_bleu_accuracy = bleu_accuracy
            epochs_since_last_improvement = 0
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_since_last_improvement = 0
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)
        
        if epoch == 29:
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)
        
        if epoch == (num_epochs-1):
            save_checkpoint(epoch, encoder, decoder, optimizer, validation_loss, epochs_since_last_improvement, log_dir)

        else:
            epochs_since_last_improvement += 1
        
        if epochs_since_last_improvement == 100:
            break
    
    writer.close()
        
