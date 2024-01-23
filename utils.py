import random
import numpy as np
import torch
import os
from model import EncoderCNN, DecoderRNN

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed


def save_encoder(epoch, model, optimizer, loss, epochs_since_last_improvement, save_path):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss' : loss,
             'epochs_since_last_improvement' : epochs_since_last_improvement}

    filename = f'best_encoder.pth.tar'
    save_file = os.path.join(save_path,filename)
    torch.save(state, save_file)


def save_decoder(epoch, model, optimizer, loss, epochs_since_last_improvement, save_path):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss' : loss,
             'epochs_since_last_improvement' : epochs_since_last_improvement}

    filename = f'best_decoder.pth.tar'
    save_file = os.path.join(save_path,filename)
    torch.save(state, save_file)


def load_checkpoint_encoder(checkpoint, embed_size):
    checkpoint = torch.load(checkpoint)
    epoch = checkpoint['epoch']
    print('\nLoaded checkpoint from epoch %d.\n' % epoch)
    encoder = EncoderCNN(embed_size)
    encoder.load_state_dict(checkpoint['model'])

    validation_loss = checkpoint['loss']
    epochs_since_last_improvement = checkpoint['epochs_since_last_improvement']

    return epoch, encoder, validation_loss, epochs_since_last_improvement

def load_checkpoint_decoder(checkpoint,embed_size, hidden_size, vocab, num_layers, encoder, learning_rate):
    checkpoint = torch.load(checkpoint)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    decoder.load_state_dict = checkpoint['model']

    # daca pastrez optimizer aici (aici ci nu in encoder pentru ca aici am acces si la decoder si la encoder) imi da eroare 
    # de device in train
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    # optimizer = torch.optim.Adam(params, lr=learning_rate)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    return decoder

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


