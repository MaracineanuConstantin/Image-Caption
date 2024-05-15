import random
import numpy as np
import torch
import os
from model import EncoderCNN, DecoderRNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed


def save_checkpoint(epoch, encoder, decoder, optimizer, loss, epochs_since_last_improvement, save_path):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'optimizer': optimizer.state_dict(),
             'loss': loss,
             'epochs_since_last_improvement': epochs_since_last_improvement}
    
    filename = f'best_model_{epoch}.pt'
    save_file = os.path.join(save_path, filename)
    torch.save(state, save_file)

def load_checkpoint(checkpoint, embed_size, hidden_size, vocab, num_layers, learning_rate):
    checkpoint = torch.load(checkpoint)
    epoch = checkpoint['epoch']
    # print(f'Loaded checkpoint from epoch {epoch}.')
    encoder = EncoderCNN(embed_size)
    encoder.load_state_dict(checkpoint['encoder'])
    
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    decoder.load_state_dict(checkpoint['decoder'])

    validation_loss = checkpoint['loss']
    epochs_since_last_improvement = checkpoint['epochs_since_last_improvement']

    return epoch, encoder, decoder, validation_loss, epochs_since_last_improvement



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


