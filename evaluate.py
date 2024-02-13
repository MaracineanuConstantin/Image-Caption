import torch
from model import EncoderCNN, DecoderRNN
from utils import load_checkpoint
from nltk.translate.bleu_score import corpus_bleu
import pickle
from build_vocab import Vocabulary
from tqdm import tqdm
from torchvision import transforms
from data_loader import validation_loader

def evaluate(encoder, decoder, val_dir, val_caption_path, vocab, batch_size, num_workers, device):
    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225))])

    encoder.eval()
    decoder.eval()

     # Lists to store references (true captions), and hypothesis (prediction) for each image
    references = list()
    hypotheses = list()

    val_loader = validation_loader(val_dir, val_caption_path, vocab, transform, batch_size,
                            num_workers=num_workers)
    

    for i, (images, captions, lengths) in enumerate(tqdm(val_loader)):
        images = images.to(device)
        captions = captions.to(device)
        

        features = encoder(images)
        sampled_ids = decoder.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Generate a caption from the image
        feature = encoder(images)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()


        # Convert word_ids to words
        hypotheses_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            hypotheses_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(hypotheses_caption)
        
        hypotheses.append(sentence)

        # convert the captions to cpu
        captions = captions[0].cpu().numpy()
        # Convert word_ids to words
        reference_caption = []
        for word_id in captions:
            word = vocab.idx2word[word_id]
            reference_caption.append(word)
            if word == '<end>':
                break
        ref_sentence = ' '.join(reference_caption)
        references.append(ref_sentence)

    bleu4 = corpus_bleu(references, hypotheses)
    print (f'Rezultatul bleu este: {bleu4}')

def main():
    # config params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_path = 'data/vocab.pkl'
    val_dir = 'data/resizedval2014'
    val_caption_path = 'data/annotations/captions_val2014.json'
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 1
    num_workers = 0
    learning_rate = 0.001
    # teoretic nu ar trebui sa am nevoie de parametrii

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    start_epoch, encoder, decoder, validation_loss, epochs_since_last_improvement = load_checkpoint("experiments/38/best_model.pth.tar", embed_size, hidden_size, vocab, num_layers, learning_rate)

    encoder.to(device)
    decoder.to(device)

    evaluate(encoder, decoder, val_dir, val_caption_path, vocab, batch_size, num_workers, device)


if __name__ == '__main__':
    main()