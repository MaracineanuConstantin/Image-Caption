import os
import torch
from data_loader import validation_loader
import random
import shutil
from torchvision import transforms
from build_vocab import Vocabulary
import pickle
from tqdm import tqdm

val_dir = 'data/resizedval2014'
val_caption_path = 'data/annotations/captions_val2014.json'
vocab_path = 'data/vocab.pkl'
batch_size = 128
crop_size = 224
num_workers = 0
test_photos_path = './data/SPLITtest/'
val_photos_path = './data/SPLITval/'

transform = transforms.Compose([ 
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)



val_loader = validation_loader(val_dir, val_caption_path, vocab, transform, batch_size,
                            num_workers=num_workers)


image_ids = list(val_loader.dataset.coco.imgs.keys())
random.shuffle(image_ids)

split_index = len(image_ids)//2
test_images_ids = image_ids[:split_index]
val_images_ids = image_ids[split_index:]


if not os.path.exists(test_photos_path):
    os.makedirs(test_photos_path)

if not os.path.exists(val_photos_path):
    os.makedirs(val_photos_path)


for image_id in tqdm(test_images_ids):
    filename = val_loader.dataset.coco.imgs[image_id]['file_name']
    original_dir = os.path.join(val_dir, filename)
    shutil.copy(original_dir, os.path.join(test_photos_path, filename))


for image_id in tqdm(val_images_ids):
    filename = val_loader.dataset.coco.imgs[image_id]['file_name']
    original_dir = os.path.join(val_dir, filename)
    shutil.copy(original_dir, os.path.join(val_photos_path, filename))

