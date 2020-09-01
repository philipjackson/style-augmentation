from __future__ import print_function

import argparse
import os

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from styleaug.stylePredictor import StylePredictor
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',default='pbn',type=str,help='Path to style image directory (make sure the images are in a subdirectory of data_dir or else ImageFolder will complain)')
parser.add_argument('--batchsize',default=8,type=int)
parser.add_argument('--input_size',default=256,type=int,help='Size to resize images to')
parser.add_argument('--checkpoint',default="",type=str,help='Path to style predictor checkpoint')
args = parser.parse_args()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = Compose([Resize((args.input_size,args.input_size)), ToTensor()])

# create the one and only loader:
print("Creating loaders... ", end=' ')
dataset = ImageFolder(args.data_dir,transform=transform)
loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
print("Done")

# create models:
print("Creating models... ", end=' ')
stylePredictor = StylePredictor()
stylePredictor.to(device)
stylePredictor.eval()
print("Done")

# load checkpoint:
checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
stylePredictor.load_state_dict(checkpoint['state_dict_stylepredictor'])


# =================================== MAIN LOOP ===================================

embeddings = np.zeros((len(dataset),100))
i = 0 # number of images processed so far
for images, _ in tqdm(loader):
   
    style_im = images.to(device)
    
    embedding_batch = stylePredictor(style_im)
    embeddings[i:i+embedding_batch.shape[0],:] = embedding_batch.detach().cpu().numpy()
    i += embedding_batch.shape[0]

embeddings = embeddings[:i] # probably there are a few empty rows to chop off here because of drop_last=True



# get mean vector and covariance matrix:
mean = np.mean(embeddings, axis=0)
sigma = np.cov(embeddings, rowvar=False)

# save all embeddings:
np.save(os.path.join('embeddings.npy'), embeddings)

# save mean and convariance:
np.save(os.path.join('mean.npy'), mean)
np.save(os.path.join('covariance.npy'), sigma)
