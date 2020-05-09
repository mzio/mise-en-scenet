import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import Encoder, VGG16
from dataloader import load_movie_data
from train import train, evaluate
from args import args


if __name__ == "__main__":

    # torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    style_dims = [64, 128, 256, 512]

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define models
    vgg = VGG16(requires_grad=False).to(args.device)
    model = Encoder(num_classes=13, style_dim=style_dims[args.gram_ix])
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load data
    print('Loading data...')
    dataloader_train = load_movie_data(split='train', label_type='director', batch_size=args.batch_size)
    dataloader_val = load_movie_data(split='val', label_type='director', batch_size=args.batch_size)

    for epoch in range(args.epochs):
        train(model, vgg, dataloader_train, criterion, optimizer, epoch, gram_ix=args.gram_ix)
        evaluate(model, vgg, dataloader_val, criterion, epoch, gram_ix=args.gram_ix, split='Val')
