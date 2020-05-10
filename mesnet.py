import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from models import Encoder, VGG16, BasicNet
from dataloader import load_movie_data
from train import train, evaluate
from args import args


if __name__ == "__main__":

    # torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    style_dims = [64, 128, 256, 512]

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = f'model-{args.label_type}-s={args.seed}-lr={args.learning_rate}-bs={args.batch_size}-mt={args.model_type}-do={args.dropout}'
    if args.load_model:
        model_name += '-e={}.pt'.format(args.load_model_epochs)
    else:
        model_name += '-e={}.pt'.format(args.epochs)

    if args.model_name is not None and args.eval:
        model_name = args.model_name
    PATH = f'./models/{model_name}'

    # Possible classes
    directors = ['Akira Kurosawa', 'Ang Lee', 'Ang Lee', 'Cai Chusheng',
                 'Hideo Nakata', 'Hirokazu Koreeda', 'Isao Takahata',
                 'Ki-young Kim', 'Tsai Ming-liang', 'Wong Kar-wai',
                 'Wu Yonggang', 'Yasujirō Ozu', 'Zhang Yimou']
    genres = ['genre_action', 'genre_adventure', 'genre_animation',
              'genre_comedy', 'genre_crime', 'genre_drama', 'genre_fantasy',
              'genre_horror', 'genre_musical', 'genre_mystery', 'genre_romance',
              'genre_thriller', 'genre_war']

    # Define models
    if args.label_type == 'director':
        criterion = nn.CrossEntropyLoss()
        num_classes = len(directors)
    elif args.label_type == 'genre':
        criterion = nn.BCEWithLogitsLoss()
        num_classes = len(genres)
    
    vgg = None
    if args.model_type == 'vgg-pretrained':
        vgg = VGG16(requires_grad=False).to(args.device)
        model = Encoder(num_classes=num_classes, style_dim=style_dims[args.gram_ix])
    else:
        model = BasicNet(num_classes=num_classes, in_channels=3)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    old_epoch = 0

    if args.load_model:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        old_epoch = checkpoint['epoch']

    # Load data
    print('Loading data...')
    dataloader_train = load_movie_data(split='train', label_type=args.label_type, batch_size=args.batch_size)
    dataloader_val = load_movie_data(split='val', label_type=args.label_type, batch_size=args.batch_size)

    if args.eval:
        embedding_name = f'embedding-{model_name[:-3]}.csv'  # .pt model file suffix
        save_dict = {'frame': [], 'video': [], 'movie': [], 'director': []}
        for genre in genres:
            save_dict[genre] = []
        # Embeddings
        for edim in range(120):  # check BasicNet architecture for this
            save_dict[f'e_{edim}'] = []
        
        # Save for all splits
        dataloader_test = load_movie_data(split='test', label_type=args.label_type, batch_size=args.batch_size)
        evaluate(model, vgg, dataloader_train, criterion, old_epoch, gram_ix=args.gram_ix,
                 split='Train', save_dict=save_dict, genres=genres)
        evaluate(model, vgg, dataloader_val, criterion, old_epoch, gram_ix=args.gram_ix,
                 split='Val', save_dict=save_dict, genres=genres)
        evaluate(model, vgg, dataloader_test, criterion, old_epoch, gram_ix=args.gram_ix,
                 split='Test', save_dict=save_dict, genres=genres)
        # Process saved embeddings and other info
        df = pd.DataFrame(save_dict)
        df.to_csv(f'./embeddings/{embedding_name}', index=False)
        print(f'Embeddings saved to ./embeddings/{embedding_name}')
        raise SystemExit(0)

    for epoch in range(args.epochs):
        train(model, vgg, dataloader_train, criterion, optimizer, old_epoch + epoch, gram_ix=args.gram_ix)
        evaluate(model, vgg, dataloader_val, criterion, old_epoch + epoch, gram_ix=args.gram_ix, split='Val')

        if (epoch + 1) % 10 == 0:
            model_name = model_name.split('-e=')[0] + '-e={}'.format(old_epoch + epoch)
            PATH = f'./models/{model_name}.pt'
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

    model_name = model_name.split('-e=')[0] + '-e={}'.format(old_epoch + args.epochs)
    PATH = f'./models/{model_name}.pt'
    torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)
