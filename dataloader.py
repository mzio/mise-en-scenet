"""
Data for available movies for training the MESnet
"""
import os
import csv
import numpy as np
import pandas as pd

from os.path import join, isfile
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from args import args


class MovieDataset(Dataset):
    """
    Movie dataset class. Per a specific video, loads the stacked frames for further processing
    Args:
        split (str): Which data split to use. ('train', 'val', 'test')
        label (str): Which label type to use. ('director', 'genre')
                     - If 'genre' specified, sets up as multiclass class
        transforms (torchvision.Transforms): Any image transformations
    """

    def __init__(self, split='train', label_type='director', transforms=None):
        self.split = split
        self.label_type = label_type
        self.transforms = transforms

        self.frames_headers = [f'frame_{ix + 1:02d}' for ix in range(24)]
        self.directors = ['Akira Kurosawa', 'Ang Lee', 'Ang Lee', 'Cai Chusheng',
                          'Hideo Nakata', 'Hirokazu Koreeda', 'Isao Takahata',
                          'Ki-young Kim', 'Tsai Ming-liang', 'Wong Kar-wai',
                          'Wu Yonggang', 'Yasujirō Ozu', 'Zhang Yimou']
        self.genres = ['genre_action', 'genre_adventure', 'genre_animation',
                       'genre_comedy', 'genre_crime', 'genre_drama', 'genre_fantasy',
                       'genre_horror', 'genre_musical', 'genre_mystery', 'genre_romance',
                       'genre_thriller', 'genre_war']

        self.load_data()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        try:
            data_row = self.df.iloc[ix]
            images = [self.load_img(data_row, frame_ix)
                      for frame_ix in self.frames_headers]
            images = np.stack(images)
            label = self.load_label(data_row)
            return images, label
        except: 
            if ix + 1 < self.df.shape[0]:
                return self.__getitem__(ix + 1)
        return images, label

    def load_data(self):
        self.df = pd.read_csv(f'{args.datasets_dir}/{self.split}.csv')

    def load_img(self, data_row, frame_ix):
        img = Image.open(f'{args.frames_dir}/{data_row[frame_ix]}')
        img = img.resize((224, 224), resample=Image.LANCZOS)
        img = np.array(img, dtype=np.float32)
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def load_label(self, data_row):
        if self.label_type == 'director':
            director_ix = self.directors.index(data_row['director'])
            return torch.tensor(director_ix, dtype=torch.long)
        elif self.label_type == 'genre':
            return torch.tensor(data_row[genres], dtype=torch.long)

    def get_director(self, ix):
        return self.directors[ix]

    def get_genres(self, indices):
        return np.array(self.genres)[indices]


class MovieFrameDataset(Dataset):
    """
    Movie dataset class by frames. Loads the frames of a specific video for input for further processing
    Args:
        split (str): Which data split to use. ('train', 'val', 'test')
        label (str): Which label type to use. ('director', 'genre')
                     - If 'genre' specified, sets up as multiclass class
        transforms (torchvision.Transforms): Any image transformations
    """
    def __init__(self, split='train', label_type='director', transforms=None):
        self.split = split
        self.label_type = label_type
        self.transforms = transforms
        
        self.directors = ['Akira Kurosawa', 'Ang Lee', 'Ang Lee', 'Cai Chusheng', 
                          'Hideo Nakata', 'Hirokazu Koreeda', 'Isao Takahata', 
                          'Ki-young Kim', 'Tsai Ming-liang', 'Wong Kar-wai', 
                          'Wu Yonggang', 'Yasujirō Ozu', 'Zhang Yimou']
        self.genres = ['genre_action', 'genre_adventure', 'genre_animation', 
                       'genre_comedy', 'genre_crime', 'genre_drama', 'genre_fantasy', 
                       'genre_horror', 'genre_musical', 'genre_mystery', 'genre_romance', 
                       'genre_thriller', 'genre_war']
        
        self.load_data()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ix):
        data_row = self.df.iloc[ix]
        image = self.load_img(data_row)
        label = self.load_label(data_row)
        return image, label

    def load_data(self):
        self.df = pd.read_csv(f'{args.datasets_dir}/{self.split}.csv')

    def load_img(self, data_row):
        img = Image.open(f"{args.frames_dir}/{data_row['frame']}")
        img = img.resize((224, 224), resample=Image.LANCZOS)
        img = np.array(img, dtype=np.float32)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def load_label(self, data_row):
        if self.label_type == 'director':
            director_ix = self.directors.index(data_row['director'])
            return torch.tensor(director_ix, dtype=torch.long)
        elif self.label_type == 'genre':
            return torch.tensor(data_row[genres], dtype=torch.long)
        
    def get_director(self, ix):
        return self.directors[ix]

    def get_genres(self, indices):
        return np.array(self.genres)[indices]


def load_movie_data(split='train', label_type='director', dataset_type='frames', 
                    batch_size=args.batch_size, num_workers=0):
    # Mean and standard deviation of pre-trained model images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset_type == 'frames':
        dataset = MovieFrameDataset(split=split, label_type=label_type, transforms=transform)
    else:
        dataset = MovieDataset(split=split, label_type=label_type, transforms=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers)
    return dataloader
