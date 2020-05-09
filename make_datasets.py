import csv
import pandas as pd
import numpy as np

import os
from os.path import isfile, join

from args import args


movie_to_director = {
    'chungking_express': 'Wong Kar-wai',
    'crouching_tiger_hidden_dragon': 'Ang Lee',
    'goddess': 'Wu Yonggang',
    'grave_of_the_fireflies': 'Isao Takahata',
    'i_was_born_but': 'Yasujirō Ozu',
    'new_woman': 'Cai Chusheng',
    'rashomon': 'Akira Kurosawa',
    'ringu': 'Hideo Nakata',
    'the_wedding_banquet': 'Ang Lee',
    'to_live': 'Zhang Yimou',
    'the_housemaid': 'Ki-young Kim',
    'the_hole': 'Tsai Ming-liang',
    'after_life': 'Hirokazu Koreeda'
}

movie_to_genre = {
    'chungking_express': ['comedy', 'crime', 'drama'],
    'crouching_tiger_hidden_dragon': ['action', 'adventure', 'fantasy'],
    'goddess': ['drama'],
    'grave_of_the_fireflies': ['animation', 'drama', 'war'],
    'i_was_born_but': ['comedy', 'drama'],
    'new_woman': ['drama'],
    'rashomon': ['crime', 'drama', 'mystery'],
    'ringu': ['horror', 'mystery'],
    'the_wedding_banquet': ['comedy', 'drama', 'romance'],
    'to_live': ['drama', 'war'],
    'the_housemaid': ['crime', 'drama', 'thriller'],
    'the_hole': ['drama', 'fantasy', 'musical'],
    'after_life': ['drama', 'fantasy']
}

def get_data(sets, all_frames, csv_columns, split='train'):
    if split == 'train':
        set_ix = 0
    elif split == 'val':
        set_ix = 1
    elif split == 'test':
        set_ix = 2
    else:
        raise ValueError
    set_dict_data = []
    # scenes = list(map(lambda x: x[:-4], sets[set_ix]))
    scenes = sets[set_ix]

    for scene in scenes:
        movie = scene.split('-')[1]
        director = movie_to_director[movie]
        genres = movie_to_genre[movie]
        for ix, frame in enumerate([f for f in all_frames if scene in f]):
            dict_data = {x: 0 for x in csv_columns}
            dict_data['frame'] = frame
            dict_data['movie'] = movie
            dict_data['director'] = director
            dict_data['video'] = f'{scene}.mp4'
            for g in genres:
                dict_data[f'genre_{g}'] = 1

            set_dict_data.append(dict_data)
    return set_dict_data

if __name__ == "__main__":
    
    np.random.seed(args.seed)
    all_frames = [f for f in os.listdir(args.frames_dir) if isfile(join(args.frames_dir, f))]

    # Organize by video (Want to have even representation across all splits)
    dict_videos = {film: [] for film in movie_to_genre}
    
    total_videos = 0
    for f in all_frames:
        film_name = f.split('-')[1]
        video_name = f.split('-frame')[0]
        
        if video_name not in dict_videos[film_name]:
            dict_videos[film_name].append(video_name)
            total_videos += 1

    # Shuffle scenes and split into train, val, test sets
    for videos in dict_videos.values():
        np.random.shuffle(videos)

    train_set = []
    val_set = []
    test_set = []
    
    for videos in dict_videos.values():
        num_videos = len(videos)
        train_set.extend(videos[:int(num_videos * 0.6)])
        val_set.extend(videos[int(num_videos * 0.6): int(num_videos * 0.8)])
        test_set.extend(videos[-(num_videos - int(num_videos * 0.8)):]) 

    for set_ in [train_set, val_set, test_set]:
        np.random.shuffle(set_)

    # Sanity check
    print(f'Total number of videos {total_videos}')
    print(f'Total number of videos (hopefully again): {len(train_set) + len(val_set) + len(test_set)}')

    print(f'Train: {len(train_set) / (len(train_set) + len(val_set) + len(test_set))}')
    print(f'Val: {len(val_set) / (len(train_set) + len(val_set) + len(test_set))}')
    print(f'Test: {len(test_set) / (len(train_set) + len(val_set) + len(test_set))}')

    # Setup CSVs
    ##  Get all genres
    all_genres = []
    for movie, genres in movie_to_genre.items():
        all_genres.extend([f'genre_{g}' for g in genres])
    all_genres = sorted(list(set(all_genres)))

    ## Setup columns
    csv_columns = ['frame', 'video', 'director', 'movie']
    csv_columns.extend(all_genres)
    print(csv_columns)

    ## Get datasets
    train = get_data([train_set, val_set, test_set], all_frames, csv_columns, split='train')
    val = get_data([train_set, val_set, test_set], all_frames, csv_columns, split='val')
    test = get_data([train_set, val_set, test_set], all_frames, csv_columns, split='test')

    datasets = {'train': train, 'val': val, 'test': test}
    for split, dataset in datasets.items():
        with open(f'{args.datasets_dir}/{split}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            for data in dataset:
                writer.writerow(data) 
