"""
Arguments - note that these are meant to be called from Google CoLab
"""

import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int,
                    help="Random seed")
parser.add_argument('--frames_dir', default='./data/uniform_frames-0.2_fps',
                    type=str, help="Directory where video frames are located")
parser.add_argument('--videos_dir', default='./data/uniform_scenes',
                    type=str, help="Directory where videos are located")
parser.add_argument('--datasets_dir', default='./data/datasets',
                    type=str, help="Directory where CSV datasets are located")
# Training
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help="Learning rate")
parser.add_argument('--batch_size', default=10, type=int,
                    help="Number of videos to load per batch")
parser.add_argument('--epochs', default=10, type=int,
                    help="Number of training epochs")
parser.add_argument('--log_frequency', default=100, type=int,
                    help="Number batches to run before logging training / evaluation metrics")
# Style
parser.add_argument('--gram_ix', default=2, type=int,
                    help="Layer of style VGG to use for gram style calculation")
# Setup
parser.add_argument('--label_type', default='director', type=str,
                    help="Which groundtruth labels to load ('director', 'genre')")
parser.add_argument('--dataset_type', default='frames', type=str,
                    help="Whether to load dataset by frames or video ('frames', 'videos')")

args = parser.parse_args()
