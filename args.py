"""
Arguments - note that these are meant to be called from Google CoLab
"""

import argparse
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int,
                    help="Random seed")
parser.add_argument('--frames_dir', default='./drive/My Drive/Harvard/4S20/GenEd 1049/final_project/uniform_frames-0.2_fps',
                    type=str, help="Directory where video frames are located")
parser.add_argument('--videos_dir', default='./drive/My Drive/Harvard/4S20/GenEd 1049/final_project/uniform_scenes',
                    type=str, help="Directory where videos are located")
parser.add_argument('--datasets_dir', default='./drive/My Drive/Harvard/4S20/GenEd 1049/final_project/datasets',
                    type=str, help="Directory where CSV datasets are located")
parser.add_argument('--batch_size', default=10, type=int,
                    help="Number of videos to load per batch")

args = parser.parse_args()
