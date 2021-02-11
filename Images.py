#loading the images

from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random

import tarfile
import io
import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from arabic_dataset import get_captions_dic


class Dataset(Dataset):
    def __init__(self, captions_path='/Users/arwa/PycharmProjects/Dataset/Dataset/data/Flickr8k_text/Flickr_8k.devImages.txt ', features_path=' /Users/arwa/PycharmProjects/Dataset/Dataset/Flicker8k_Dataset ', split='TRAIN'):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """
        self.split = split
        self.captions=get_captions_dic(split)
        self.features_path=features_path


    def get_encoded_image(self, id):
        """
        Gets an image features by id

        :param id: name of targeted image
        :return: an encoded image
        """

        encoded_image=[]
        return encoded_image


    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.captions.keys())

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        img=[] # image features
        caption=[]
        caption_length=0
        all_captions=[]

        return img, caption, caption_length, all_captions

def main():
        # DataLoader
        loader = torch.utils.data.DataLoader(
            Dataset('/Users/arwa/PycharmProjects/Dataset/Dataset/data/Flickr8k_text/Flickr_8k.devImages.txt ', ' /Users/arwa/PycharmProjects/Dataset/Dataset/Flicker8k_Dataset ', transform=None),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
     if __name__ == "__main__":
             main()
