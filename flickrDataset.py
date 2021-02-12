#loading the images
import torch
import pandas as pd
from torch.utils.data import Dataset
from arabic_dataset import get_captions_dic, get_tokenizer, tokenize_captions, create_input_files
import numpy as np
import base64

class Flickr8kDataset(Dataset):
    def __init__(self, features_path="/Users/sarahalhabib/Documents/مستوى ثامن/second/flickr8k_features.tsv", split='TRAIN'):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.captions_dic = get_captions_dic(self.split)
        self.features_path = features_path
        self.tokenizer, self.max_len = get_tokenizer(self.captions_dic)

        # load features
        self.features = pd.read_csv(self.features_path, sep='\t')
        self.features = self.features.to_numpy()
        print("done downloading")

    def get_encoded_image(self, id_name):
        """
        Gets an image features by id

        :param id: name of targeted image
        :return: an encoded image
        """
        img_index = np.where(self.features == id_name)
        img_index = (img_index[0])[0]
        print(id_name, img_index)
        num_boxes = self.features[img_index, 7]
        encoded = self.features[img_index, 9]
        decoded_features = np.frombuffer(base64.b64decode(encoded), np.float32)
        decode_reshape = decoded_features.reshape(num_boxes, 2048)
        decoded_features_tensor = torch.tensor(decode_reshape)
        print(decoded_features_tensor.shape)
        decoded_features_mean = decoded_features_tensor.mean(0)
        print(decoded_features_mean.shape)

        return decoded_features_tensor

    def numeralize_captions(self, id_name, i):
        captions_text = self.captions_dic[id_name]
        print("captions text: ", captions_text)
        all_captions, caplens = tokenize_captions(captions_text, self.tokenizer, self.max_len)
        print("caplens = ", caplens)
        print("all_captions: ", all_captions)
        caption = torch.LongTensor(all_captions[i])
        caplen = torch.LongTensor([caplens[i]])
        all_captions = torch.LongTensor(all_captions)

        return caption, caplen, all_captions

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.captions_dic.keys())*3

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """
        list_id = list(self.captions_dic)
        id_name = list_id[index // 3]

        img = self.get_encoded_image(id_name)  # image features
        caption, caption_length, all_captions = self.numeralize_captions(id_name, index%3)

        if self.split == 'TRAIN':
            return img, caption, caption_length
        else:
            return img, caption, caption_length, all_captions

filename = "/Users/sarahalhabib/Documents/مستوى ثامن/Flickr8k.arabic.full.txt"
create_input_files(filename)
dataset = Flickr8kDataset()
print("dataset len = ", dataset.__len__())
img, caption, caption_length=dataset.__getitem__(448) #149 1
print(img.shape, caption.shape, caption_length.shape)
print(caption, caption_length)