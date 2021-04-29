import torch
from torch.utils.data import Dataset
from dataset.arabic_dataset import get_captions_dic, get_tokenizer, tokenize_captions
import numpy as np
import base64
from PIL import Image
import os
import torchvision.transforms as transforms


class Flickr8kDataset(Dataset):
    def __init__(self, imgs, split='TRAIN', withEncoder=False):
        """
        :param imgs: images' features numpy array or images file if withEncoder
        :param split: data split TRAIN, VAL, or TEST
        """

        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.withEncoder = withEncoder
        self.captions_dic = get_captions_dic(self.split)
        self.tokenizer, self.max_len = get_tokenizer(self.captions_dic)

        if self.withEncoder:
            self.imgs_path = imgs
        else:
            self.features = imgs

    def get_img(self, id_name):
        """
        Gets an image by its id

        :param id_name: name of targeted image
        :return: a transformed image
        """
        transform = transforms.Compose(
            [
                transforms.Resize((356, 356)),
                transforms.RandomCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        img = Image.open(os.path.join(self.imgs_path, id_name+'.jpg')).convert("RGB")
        transformed_img = transform(img)

        return transformed_img

    def get_encoded_image(self, id_name):
        """
        Gets an image features by id

        :param id_name: name of targeted image
        :return: an encoded image
        """
        img_index = np.where(self.features == id_name)
        img_index = (img_index[0])[0]
        num_boxes = self.features[img_index, 7]

        encoded = self.features[img_index, 9]
        decoded_features = np.frombuffer(base64.b64decode(encoded), np.float32)
        decode_reshape = decoded_features.reshape(num_boxes, 2048)
        decoded_features_tensor = torch.from_numpy(decode_reshape)

        return decoded_features_tensor

    def numeralize_captions(self, id_name, i):
        """
        Gets an normalized caption by id and index

        :param id_name: name of targeted image
        :param i: caption index in {0,1,2}
        :return: an encoded image
        """
        captions_text = self.captions_dic[id_name]
        all_captions, caplens = tokenize_captions(captions_text, self.tokenizer, self.max_len)
        caption = torch.LongTensor(all_captions[i])
        caplen = torch.LongTensor([caplens[i]])
        all_captions = torch.LongTensor(all_captions)

        return caption, caplen, all_captions

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        if self.split == "TEST":
            return len(self.captions_dic.keys())
        return len(self.captions_dic.keys())*3

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data
        """
        list_id = list(self.captions_dic)

        if self.split == "TEST":
            id_name = list_id[index]
        else:
            id_name = list_id[index // 3]

        if self.withEncoder:
            img = self.get_img(id_name)
        else:
            img = self.get_encoded_image(id_name)  # image features

        caption, caption_length, all_captions = self.numeralize_captions(id_name, index % 3)

        if self.split == 'TRAIN':
            return img, caption, caption_length
        elif self.split == 'TEST':
            return img, caption, caption_length, all_captions, torch.tensor(index)
        else:
            return img, caption, caption_length, all_captions, torch.tensor(index // 3)