# -*- coding: utf-8 -*-

from pyarabic.araby import strip_tatweel, strip_tashkeel
import string
from string import punctuation
import re
import keras
import pandas as pd
import numpy as np


class Arabic_preprocessing:

    def __init__(self):
        
        #preparing punctuations list
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        self.all_punctuations = set(arabic_punctuations + english_punctuations)

    def normalize_arabic(self, text):
        text = re.sub("[إأآاٱ]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)  # replace ta2 marboota by ha2
        text = re.sub("گ", "ك", text)
        text = strip_tatweel(text) #remove tatweel 
        text = strip_tashkeel(text) #remove tashkeel
        text = re.sub(r'\bال(\w\w+)', r'\1', text)  # remove al ta3reef

        return text

    def remove_punctuations(self, text):
        return ''.join([c for c in text if c not in self.all_punctuations]) #remove punctuations

    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)

    def remove_english_characters(self, text):
        return re.sub(r'[a-zA-Z]+', '', text)
  
    def preprocess_arabic_text(self, text):
        text = self.remove_punctuations(text)
        text = self.normalize_arabic(text)
        text = self.remove_english_characters(text)
        text = self.remove_repeating_char(text)
        text = ' '.join([w for w in text.split() if len(w)>1 and w.isalpha()]) #remove one-character & numeric words
        return text


def load_data(filename):
  file = open(filename,'r')
  text = file.read()
  file.close()
  return text


def get_captions(file_text):
    """given file content, returns images names and their captions as dictionary"""
    cpts = {}
    #loop through lines
    for line in file_text.split('\n'): # each line contains image name & its caption separated by tab
        #split by tabs
        img_cpt = line.split('\t')
        if len(img_cpt) < 2: continue
        img, cpt = img_cpt
        #remove image extension & index (remove everything befor the dot)
        img_name = img.split('.')[0]
        #add to dictionary
        if img_name not in cpts:
            cpts[img_name] = [cpt]
        else:
            cpts[img_name].append(cpt)
    return cpts


def preprocess_captions(cpts):
    """ clean captions to get rid of useless textual info & reduce vocabulary size. Preprocessing includes:
        - remove punctuations & diacritics
        - normalize (or standarize) Hamza & Ha2
        - remove repeating characters
        - remove english characters
        - remove one-character words
    """
    process_arab = Arabic_preprocessing()
    for img, cpt in cpts.items():
        processed_captions = [process_arab.preprocess_arabic_text(c) for c in cpt]
        cpts[img] = processed_captions


def add_start_end_to_captions(cpts):
    """precede each caption with <START> and end each caption with <END>"""
    start, end = '<START>', '<END>'
    #start, end = 'start', 'end'
    for k, v in cpts.items():
        image_captions = [start + ' ' + cpt + ' ' + end for cpt in v]
        cpts[k] = image_captions

            
def get_vocabulary(cpts):
    """retruns a list of all unique words in captions"""
    captions_flattened = [cpt for image_captions in cpts.values() for cpt in image_captions]
    all_captions = ' '.join(captions_flattened)
    v = set(all_captions.split())
    return sorted(list(v))


def get_frequent_vocabulary(cpts, frequency=5):
    """retruns a list of all unique words that appeared more than `frequency` times"""
    captions_flattened = [cpt for image_captions in cpts.values() for cpt in image_captions]
    all_captions = ' '.join(captions_flattened)
    frequent_vocabulary = []
    for i,v in enumerate(vocabulary):
        if all_captions.count(v) >= frequency: frequent_vocabulary.append(v)
    return frequent_vocabulary


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def create_input_files(file_name):
    # load and preprocess
    captions_file_text = load_data(file_name)
    captions_dic = get_captions(captions_file_text)
    preprocess_captions(captions_dic)
    add_start_end_to_captions(captions_dic)
    df = pd.DataFrame(list(captions_dic.items()), columns=['images', 'captions'])

    # split
    train, validate, test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.7 * len(df)), int(.8 * len(df))])

    # save
    train.to_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_train.csv")
    validate.to_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_validate.csv")
    test.to_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_test.csv")


def get_captions_dic(split):
    if split == "TRAIN":
        df = pd.read_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_train.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)

    elif split == "VALIDATE":
        df = pd.read_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_validate.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)

    else:
        df = pd.read_csv("/Users/sarahalhabib/Documents/مستوى ثامن/flickr_test.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)

    return captions_dic


filename = "/Users/sarahalhabib/Documents/مستوى ثامن/Flickr8k.arabic.full.txt"

captions_file_text = load_data(filename)

captions = get_captions(captions_file_text)
print('Captions #:', len(captions))
print('Caption example:', list(captions.values())[0])

k = '299178969_5ca1de8e40'  # 2660480624_45f88b3022
print('before >>', captions[k])
preprocess_captions(captions)
print('captions preprocessed :)')
print('after >>', captions[k])

# يوضح أن (preprocess_captions(cpts)) تشتغل بشكل صحيح
re.search('\w+', ' 456')
i = 0
for k, v in captions.items():
    print(v)
    i += 1
    if i == 10: break

add_start_end_to_captions(captions)

for k, v in captions.items():
    for cpt in v:
        a = [w for w in cpt.split() if len(w) == 1 and w != 'و']
        if len(a) > 0: print(cpt)

vocabulary = get_vocabulary(captions)
print('Vocabulary size (number of unique words):', len(vocabulary))

frequent_vocabulary = get_frequent_vocabulary(captions, 3)
print('Frequent vocabulary size (number of unique words):', len(frequent_vocabulary))

y = []
for k,image_captions in captions.items():
    for image_caption in image_captions:
        y.append(image_caption)
print(len(y))#24273


num_words = len(frequent_vocabulary) + 1
tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>', lower=False, filters='') # num_words=num_words, 
tokenizer.fit_on_texts(y)
tokenizer.word_index['<PAD>'] = 0
##### fix for keeping only most common `num_words`
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
#####
word2index = tokenizer.word_index
index2word = {v:k for k,v in word2index.items()}
#tokenize captions
y_tok = tokenizer.texts_to_sequences(y)

print(len(frequent_vocabulary), len(word2index))

print('tokenized caption:', y_tok[0])
print('untokenized caption:', tokenizer.sequences_to_texts([y_tok[0]])[0])

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = keras.preprocessing.sequence.pad_sequences(y_tok, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(y_tok)


print(captions[k])

print(y_tok)
print(cap_vector)
print(max_length)

create_input_files(filename)
train_dic = get_captions_dic("TRAIN")

print(train_dic)

