# -*- coding: utf-8 -*-


# this code is adapted from https://github.com/ObeidaElJundi/Arabic-Image-Captioning/blob/master/resources/arabic_preprocessing.py


from pyarabic.araby import strip_tatweel, strip_tashkeel
import string
import re
import keras
import pandas as pd
import pickle


class ArabicPreprocessing:

    def __init__(self):
        
        #preparing punctuations list
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        self.all_punctuations = set(arabic_punctuations + english_punctuations)

    def normalize_arabic(self, text):
        text = re.sub(r'\bال(\w\w+)', r'\1', text)  # remove al ta3reef
        text = re.sub("[إأآاٱ]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ة", "ه", text)  # replace ta2 marboota by ha2
        text = re.sub("گ", "ك", text)
        text = strip_tatweel(text) #remove tatweel 
        text = strip_tashkeel(text) #remove tashkeel
        

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
        text = ' '.join([w for w in text.split() if w.isalpha()]) #remove numeric words
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
    process_arab = ArabicPreprocessing()
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


def get_frequent_vocabulary(cpts, vocabulary, frequency=3):
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

    # # create random splits
    # df = pd.DataFrame(list(captions_dic.items()), columns=['images', 'captions'])

    # # split
    # train, validate, test = np.split(df.sample(frac=1, random_state=42),
    #                                   [int(.7 * len(df)), int(.8 * len(df))])

    # the original splits
    train_keys = ((pd.read_csv('Flickr8k_text/Flickr_8k.trainImages.txt', sep='.', header=None)).to_numpy())[:, 0]
    validate_keys = ((pd.read_csv('Flickr8k_text/Flickr_8k.devImages.txt', sep='.', header=None)).to_numpy())[:, 0]
    test_keys = ((pd.read_csv('Flickr8k_text/Flickr_8k.testImages.txt', sep='.', header=None)).to_numpy())[:, 0]

    train_values = [captions_dic[key] for key in train_keys]
    validate_values = [captions_dic[key] for key in validate_keys]
    test_values = [captions_dic[key] for key in test_keys]

    train = pd.DataFrame(dict(zip(train_keys, train_values)).items())
    validate = pd.DataFrame(dict(zip(validate_keys, validate_values)).items())
    test = pd.DataFrame(dict(zip(test_keys, test_values)).items())

    # save
    train.to_csv("Flickr8k_text/train.csv")
    validate.to_csv("Flickr8k_text/validate.csv")
    test.to_csv("Flickr8k_text/test.csv")

    tokenizer = create_tokenizer(captions_dic)
    # save tokenizer object
    with open('dataset/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_captions_dic(split):
    # # get random splits
    # if split == "TRAIN":
    #     df = pd.read_csv("refined_Arabic_Flickr8k_text/flickr_train.csv", index_col=[0])
    #     captions_numpy = df.to_numpy()
    #     captions_dic = dict(captions_numpy)

    # elif split == "VAL":
    #     df = pd.read_csv("refined_Arabic_Flickr8k_text/flickr_validate.csv", index_col=[0])
    #     captions_numpy = df.to_numpy()
    #     captions_dic = dict(captions_numpy)

    # else:
    #     df = pd.read_csv("refined_Arabic_Flickr8k_text/flickr_test.csv", index_col=[0])
    #     captions_numpy = df.to_numpy()
    #     captions_dic = dict(captions_numpy)

    # get the original splits
    if split == "TRAIN":
        df = pd.read_csv("Flickr8k_text/train.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)

    elif split == "VAL":
        df = pd.read_csv("Flickr8k_text/validate.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)

    else:
        df = pd.read_csv("Flickr8k_text/test.csv", index_col=[0])
        captions_numpy = df.to_numpy()
        captions_dic = dict(captions_numpy)
    return captions_dic


def create_tokenizer(captions_dic):
    vocabulary = get_vocabulary(captions_dic)
    frequent_vocabulary = get_frequent_vocabulary(captions_dic, vocabulary, 3)
    num_words = len(frequent_vocabulary) + 1
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>', lower=False,
                                                   filters='')  # num_words=num_words,
    captions_list = []
    for k, image_captions in captions_dic.items():
        for image_caption in image_captions:
            captions_list.append(image_caption)

    tokenizer.fit_on_texts(captions_list)
    tokenizer.word_index['<PAD>'] = 0
    tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if
                            i <= num_words}  # <= because tokenizer is 1 indexed

    return tokenizer


def get_tokenizer(captions_dic):
    # load tokenizer object
    with open('dataset/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    captions_list = []
    for k, image_captions in captions_dic.items():
        image_captions = image_captions.replace("'", "")
        image_captions = image_captions.strip('][').split(', ')
        for image_caption in image_captions:
            captions_list.append(image_caption)

    tokenized = tokenizer.texts_to_sequences(captions_list)
    max_length = calc_max_length(tokenized)

    return tokenizer, max_length


def tokenize_captions(captions_list, tokenizer, max_len):
    captions_list = captions_list.replace("'", "")
    captions_list = captions_list.strip('][').split(', ')
    numeralize = tokenizer.texts_to_sequences(captions_list)
    captions_lengths = [len(i) for i in numeralize]
    pad_sequences = keras.preprocessing.sequence.pad_sequences(numeralize, maxlen=max_len, padding='post')
    return pad_sequences, captions_lengths



