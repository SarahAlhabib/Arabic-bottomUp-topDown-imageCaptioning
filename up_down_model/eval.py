import torch.optim
from torch.utils.data import DataLoader
from dataset.flickrDataset import Flickr8kDataset
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval
import pickle
import pandas as pd
import numpy as np
import time

# Parameters
caption_file = 'Flickr8k.arabic.full.tsv'
images_features_file = 'flickr8k_bottomUp_features.tsv'
embeddings_file = 'full_grams_cbow_300_twitter.mdl'
data_name = 'Arabic_flickr8k_3_cap_per_img'

checkpoint_file = None #"checkpoint_Arabic_flickr8k_3_cap_per_img.pth.tar" # model checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Read word map
with open('dataset/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

word_map = tokenizer.word_index
index2word = {v :k for k ,v in word_map.items()}
vocab_size = len(word_map.keys())

# Read features
features = pd.read_csv(images_features_file, sep='\t')
features = features.to_numpy()
print("done downloading")

# Load model
# torch.nn.Module.dump_patches = True #line added
checkpoint = torch.load(checkpoint_file, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

nlgeval = NLGEval()  # loads the evaluator
batch_size = 1
workers = 1  # for data-loading; right now, only 1 works with h5py


def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """
    # DataLoader
    Test_loader = DataLoader(Flickr8kDataset(imgs=features, split='TEST'),
                             batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    indexes = list()

    # For each image
    for i, (imgs, caps, caplens, allcaps, index) in enumerate(
            tqdm(Test_loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        imgs = imgs.to(device)  # (1, 3, 256, 256)
        imgs_mean = imgs.mean(1)
        imgs_mean = imgs_mean.expand(k ,2048)
        # compute mean here instead of normalize before
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<START>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)
        # two LSTM so two decoder
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            h1 ,c1 = decoder.top_down_attention(
                torch.cat([h2 ,imgs_mean ,embeddings], dim=1),
                (h1 ,c1))  # (batch_size_t, decoder_dim)
            at1 = decoder.att1(imgs)
            at2 = decoder.att2(h1)
            at3 = decoder.att3(decoder.tanh(at1 + at2.unsqueeze(1))).squeeze(2)  # (batch_size, 36)
            alpha= decoder.att4(at3)
            attention_weighted_encoding = (imgs * alpha.unsqueeze(2)).sum(dim=1)
            h2 ,c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding ,h1], dim=1) ,(h2 ,c2))

            scores = F.log_softmax(decoder.word(h2), dim=1)  # (s, vocab_size)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            prev_word_inds = torch.LongTensor(prev_word_inds.to("cpu")).to(device)
            next_word_inds = torch.LongTensor(next_word_inds.to("cpu")).to(device)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            imgs_mean = imgs_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [index2word[w] for w in c if w not in {word_map['<START>'], word_map['<END>'], word_map['<PAD>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        # print(img_caps)
        references.append(img_caps)

        # Hypotheses
        hypothesis = \
        ([index2word[w] for w in seq if w not in {word_map['<START>'], word_map['<END>'], word_map['<PAD>']}])
        hypothesis = ' '.join(hypothesis)
        # print(hypothesis)
        hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

        # store images indexes
        for ind in index:
            indexes.append(ind)

    # creat resutls.csv
    df = pd.read_csv("Flickr8k_text/test.csv", index_col=[0])
    test_numpy = df.to_numpy()

    id_list = list()
    for index in indexes:
        id_list.append(test_numpy[index, 0])

    results = [id_list] + [hypotheses] + [references]
    df = pd.DataFrame(np.array(results).T, columns=["id", "hypotheses", "reference"])
    df.to_csv("results.csv")

    # Calculate scores
    metrics_dict = nlgeval.compute_metrics(references, hypotheses)

    return metrics_dict


if __name__ == '__main__':
    start = time.time()
    beam_size = 5
    metrics_dict = evaluate(beam_size)
    end = time.time()
    print("metrics_dict", metrics_dict)
    print("time: ", end - start)