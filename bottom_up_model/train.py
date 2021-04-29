"""
this code is adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py
and https://github.com/poojahira/image-captioning-bottom-up-top-down/blob/master/train.py
"""
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from up_down_model.model import Decoder
from utils import load_embeddings, adjust_learning_rate, save_checkpoint, AverageMeter, clip_gradient, accuracy
from nltk.translate.bleu_score import corpus_bleu
from dataset.flickrDataset import Flickr8kDataset
import pickle
from random import randint
import pandas as pd
import numpy as np
import time

images_features_file = 'flickr8k_bottomUp_features.tsv'
embeddings_file = 'full_grams_cbow_300_twitter.mdl'
data_name = 'Arabic_flickr8k_3_cap_per_img'

# Model parameters
emb_dim = 300  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Training parameters
start_epoch = 0  # depends on the checkpoint
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # "checkpoint_Arabic_flickr8k_3_cap_per_img.pth.tar" # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    with open('dataset/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    word_map = tokenizer.word_index
    index2word = {v:k for k,v in word_map.items()}
    vocab_size = len(word_map.keys())

    # Read features
    features = pd.read_csv(images_features_file, sep='\t')
    features = features.to_numpy()
    print("done downloading")

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = Decoder(attention_dim,
                          emb_dim,
                          decoder_dim,
                          vocab_size=vocab_size,
                          features_dim=2048,
                          dropout=dropout)
        # embeddings
        embeddings = load_embeddings(embeddings_file, word_map)
        decoder.load_pretrained_embeddings(embeddings)
        decoder.fine_tune_embeddings(True)

        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    # split in {"TRAIN", "VAL", "TEST"}
    train_loader = DataLoader(Flickr8kDataset(imgs=features, split='TRAIN'),
                            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = DataLoader(Flickr8kDataset(imgs=features, split='VAL'),
                            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              criterion_ce=criterion_ce,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation# One epoch's training
        recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion_ce=criterion_ce,
                                index2word=index2word)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, decoder, criterion_ce, decoder_optimizer, epoch):
    """
        Performs one epoch's training.
        :param train_loader: DataLoader for training data
        :param decoder: decoder model
        :param criterion_ce: language layer loss
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward
        scores, sorted_captions, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = sorted_captions[:, 1:]


        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion_ce(scores.data, targets.data)

        # Backpropagation
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, decoder, criterion_ce, index2word):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: language layer loss
    :param index2word: wordmap
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    indexes = list()
    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps, index) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, encoded_captions, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = encoded_captions[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion_ce(scores.data, targets.data)

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<START>'], word_map['<PAD>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            # store images indexes
            for ind in index[sort_ind]:
                indexes.append(ind)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

        rand = randint(0, 500)

        ref_numeric = references[rand]
        hyp_numeric = hypotheses[rand]
        refs = list()
        hyp = list()

        #df = pd.read_csv("flickr_validate.csv", index_col=[0])
        df = pd.read_csv("Flickr8k_text/validate.csv", index_col=[0])
        val_numpy = df.to_numpy()

        for word in hyp_numeric:
            hyp.append(index2word[word])

        for reference in ref_numeric:
            ref = list()
            for word in reference:
                ref.append(index2word[word])
            refs.append(ref)

        print("img_id:", val_numpy[indexes[rand], 0])
        print("reference:", refs[0], "\n", refs[1], "\n", refs[2])
        print("hypotheses:", hyp)


        id_list = list()
        for index in indexes:
            id_list.append(val_numpy[index,0])

        references_list = list()
        hypotheses_list = list()

        for ref_numeric in references:
            refs = list()
            for reference in ref_numeric:
                ref = list()
                for word in reference:
                    ref.append(index2word[word])
                refs.append(ref)
            references_list.append(refs)

        for hyp_numeric in hypotheses:
            hyp = list()
            for word in hyp_numeric:
                hyp.append(index2word[word])
            hypotheses_list.append(hyp)

        results = [id_list] + [hypotheses_list] + [references_list]
        df = pd.DataFrame(np.array(results).T, columns=["id","hypotheses","reference"])
        df.to_csv("results.csv")

    return bleu4


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("time: ", end - start)
