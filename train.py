"""
this code is adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py
"""
import torch
import torch.optim
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Decoder
from utils import adjust_learning_rate, save_checkpoint, clip_gradient

caption_file = ''
images_features_file = ''

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
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    vocab_size=0

    # read vocab

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = Decoder(attention_dim,emb_dim,decoder_dim,vocab_size=vocab_size,features_dim=2048,dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        # embedding
        # optimizer?
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = DataLoader()

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
              criterion=criterion,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = 0

        # Check if there was an improvement
        is_best = False

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, decoder, criterion, decoder_optimizer, epoch):
    return
