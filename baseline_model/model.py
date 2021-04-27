"""
this code is adapted from  https://github.com/poojahira/image-captioning-bottom-up-top-down/blob/master/models.py
"""

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5):  ##

        super(Decoder, self).__init__()

        self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.att1 = nn.Linear(features_dim, attention_dim)  # attention layer
        self.dropout = nn.Dropout(p=dropout)
        self.language_model = nn.LSTMCell(embed_dim, decoder_dim, bias=True)
        self.word = nn.Linear(decoder_dim, vocab_size)

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.word.bias.data.fill_(0)
        self.word.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim) ###
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Flatten image
        image_features_mean = image_features.mean(1).to(device)  # (batch_size, num_pixels, encoder_dim)
        # image_features_mean = torch.flatten(image_features, start_dim=1)

        # Sort input data
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # intialize hidden state with bottom up features mean
        h1 = self.att1(image_features_mean[:batch_size])

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            h1, c1 = self.language_model(embeddings[:batch_size_t, t, :], (h1[:batch_size_t], c1[:batch_size_t]))
            preds = self.word(self.dropout(h1))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind
