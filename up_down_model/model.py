"""
this code is adapted from  https://github.com/poojahira/image-captioning-bottom-up-top-down/blob/master/models.py
"""

import torch
from torch import nn
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Decoder(nn.Module):


    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5): ##

        super(Decoder, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout



        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + decoder_dim , decoder_dim, bias=True) # top down attention LSTMCell

        self.att1 = nn.Linear(features_dim, attention_dim) #attention layer
        self.att2 = nn.Linear(decoder_dim, attention_dim)
        self.att3 = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.att4 = nn.Softmax(dim=1)

        self.language_model = nn.LSTMCell(features_dim + decoder_dim, decoder_dim, bias=True)  # language model LSTMCell
        self.word1 = nn.Linear(decoder_dim, vocab_size)
        self.word = nn.Linear(decoder_dim, vocab_size)
        self.act = nn.Softmax(dim=1)

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

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
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

        # Sort input data
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim) 
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up 
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(max(decode_lengths)):

            batch_size_t = sum([l > t for l in decode_lengths])

            h1, c1 = self.top_down_attention(torch.cat([h2[:batch_size_t], image_features_mean[:batch_size_t], embeddings[:batch_size_t, t, :]], dim=1),(h1[:batch_size_t], c1[:batch_size_t]))

            at1 = self.att1(image_features[:batch_size_t])  # (batch_size, 36, attention_dim), att encoder
            at2 = self.att2(h1[:batch_size_t])              # (batch_size, attention_dim) , att decoder
            at3 = self.att3(self.dropout(self.tanh(at1 + at2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
            alpha = self.att4(at3)
            attention_weighted_encoding = (image_features[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)

            h2, c2 = self.language_model(torch.cat([attention_weighted_encoding[:batch_size_t],h1[:batch_size_t]], dim=1),(h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.word(self.dropout(h2)) # (batch_size_t, vocab_size)

            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

