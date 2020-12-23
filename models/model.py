from models.layer import EncoderVisual, Decoder
from models.sublayer import SelfAttention
import torch.nn as nn
import torch
import numpy as np
import random


class CapModel(nn.Module):
    def __init__(self, args, vocab):
        super(CapModel, self).__init__()
        self.encoder = EncoderVisual(args)
        self.decoder = Decoder(args, vocab)

    def forward(self, visual_feats, caption, max_words=None, teacher_forcing_ratio=1.0):
        visual_feats_embed = self.encoder(visual_feats)
        outputs = self.decoder(visual_feats_embed, caption, max_words, teacher_forcing_ratio)
        return outputs



class Disc(nn.Module):
    def __init__(self, opt):
        super(Disc, self).__init__()

        self.lstm = nn.LSTM(opt.word_size, 512, batch_first=True, bidirectional=False)
        self.layer_norm = nn.LayerNorm(512)
        self.lstm_drop = nn.Dropout(0.3)

        self.att = SelfAttention(512, 512, 512, 0.5)
        self.att_norm = nn.LayerNorm(512)
        # self.att = SelfAttention(1024, 1024, 0.2)
        # self.layer_norm_att = nn.LayerNorm(1024)

        self.out = nn.Sequential(
            # self.get_discriminator_block(1024, 512),
            self.get_discriminator_block(512, 256),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    @staticmethod
    def get_discriminator_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, inputs, att_mask):
        # input shape: bs * sentence_len * word_embedding_size
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.lstm_drop(self.layer_norm(lstm_out))

        att_out = self.att(lstm_out, att_mask)
        att_out = self.att_norm(att_out)
        # att_out = self.layer_norm_att(att_out)
        att_out = att_out[:, 0, :]
        out = self.out(att_out)
        return out.squeeze()

