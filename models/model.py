from models.layer import EncoderVisual, EncoderVisualGraph, Decoder
from models.sublayer import SelfAttention, JointEmbedVideoModel2, AttentionShare, ResBlock
import torch.nn as nn
import torch
import torch.nn.functional as F
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

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)


class CapGnnModel(nn.Module):
    def __init__(self, args, vocab):
        super(CapGnnModel, self).__init__()
        self.use_visual_gan = args.use_visual_gan
        self.encoder = CapGnnEncoder(args)
        self.decoder = Decoder(args, vocab, multi_modal=False)

    def forward(self, visual_feats, region_feats, caption, max_words=None, teacher_forcing_ratio=1.0):
        # frame_feats = visual_feats[:, :, :self.a_feature_size].contiguous()
        # i3d_feats = visual_feats[:, :, -self.m_feature_size:].contiguous()
        obj_proposals, motion_proposals = self.encoder(visual_feats, region_feats)
        # motion_proposals = self.motion_encoder(visual_feats[:, :, -self.m_feature_size:], region_feats)
        outputs = self.decoder(obj_proposals, caption, max_words, teacher_forcing_ratio, motion_proposals)

        return outputs, obj_proposals, motion_proposals

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)

    def load_encoder(self, model, model_path):
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        self.encoder = model.encoder
        self.decoder.word_embed = model.decoder.word_embed
        # self.encoder.eval()
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        for param in self.decoder.word_embed.parameters():
            param.requires_grad = False


class CapGnnEncoder(nn.Module):
    def __init__(self, args):
        super(CapGnnEncoder, self).__init__()
        self.a_feature_size = args.a_feature_size
        # self.m_feature_size = args.m_feature_size
        self.obj_encoder = EncoderVisualGraph(args, input_type='object')
        self.motion_pre_encoder = EncoderVisual(args)
        self.motion_encoder = EncoderVisualGraph(args, input_type='motion', use_embed=False)

    def forward(self, visual_feats, region_feats):
        obj_proposals = self.obj_encoder(visual_feats[:, :, :self.a_feature_size], region_feats)
        motion_input = self.motion_pre_encoder(visual_feats)
        motion_proposals = self.motion_encoder(motion_input, region_feats)
        return obj_proposals, motion_proposals


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



class DiscLanguage(nn.Module):
    def __init__(self, args, vocab_size):
        super(DiscLanguage, self).__init__()
        self.dim = args.gan_word_size
        self.seq_len = args.max_words

        self.block = nn.Sequential(
            ResBlock(self.dim),
            # ResBlock(self.dim),
            # ResBlock(self.dim),
            # ResBlock(self.dim),
            # ResBlock(self.dim),
        )

        self.conv1d = nn.Conv1d(vocab_size, self.dim, 1)

        self.lstm = nn.LSTM(args.gan_word_size, 512, batch_first=True, bidirectional=False)
        self.layer_norm = nn.LayerNorm(512)
        self.lstm_drop = nn.Dropout(0.3)

        self.att = SelfAttention(512, 512, 512, 0.5)
        self.att_norm = nn.LayerNorm(512)
        self.out = nn.Linear(512, 1)

    def forward(self, input, att_mask=None):
        # (BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)

        lstm_out, _ = self.lstm(output.transpose(1, 2))
        lstm_out = self.lstm_drop(self.layer_norm(lstm_out))

        att_out = self.att(lstm_out, att_mask)
        att_out = self.att_norm(att_out)
        # att_out = self.layer_norm_att(att_out)
        att_out = att_out[:, 0, :]
        out = self.out(att_out)
        return out


class DiscVisual(nn.Module):
    def __init__(self, args):
        super(DiscVisual, self).__init__()
        self.mid_size = 512
        self.obj_embed = nn.Linear(args.visual_hidden_size, 512)
        self.motion_embed = nn.Linear(args.visual_hidden_size, 512)
        self.text_linear = nn.Linear(args.word_size, 512)

    def forward(self, text, pos_tags, obj_proposals, motion_proposals):
        # text: bs * T * 300
        # post_gags: 0 => obj, 1 => motion
        bs, win_len, _ = text.size()
        num_psl = obj_proposals.size(1)

        adj_obj = (pos_tags == 0).type(torch.float32)
        adj_motion = (pos_tags == 1).type(torch.float32)

        num_obj = (1/adj_obj.sum(dim=-1))
        num_motion = (1/adj_motion.sum(dim=-1))


        num_obj[num_obj == float("Inf")] = 0.

        num_motion[num_motion == float("Inf")] = 0.

        adj_obj = adj_obj.repeat_interleave(num_psl).view(bs, win_len, -1)
        adj_motion = adj_motion.repeat_interleave(num_psl).view(bs, win_len, -1)

        obj_embed = self.obj_embed(obj_proposals).transpose(-1, -2)
        motion_embed = self.motion_embed(motion_proposals).transpose(-1, -2)
        word_embed = self.text_linear(text)
        att_obj = torch.div(torch.matmul(word_embed, obj_embed), torch.tensor(np.sqrt(self.mid_size)))
        att_motion = torch.div(torch.matmul(word_embed, motion_embed), torch.tensor(np.sqrt(self.mid_size)))

        # att_obj = F.softmax(att_obj, dim=-1)
        # att_motion = F.softmax(att_motion, dim=-1)
        att_obj = att_obj * adj_obj
        att_motion = att_motion * adj_motion

        obj_similarity = att_obj.sum(dim=-1).sum(1)
        motion_similarity = att_motion.sum(dim=-1).sum(1)
        return obj_similarity*num_obj + motion_similarity*num_motion


class DiscVisual2(nn.Module):
    def __init__(self, args, vocab_size):
        super(DiscVisual2, self).__init__()

        # self.bow_emb = nn.Linear(vocab_size, 512)

        self.dim = args.gan_word_size
        self.seq_len = args.max_words

        self.block = nn.Sequential(
            ResBlock(self.dim),
        )

        self.conv1d = nn.Conv1d(vocab_size, self.dim, 1)

        self.lstm = nn.LSTM(args.gan_word_size, 512, batch_first=True, bidirectional=False)
        self.layer_norm = nn.LayerNorm(512)
        self.lstm_drop = nn.Dropout(0.3)

        self.att = SelfAttention(512, 512, 512, 0.5)
        self.att_norm = nn.LayerNorm(512)

        # --------------------------
        self.obj_scorer = JointEmbedVideoModel2(512)
        self.motion_scorer = JointEmbedVideoModel2(512)
        self.attention_obj = AttentionShare(input_value_size=1024, input_key_size=512, output_size=512)
        self.attention_motion = AttentionShare(input_value_size=1024, input_key_size=512, output_size=512)
        self.weighted_score = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, text, pos_tags, obj_proposals, motion_proposals, att_mask=None):
        # obj_embed = self.sent_encoder(text, 0, pos_tags)
        # motion_embed = self.sent_encoder(text, 1, pos_tags)
        # text_bow = make_one_hot_encoding(text, self.vocab_size)[:, 1:]
        # ------old way
        # text = text.sum(dim=1)
        # text[:, 0] = 0
        # text_emb = self.bow_emb(text)
        #  -------- old way

        output = text.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)

        lstm_out, _ = self.lstm(output.transpose(1, 2))
        lstm_out = self.lstm_drop(self.layer_norm(lstm_out))

        att_out = self.att(lstm_out, att_mask)
        att_out = self.att_norm(att_out)
        # att_out = self.layer_norm_att(att_out)
        text_emb = att_out[:, 0, :]

        att_obj, _ = self.attention_obj(obj_proposals, text_emb)
        att_motion, _ = self.attention_motion(motion_proposals, text_emb)
        obj_score = self.obj_scorer(att_obj, text_emb)
        motion_score = self.motion_scorer(att_motion, text_emb)

        weighted_score = self.weighted_score(text_emb)
        output = obj_score * weighted_score + motion_score * (1 - weighted_score)
        return output
