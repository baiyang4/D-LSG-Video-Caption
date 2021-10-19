from models.layer import EncoderVisual, EncoderVisualGraph, Decoder, EncoderVisualGAT, EncoderVisualGraphTUN, PSLScore, PSLScore2
from models.sublayer import SelfAttention, JointEmbedVideoModel2, AttentionShare, ResBlock, LatentGNN, LatentPSL
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
        outputs, _ = self.decoder(visual_feats_embed, caption, max_words, teacher_forcing_ratio)
        return outputs

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)


class CapGnnModel(nn.Module):
    def __init__(self, args, vocab):
        super(CapGnnModel, self).__init__()
        self.use_visual_gan = args.use_visual_gan
        self.encoder = CapGnnEncoder(args)
        self.decoder = Decoder(args, vocab, multi_modal=True)

    def forward(self, visual_feats, region_feats, caption, max_words=None, teacher_forcing_ratio=1.0):
        # frame_feats = visual_feats[:, :, :self.a_feature_size].contiguous()
        # i3d_feats = visual_feats[:, :, -self.m_feature_size:].contiguous()
        obj_proposals, motion_proposals = self.encoder(visual_feats, region_feats)
        # motion_proposals = self.motion_encoder(visual_feats[:, :, -self.m_feature_size:], region_feats)
        outputs, alpha_all = self.decoder(obj_proposals, caption, max_words, teacher_forcing_ratio, motion_proposals)
        if len(alpha_all) > 0:
            alpha_all = torch.cat(alpha_all, dim=-1).transpose(1,2)
        return outputs, obj_proposals, motion_proposals, alpha_all

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
    def __init__(self, args, baseline=False):
        super(CapGnnEncoder, self).__init__()
        self.a_feature_size = args.a_feature_size
        # self.m_feature_size = args.m_feature_size
        # self.obj_encoder = EncoderVisualGraph(args, input_type='object', baseline=baseline)
        # self.obj_encoder = EncoderVisualGAT(args, input_type='object', baseline=baseline)
        self.obj_encoder = EncoderVisualGraphTUN(args, input_type='object', baseline=baseline)
        self.motion_pre_encoder = EncoderVisual(args)
        # self.motion_encoder = EncoderVisualGraph(args, input_type='motion', use_embed=False, baseline=baseline)
        # self.motion_encoder = EncoderVisualGAT(args, input_type='motion', use_embed=False, baseline=baseline)
        self.motion_encoder = EncoderVisualGraphTUN(args, input_type='motion', use_embed=False, baseline=baseline)

    def forward(self, visual_feats, region_feats):
        obj_proposals = self.obj_encoder(visual_feats[:, :, :self.a_feature_size], region_feats)
        motion_input = self.motion_pre_encoder(visual_feats)
        motion_proposals = self.motion_encoder(motion_input, region_feats)
        return obj_proposals, motion_proposals


class CapBaselineModel(nn.Module):
    def __init__(self, args, vocab):
        super(CapBaselineModel, self).__init__()
        self.use_visual_gan = args.use_visual_gan
        self.encoder = CapGnnEncoder(args, baseline=True)
        self.linear_baseline = nn.Linear(args.visual_hidden_size*2, args.visual_hidden_size)
        self.decoder = Decoder(args, vocab, multi_modal=False, baseline=True)

    def forward(self, visual_feats, region_feats, caption, max_words=None, teacher_forcing_ratio=1.0):
        obj_proposals, motion_proposals = self.encoder(visual_feats, region_feats)
        # step_feats = self.linear_baseline(torch.cat([obj_proposals, motion_proposals], dim=-1))
        outputs, _ = self.decoder(motion_proposals, caption, max_words, teacher_forcing_ratio)
        return outputs, 0, 0, 0

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)


class CapBaseline1(nn.Module):
    def __init__(self, args, vocab):
        super(CapBaseline1, self).__init__()
        self.use_visual_gan = args.use_visual_gan
        self.encoder = EncoderVisual(args, baseline=True)
        self.decoder = Decoder(args, vocab, multi_modal=False, baseline=True)

    def forward(self, visual_feats, region_feats, caption, max_words=None, teacher_forcing_ratio=1.0):
        visual_feats_encode = self.encoder(visual_feats)
        outputs, _ = self.decoder(visual_feats_encode, caption, max_words, teacher_forcing_ratio)
        return outputs, 0, 0, 0

    def update_beam_size(self, beam_size):
        self.decoder.update_beam_size(beam_size)


class DiscV2(nn.Module):
    def __init__(self, opt, vocab_size):
        super(DiscV2, self).__init__()
        self.dim = 512
        self.num_top = opt.num_topk
        self.seq_len = opt.max_words
        self.num_psl = opt.num_proposals

        self.block = nn.Sequential(
            ResBlock(self.dim),
        )

        self.conv1d = nn.Conv1d(vocab_size, self.dim, 1)
        self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=False)
        self.layer_norm = nn.LayerNorm(512)
        self.lstm_drop = nn.Dropout(0.3)

        self.att = SelfAttention(512, 512, 512, 0.3)
        self.att_norm = nn.Sequential(
            nn.Tanh(),
            nn.LayerNorm(512)
        )
        self.motion_psl_score = PSLScore2(opt.num_proposals, self.num_top)
        self.obj_psl_score = PSLScore2(opt.num_proposals, self.num_top)
        self.text_sum = LatentPSL(512, 1)
        self.fusion = nn.Parameter(torch.empty(size=(2, 512)))
        nn.init.xavier_uniform_(self.fusion, gain=nn.init.calculate_gain('tanh'))

    @staticmethod
    def get_discriminator_block(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, inputs, obj_proposals, motion_proposals, att_mask=None, alpha_all=None):
        # input shape: bs * sentence_len * word_embedding_size
        output = inputs.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        # att_out=output.transpose(1, 2)
        # lstm_out, _ = self.lstm(inputs)
        lstm_out, _ = self.lstm(output.transpose(1, 2))
        lstm_out = self.lstm_drop(self.layer_norm(lstm_out))

        att_out = self.att(lstm_out, att_mask)
        att_out = self.att_norm(att_out)

        seq_mask = att_mask[:, 0, :].unsqueeze(dim=2).repeat(1, 1, self.num_psl*2)
        alpha_all = alpha_all * seq_mask
        seq_mask_spl = att_mask[:, 0, :].unsqueeze(dim=2).repeat(1, 1, self.num_top)
        obj_score_out = self.obj_psl_score(obj_proposals, alpha_all[:, :, :self.num_psl], att_out, seq_mask_spl)
        motion_score_out = self.motion_psl_score(motion_proposals, alpha_all[:, :, -self.num_psl:], att_out, seq_mask_spl)
        sent_sum = self.text_sum(att_out).squeeze()  # 128, 512
        fusion_score = torch.matmul(sent_sum, self.fusion.T)  # 128, 2, 1
        fusion_score = F.softmax(fusion_score, dim=-1)
        score_final = obj_score_out * fusion_score[:, 0] + motion_score_out * fusion_score[:, 1]
        # score_out = score_out.mean(dim=-1)
        return score_final

