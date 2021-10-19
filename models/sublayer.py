import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


class AttentionShare(nn.Module):
    def __init__(self, input_value_size, input_key_size, output_size, dropout=0.1):
        super(AttentionShare, self).__init__()
        self.input_value_size = input_value_size
        self.input_key_size = input_key_size
        self.attention_size = output_size
        self.dropout = dropout

        self.K = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.Q = nn.Linear(in_features=input_key_size, out_features=output_size, bias=False)
        self.V = nn.Linear(in_features=input_value_size, out_features=output_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            nn.Tanh(),
            nn.LayerNorm(output_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, meta_state, hidden_previous):
        K = self.K(meta_state)
        Q = self.Q(hidden_previous).unsqueeze(2)
        V = self.V(meta_state).transpose(-1, -2)

        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        weight = F.softmax(logits, dim=1)
        # weight = F.sigmoid(logits)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)

        attention = mid_step.squeeze(2)

        attention = self.output_layer(attention)

        return attention, weight


class SelfAttention(nn.Module):
    def __init__(self, input_size, attention_size, output_size, dropout=0.2, get_pe=False):
        super(SelfAttention, self).__init__()

        self.attention_size = attention_size
        self.dropout = dropout
        self.get_pe = get_pe
        self.pe = PositionalEncoding_old(attention_size)
        self.K = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.Q = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.V = nn.Linear(in_features=input_size, out_features=self.attention_size, bias=False)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.attention_size, out_features=output_size, bias=False),
            # nn.Tanh(),
            nn.Dropout(self.dropout)
        )

    def forward(self, x, att_mask=None):
        if self.get_pe:
            x = self.pe(x)
        K = self.K(x)
        Q = self.Q(x).transpose(-1, -2)
        V = self.V(x).transpose(-1, -2)
        logits = torch.div(torch.matmul(K, Q), torch.tensor(np.sqrt(self.attention_size)))
        if att_mask is not None:
            zero_vec = -9e15 * torch.ones_like(logits)
            logits = torch.where(att_mask > 0, logits, zero_vec)
            # logits = logits * att_mask
        weight = F.softmax(logits, dim=-1)
        weight = weight.transpose(-1, -2)
        mid_step = torch.matmul(V, weight)
        # mid_step = torch.matmul(V, weight)
        attention = mid_step.transpose(-1, -2)

        attention = self.output_layer(attention)

        return attention


class PositionalEncoding_old(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.2, max_len=72):
        super(PositionalEncoding_old, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 3, padding=1),  # nn.Linear(DIM, DIM),
            # nn.ReLU(True),
            # nn.Conv1d(dim, dim, 3, padding=1),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.adj_Q = nn.Linear(2048,2048)
        self.adj_K = nn.Linear(2048, 2048)
        self.graph_update = nn.Linear(2048, 1024)

    def forward(self, region_feats):
        win_len = region_feats.size(1)
        num_obj = region_feats.size(2)
        bs = region_feats.size(0)
        feature_size = region_feats.size(-1)
        try:
            feats = region_feats.contiguous().view(bs, win_len * num_obj, feature_size)
        except:
            feats = None
            print('hehe')
        adj_Q = self.adj_Q(feats)
        adj_K = self.adj_K(feats).transpose(2,1)
        adj = torch.matmul(adj_Q, adj_K)
        adj_norm = F.softmax(adj, dim=-1)
        region_feats_gnn = torch.matmul(adj_norm, self.graph_update(feats))
        region_feats_gnn = region_feats_gnn.view(bs, win_len, num_obj, -1)
        return region_feats_gnn


class LatentGNN(nn.Module):
    def __init__(self, input_size, num_latent, norm_func):
        super(LatentGNN, self).__init__()
        self.norm_func = F.normalize
        self.v2l_adj_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=num_latent,
                      kernel_size=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_latent),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_seq, mask=None):
        v2l_graph_adj = self.v2l_adj_conv(input_seq.permute(0, 2, 1).unsqueeze(dim=2))
        # print(v2l_graph_adj.shape)
        v2l_graph_adj = v2l_graph_adj.squeeze(dim=2)
        # print(v2l_graph_adj.shape)

        if mask is not None:
            zero_vec = torch.zeros_like(v2l_graph_adj)
            v2l_graph_adj = torch.where(mask > 0, v2l_graph_adj, zero_vec)
        # print('v2l_graph_adj shape = ', v2l_graph_adj.shape)
        v2l_graph_adj = self.norm_func(v2l_graph_adj, dim=2)

        latent_proposals = torch.matmul(v2l_graph_adj, input_seq)
        return latent_proposals


class LatentPSL(nn.Module):
    def __init__(self, input_size, num_psl):
        super(LatentPSL, self).__init__()

        # 建立都是0的矩阵，大小为（输入维度，输出维度）
        self.theta = nn.Parameter(torch.empty(size=(num_psl, input_size)))
        nn.init.xavier_uniform_(self.theta, gain=nn.init.calculate_gain('tanh'))
        self.out_norm = nn.Sequential(
            nn.Tanh(),
            nn.LayerNorm(input_size),
            nn.Dropout(0.3)
        )

    def forward(self, input_seq, mask=None):
        # input_seq.shape = (bs, seq_len, d_hidden)
        v2l_graph_adj = torch.matmul(input_seq, self.theta.T)  # (bs, seq_len, num_psl)
        v2l_graph_adj = F.softmax(v2l_graph_adj, dim=1)
        # v2l_graph_adj = self.att_drop(v2l_graph_adj)

        out_seq = torch.matmul(v2l_graph_adj.transpose(-1,-2), input_seq)  # (bs, num_psl, d_hidden)
        out_seq = self.out_norm(out_seq)

        return out_seq

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        # 学习因子
        self.alpha = alpha
        self.concat = concat

        # 建立都是0的矩阵，大小为（输入维度，输出维度）
        self.Ws = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # xavier初始化
        nn.init.xavier_uniform_(self.Ws, gain=nn.init.calculate_gain('relu'))

        self.We = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # xavier初始化
        nn.init.xavier_uniform_(self.We, gain=nn.init.calculate_gain('relu'))

        # 这里的self.a,对应的是论文里的向量a，故其维度大小应该为(2*out_features, 1)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, start_feature, end_feature):
        # start_feature.shape: (N, num_nodes_start, in_features), end_feature.shape: (N, num_nodes_end, out_features)
        # h.shape: torch.Size([2708, 8]) 8是label的个数
        Ws = torch.matmul(start_feature, self.Ws)
        We = torch.matmul(end_feature, self.We)
        a_input = self._prepare_attentional_mechanism_input(Ws, We)

        # 即论文里的eij
        # squeeze除去维数为1的维度
        # [2708, 2708, 16]与[16, 1]相乘再除去维数为1的维度，故其维度为[2708,2708],与领接矩阵adj的维度一样
        attention = self.leakyrelu(torch.matmul(a_input, self.a))

        # 对应论文公式3，attention就是公式里的a_ij
        attention = F.softmax(attention, dim=1).squeeze()
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention.transpose(1,2), Ws) + We

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Ws, We):
        N1 = Ws.size(1)  # number of nodes
        N2 = We.size(1)
        bs = We.size(0)
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times

        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times

        # https://www.jianshu.com/p/a2102492293a

        Ws_repeated_in_chunks = Ws.repeat_interleave(N2, dim=1)
        We_repeated_alternating = We.repeat(1, N1, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Ws_repeated_in_chunks, We_repeated_alternating], dim=-1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(bs, N1, N2, -1)


class JointEmbedVideoModel2(nn.Module):
    def __init__(self, hidden_size):
        super(JointEmbedVideoModel2, self).__init__()
        self.classify = nn.Linear(hidden_size, 1)
        self.visual_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.sent_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self,visual,sent):
        return self.classify(self.visual_embed(visual) * self.sent_embed(sent))
