import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from model.HGNN_AC import HGNN_AC


# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h, attn = self.ctr_ntype_layer(inputs)

        h_fc = self.fc(h)
        return h_fc, h, attn


class MAGNN_nc_mb_AC(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dims,
                 emb_dim,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 higher_order_matrix,
                 rnn_type='gru',
                 dropout_rate=0.5,
                 cuda=False,
                 feat_opt=None):
        super(MAGNN_nc_mb_AC, self).__init__()
        self.emb_dim = emb_dim
        self.feat_opt = feat_opt
        self.hidden_dim = hidden_dim
        self.higher_order_matrix = higher_order_matrix
        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims])

        # attribute completion layer
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.elu, num_heads=num_heads, cuda=cuda)

        # feature dropout after attribute completion
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)
    def w_parameter(self):
        self.w_parameter_list = []
        all_parameter_list = list(self.named_parameters())
        for i in range(len(all_parameter_list)):
            if 'attention' not in all_parameter_list[i][0]:
                self.w_parameter_list.append((all_parameter_list[i][0],
                                              all_parameter_list[i][1]))
        for name, param in self.w_parameter_list:
            yield param

    def comp_parameter(self):
        return self.hgnn_ac.parameters()

    def comp(self):
        for p in self.comp_parameter():
            yield p

    def forward(self, inputs1, inputs2):
        adj, feat_list, emb, mask_list, feat_keep_idx, feat_drop_idx, node_type_src = inputs1
        g_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs2

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=adj.device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(feat_list[i])

        feat_src = transformed_features
        # attribute completion


        for i, opt in enumerate(self.feat_opt):
            if opt == 1:
                feat_ac_high = self.hgnn_ac(self.higher_order_matrix[mask_list[i]][:, mask_list[node_type_src]],
                                       emb[mask_list[i]], emb[mask_list[node_type_src]],
                                       feat_src[mask_list[node_type_src]])
                transformed_features[mask_list[i]] = feat_ac_high

        transformed_features = self.feat_drop(transformed_features)
        # hidden layers
        logits, h, attn = self.layer1((g_list, transformed_features, type_mask, edge_metapath_indices_list, target_idx_list))

        return logits, h, attn
