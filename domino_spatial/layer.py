# Implementation of class: AvgReadout and Discriminator is referencing from GraphST: https://doi.org/10.1038/s41467-023-36796-3

import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        # Masked weighted summation
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        # Adjust the shape
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T  
        # Weighted average
        global_emb = vsum / row_sum
          
        # L2 Normalization
        return F.normalize(global_emb, p=2, dim=1)  
    
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        # Positive score
        sc_1 = self.f_k(h_pl, c_x)
        # Negative score
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        # Concatenate positive and negative scores
        logits = torch.cat((sc_1, sc_2), 1)

        return logits
