import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            m.weight.data.normal_(std=0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d) and m.affine:
        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)

class TextDiscriminator(nn.Module):
    def __init__(self):
        super(TextDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(300, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
            # nn.Sigmoid(),
        )

    def forward(self, text):
        score = self.model(text)
        return score


class TAGAN_Discriminator(nn.Module):
    def __init__(self):
        super(TAGAN_Discriminator, self).__init__()
        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        self.apply(init_weights)

    def forward(self, img, txt, len_txt, negative=False):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention
        u, m, mask = self._encode_txt(txt, len_txt)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        sim_n = 0
        idx = np.arange(0, img.size(0))
        idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device=txt.device)

        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            if negative:
                W_cond_n, b_cond_n, weight_n = W_cond[idx_n], b_cond[idx_n], weight[i][idx_n]
                sim_n += torch.sigmoid(torch.bmm(W_cond_n, img_feat) + b_cond_n).squeeze(-1) * weight_n
            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        if negative:
            att_txt_n = att_txt[:, idx_n]
            sim_n = torch.clamp(sim_n + self.eps, max=1).t().pow(att_txt_n).prod(0)
        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        if negative:
            return D, sim, sim_n
        return D, sim

    def _encode_txt(self, txt, len_txt):
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask


# img = torch.randn(4, 3, 128, 128).cuda() #batch x channel x w x h
# text = torch.randn(5, 4, 100).cuda() #vocab size x batch x embed size
# len_txt = torch.ones(4).cuda() #no. of words per batch
#
# # txt_m = torch.cat((txt[:, -1, :].unsqueeze(1), txt[:, :-1, :]), 1)
# # len_txt_m = torch.cat((len_txt[-1].unsqueeze(0), len_txt[:-1]))
#
# D = TAGAN_Discriminator()
# D = D.cuda()
#
# x, y = D(img , text, len_txt)
#
# print(x, y)
