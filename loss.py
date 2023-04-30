#from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyWithOPLoss():
    def __init__(self, op_weight: float=1.0, gamma: float=0.5):
        self.op_weight = op_weight
        self.ce_loss = F.cross_entropy
        self.op_loss = OrthogonalProjectionLoss(gamma=gamma)

    def __call__(self, features, logits, targets):
        loss_op = self.op_loss(features, targets)
        loss_ce = self.ce_loss(logits, targets)

        loss = loss_ce + self.op_weight * loss_op

        return loss

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        # device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        device = features.device

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim
        #print(labels)

        mask = torch.eq(labels, labels.t()).bool().to(device)
        #print(mask)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        #print(eye)

        mask_pos = mask.masked_fill(eye, 0).float()
        #print(mask_pos)
        mask_neg = (~mask).float()
        #print(mask_neg)
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        #neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # gamma=2
        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

if __name__ == '__main__':
    features = torch.randn((5, 2048))
    logits =  torch.randn((5, 2))
    target = torch.randint(2, (5,), dtype=torch.int64)
    loss = CrossEntropyWithOPLoss(op_weight=1, gamma=0.5)
    print(target)
    print(logits.shape)
    # loss = CrossEntropyWithOPLoss(op_weight=0.1, gamma=2)
    value = loss(features, logits, target)