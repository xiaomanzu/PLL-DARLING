"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # (bs,1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # (bs,bs)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 2*bs,embedding_dim
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), # 2*bs,2*bs
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # all < 0 # 因为对角线上是自己和自己，所以余弦相似度最大

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # (bs,bs) -> (bs*2,bs*2)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) # == 1- torch.eye(mask.shape[0])
        mask = mask * logits_mask

        # compute log_prob

        exp_logits = torch.exp(logits) * logits_mask # no identity
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) # 已经是论文中所求的结果

        # compute mean of log-likelihood over positive

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) # 分子都是一对样本具备相同语义且不是同一对样本/分母都是一对样本具备相同语义且不是同一对样本

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class PLLLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self,domain_number,temperature=0.07):
        super(PLLLoss, self).__init__()
        self.temperature = temperature
        self.domain_number = domain_number

    def forward(self, features, domain_label , w_matrix): # (batchsize, [original + augmented  = 2], embedding_dim), (bathsize, 1), (batchsize, domain_number)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 2*bs,embedding_dim
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        # compute logits

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), # (2*bs,2*bs)
            self.temperature)
        # T  T  F   F
        # 11 12 11; 12;
        # 21 22 21; 22;
        # 1;1 1;2 1;1; 2;2;
        # 2;1 2;2 2;1; 2;2;
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # all < 0 # 因为对角线上是自己和自己，所以余弦相似度最大


        # tile mask
        logits_mask = torch.scatter(
            torch.ones_like(logits),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ).bool()

        positive_mask = torch.eye(batch_size*anchor_count).bool().to(logits_mask.device)
        positive_mask = torch.roll(positive_mask,batch_size,1)
        domain_label = domain_label.view(batch_size,1,1)
        domain_label = torch.cat(torch.unbind(domain_label.expand(batch_size,anchor_count,1), dim=1), dim=0).T # (2*bs,1) -> (1,2*bs)
        exp_logits = torch.exp(logits)
        new_log_prob = torch.zeros_like(exp_logits).to(exp_logits.device)
        for i in range(self.domain_number):
            domain_mask = (domain_label == i).view(-1)
            ne_po_logits = exp_logits[:,domain_mask].sum(1,keepdim=True)
            tranpose_domain_mask = domain_mask.unsqueeze(-1).expand(anchor_count*batch_size,anchor_count*batch_size)
            tmp_log_prob = torch.where(tranpose_domain_mask,logits - torch.log(ne_po_logits+1e-8),logits - torch.log(ne_po_logits + exp_logits+1e-8))
            w_d = w_matrix[:,i][None,...,None].expand(anchor_count,batch_size,1).reshape(-1,1)
            new_log_prob += (tmp_log_prob * w_d)
        log_prob = - new_log_prob
        log_prob = log_prob[positive_mask]
        loss = log_prob.mean()
        return loss

# loss=PLLLoss(domain_number=3)
# a=torch.randn(9,2,10).cuda()
# b=torch.arange(3).unsqueeze(-1).expand(3,3).reshape(-1).cuda() # 0 0 0 1 1 1 2 2 2
# c=torch.rand(9,3).cuda()

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.m=nn.Parameter(torch.randn_like((a)).cuda())
#     def forward(self,a):
#         a=a*self.m
#         return a
# model=Model()
# optimizer = torch.optim.Adam(model.parameters(),1e-3)
# for i in range(10000):
#     e = model(a)
#     # print(e[:,0,:])
#     lo=loss(e,b,c)
#     optimizer.zero_grad()
#     lo.backward()
#     optimizer.step()
#     print(f"iter is {i}",f"loss is {lo.item()}")
