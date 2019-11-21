import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats2


def label_smoothing_criterion(distribution='uniform', alpha=0.1, std=0.5, reduction='mean'):
    '''
    distribution can be ''uniform' or 'non-uniform'
    '''

    def _cross_entropy_loss_one_hot(logits, target, reduction='mean'):
        logp = F.log_softmax(logits, dim=1)
        loss = torch.sum(-logp * target, dim=1)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(
                '`reduction` must be one of \'none\', \'mean\', or \'sum\'.')

    def _get_gaussian_label_distribution(n_classes, std=0.5):
        CLs = []
        for l in range(n_classes):
            CLs.append(stats2.norm.pdf(np.arange(n_classes), l, std))
        dists = np.stack(CLs, axis=0)
        return dists

    def _one_hot_encoding_torch(l, n_classes):
        return torch.zeros(l.size(0), n_classes).to(l.device).scatter_(1, l.view(-1, 1), 1)

    def _label_smoothing_criterion(logits, labels):
        n_classes = logits.size(1)
        device = logits.device
        if distribution == 'uniform':
            one_hot = _one_hot_encoding_torch(labels, n_classes)
            uniform = torch.ones_like(one_hot) / n_classes
            soft_labels = (1 - alpha) * one_hot + alpha * uniform
        elif distribution == 'non-uniform':
            dist = _get_gaussian_label_distribution(n_classes, std=std)
            if device is not 'cpu':
                soft_labels = torch.from_numpy(dist[labels.cpu().numpy()]).to(device)
            else:
                soft_labels = torch.from_numpy(dist[labels.numpy()])
        else:
            print('Not implemented')

        loss = _cross_entropy_loss_one_hot(logits, soft_labels.float(), reduction)
        return loss

    return _label_smoothing_criterion
