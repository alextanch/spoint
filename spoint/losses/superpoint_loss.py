import torch.nn as nn


class SuperPointLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.bce_loss = nn.BCELoss(reduction='none')

    def detector_loss(self, input, target, mask):
        soft_max = nn.functional.softmax(input, dim=1)
        loss = self.bce_loss(soft_max, target)

        loss = loss.sum(dim=1)

        if mask is None:
            loss = loss.mean()
        else:
            loss = (loss * mask).sum()
            loss = loss / (mask.sum() + 1e-10)

        return loss

    def forward(self, input, target, mask=None):
        loss = self.detector_loss(input, target, mask)

        return loss
