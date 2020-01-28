import torch.nn as nn


class SuperPointLoss:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def detector_loss(self, input, target, mask):
        bce_loss = nn.BCELoss(reduction='none').to(self.device)
        loss = bce_loss(nn.functional.softmax(input, dim=1), target)

        loss = loss.sum(dim=1)

        if mask:
            loss = (loss * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        else:
            loss = loss.mean()

        return loss

    def __call__(self, input, target, mask=None):
        loss = self.detector_loss(input, target, mask)

        return loss
