import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from spoint.utils.primitives import Primitives


class SyntheticDataSet(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.primitives = Primitives(config['primitives'])

    def __len__(self):
        return self.config.batch_size * self.config.batches_in_epoch

    def __getitem__(self, index):
        image, points = self.primitives()

        return torch.from_numpy(image)
