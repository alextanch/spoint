import torch
import torch.utils.data as data

from spoint.utils.primitives import Primitives
from spoint.utils.homography import Homography
from spoint.utils.augmentation import Augmentation


class SyntheticDataSet(data.Dataset):
    def __init__(self, config):
        self.config = config

        self.primitives = Primitives(config['primitives'])
        self.augmentation = Augmentation(config['augmentation'], size=config.size)
        self.homography = Homography(config['homography'], size=config.size)

    def __len__(self):
        return self.config.batch_size * self.config.batches_in_epoch

    def __getitem__(self, index):
        image, points = self.primitives()
        image = self.augmentation(image)
        wrap_image, wrap_points = self.homography(image, points)

        return torch.from_numpy(image)
