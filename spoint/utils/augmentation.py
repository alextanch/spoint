import cv2
import random
import numpy as np

import tensorflow as tf


class Augmentation:
    def __init__(self, config, size=(120, 160)):
        self.compose = Compose([
            Resize(size),
            GaussianNoise(**config.gaussian_noise) if 'gaussian_noise' in config else None
        ])

    def __call__(self, image):
        return self.compose(image)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image) if t else image

        return image


class Resize:
    def __init__(self, size=(120, 160)):
        self.size = size

    def __call__(self, image):
        h, w = self.size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        return image


class GaussianNoise:
    def __init__(self, prob=0.5, stddev=(5, 95)):
        self.prob = prob
        self.stddev = stddev

    def __call__(self, image):
        stddev = np.random.uniform(*self.stddev)
        noise = np.random.normal(0, stddev, image.shape)
        image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)

