import cv2
import random
import numpy as np


class Augmentation:
    def __init__(self, config, size=(120, 160)):
        self.compose = Compose([
            Resize(size),
            GaussianNoise(**config.gaussian_noise) if 'gaussian_noise' in config else None,
            SpeckleNoise(**config.speckle_noise) if 'speckle_noise' in config else None,
            RandomBrightness(**config.random_brightness) if 'random_brightness' in config else None
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
        if random.random() < self.prob:
            stddev = np.random.uniform(*self.stddev)
            noise = np.random.normal(0, stddev, image.shape)
            image = np.clip(image + noise, 0, 255)

        return image.astype(np.uint8)


class SpeckleNoise:
    def __init__(self, prob=0.5, stddev=(.0, .005)):
        self.prob = prob
        self.stddev = stddev

    def __call__(self, image):
        noisy_image = image

        if random.random() < self.prob:
            p = np.random.uniform(*self.stddev)
            sample = np.random.uniform(size=image.shape)

            noisy_image = np.where(sample <= p, np.zeros_like(image), image)
            noisy_image = np.where(sample >= (1. - p), 255. * np.ones_like(image), noisy_image)

        return noisy_image.astype(np.uint8)


class RandomBrightness:
    def __init__(self, prob=0.5, max_change=50):
        self.prob = prob
        self.max_change = max_change

    def __call__(self, image):
        if random.random() < self.prob:
            value = np.random.uniform(-self.max_change, self.max_change)
            image = np.clip(image + value, 0, 255)

        return image.astype(np.uint8)


# def random_brightness(image, max_abs_change=50):
#     import tensorflow as tf
#     return tf.clip_by_value(tf.image.random_brightness(image, max_abs_change), 0, 255)
#     sess = tf.Session()
#     with sess.as_default():
#         _img = img.val()
