import cv2
import numpy as np


class Augmentation:
    def __init__(self, config, size=(120, 160)):
        self.compose = Compose([
            Resize(size),
            GaussianNoise(**config.gaussian_noise) if 'gaussian_noise' in config else None,
            SpeckleNoise(**config.speckle_noise) if 'speckle_noise' in config else None,
            RandomBrightness(**config.random_brightness) if 'random_brightness' in config else None,
            RandomContrast(**config.random_contrast) if 'random_contrast' in config else None,
            AdditiveShade(**config.additive_shade) if 'additive_shade' in config else None,
            MotionBlur(**config.motion_blur) if 'motion_blur' in config else None
        ])

    def __call__(self, image):
        return self.compose(image)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = image if (t is None) else t(image)

        return image.astype(np.uint8)


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
        if np.random.rand() < self.prob:
            stddev = np.random.uniform(*self.stddev)
            noise = np.random.normal(0, stddev, image.shape)
            image = np.clip(image + noise, 0, 255)

        return image


class SpeckleNoise:
    def __init__(self, prob=0.5, range=(.0, .005)):
        self.prob = prob
        self.range = range

    def __call__(self, image):
        noisy_image = image

        if np.random.rand() < self.prob:
            p = np.random.uniform(*self.range)
            sample = np.random.uniform(size=image.shape)

            noisy_image = np.where(sample <= p, np.zeros_like(image), image)
            noisy_image = np.where(sample >= (1. - p), 255. * np.ones_like(image), noisy_image)

        return noisy_image


class RandomBrightness:
    def __init__(self, prob=0.5, max_change=50):
        self.prob = prob
        self.max_change = max_change

    def __call__(self, image):
        if np.random.rand() < self.prob:
            factor = np.random.uniform(-self.max_change, self.max_change)
            image = np.clip(image + factor, 0, 255)

        return image


class RandomContrast:
    def __init__(self, prob=0.5, range=(0.5, 1.5)):
        self.prob = prob
        self.range = range

    def __call__(self, image):
        if np.random.rand() < self.prob:
            factor = np.random.uniform(*self.range)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255)

        return image


class AdditiveShade:
    def __init__(self, prob=0.5, ellipses=20, transparency=(-0.5, 0.8), ksize=(250, 350)):
        self.prob = prob
        self.ellipses = ellipses
        self.transparency = transparency
        self.ksize = ksize

    def __call__(self, image):
        if np.random.rand() < self.prob:
            size = image.shape[:2]

            min_dim = min(size) / 4
            mask = np.zeros(size, np.float32)

            for i in range(self.ellipses):
                ax = int(max(np.random.rand() * min_dim, min_dim / 5))
                ay = int(max(np.random.rand() * min_dim, min_dim / 5))

                max_rad = max(ax, ay)

                x = np.random.randint(max_rad, size[1] - max_rad)  # center
                y = np.random.randint(max_rad, size[0] - max_rad)

                angle = np.random.rand() * 90

                cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 1, -1)

                transparency = np.random.uniform(*self.transparency)
                ksize = np.random.randint(*self.ksize)

                if (ksize % 2) == 0:
                    ksize += 1

                mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
                image = np.clip(image * (1 - transparency * mask), 0, 255)

        return image


class MotionBlur:
    def __init__(self, prob=0.5, ksize=10):
        self.prob = prob
        self.ksize = ksize

    def __call__(self, image):
        if np.random.rand() < self.prob:
            # Either vertial, hozirontal or diagonal blur
            mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])

            ksize = 1 + 2 * np.random.randint(0, (self.ksize + 1) / 2)  # make sure is odd
            center = int((ksize - 1) / 2)
            kernel = np.zeros((ksize, ksize))

            if mode == 'h':
                kernel[center, :] = 1.
            elif mode == 'v':
                kernel[:, center] = 1.
            elif mode == 'diag_down':
                kernel = np.eye(ksize)
            elif mode == 'diag_up':
                kernel = np.flip(np.eye(ksize), 0)

            var = ksize * ksize / 16.
            grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
            gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))

            kernel *= gaussian
            kernel /= np.sum(kernel)

            image = np.clip(cv2.filter2D(image, -1, kernel), 0, 255)

        return image


# def random_brightness(image, max_abs_change=50):
#     import tensorflow as tf
#     return tf.clip_by_value(tf.image.random_brightness(image, max_abs_change), 0, 255)
#     sess = tf.Session()
#     with sess.as_default():
#         _img = img.val()

