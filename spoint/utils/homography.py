import random

import cv2
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.nn.functional as F


class Homography:
    def __init__(self, config, size=(960, 1280)):
        self.size = size

        self.compose = Compose([
            Patch(**config.patch) if 'patch' in config else None,
            Perspective(**config.perspective) if 'perspective' in config else None,
            Rotation(**config.rotation) if 'rotation' in config else None
        ], size)

        h, w = size
        y, x = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

        self.grid = np.stack((y, x, np.ones((h, w))), axis=2).reshape(-1, 3)

    def __call__(self, image, points):
        H = self.compose()

        grid = (self.grid @ np.linalg.inv(H).T)[:, :2]

        h, w = self.size

        grid = torch.from_numpy(grid).view([1, h, w, 2]).float()
        image = torch.tensor(image, dtype=torch.float32).view(1, 1, h, w)

        image = F.grid_sample(image, grid, mode='bilinear', align_corners=True)
        image = image.squeeze().numpy().astype(np.uint8)

        return image, points


class Compose:
    def __init__(self, transforms, size):
        self.transforms = transforms
        self.size = size

    def __call__(self):
        points1 = np.stack([[-1, -1], [-1, 1], [1, -1], [1, 1]], axis=0).astype(np.float32)
        points2 = points1

        for t in self.transforms:
            points2 = points2 if (t is None) else t(points2)

        return cv2.getPerspectiveTransform(points1, points2)


class Patch:
    def __init__(self, ratio=.9):
        self.ratio = ratio

    def __call__(self, points):
        center = np.mean(points, axis=0, keepdims=True)
        points = (points - center) * self.ratio + center

        return points.astype(np.float32)


class Perspective:
    def __init__(self, prob=.5, dx=0.05, dy=0.05, std=2, artifacts=False):
        self.prob = prob
        self.dx = dx
        self.dy = dy
        self.std = std
        self.artifacts = artifacts

    def __call__(self, points):
        if random.random() < self.prob:
            dx = self.dx
            dy = self.dy

            if not self.artifacts:
                dx1, dy1 = (1 + points.min(axis=0)) / 2
                dx2, dy2 = (1 - points.max(axis=0)) / 2

                dx = min(min(dx1, self.dx), dx2)
                dy = min(min(dy1, self.dy), dy2)

            dx = truncnorm(-self.std, self.std, loc=0, scale=dx / 2).rvs(1)
            dy = truncnorm(-self.std, self.std, loc=0, scale=dy / 2).rvs(1)

            points += np.array([[dx, dy], [dx, -dy], [-dx, dy], [-dx, -dy]]).squeeze()

        return points


class Rotation:
    def __init__(self, prob=0.5, max_angle=1.57, num_angles=10, artifacts=False):
        self.max_angle = max_angle
        self.num_angles = num_angles
        self.artifacts = artifacts
        self.prob = prob

    def __call__(self, points):
        if random.random() < self.prob:
            angles = np.linspace(-self.max_angle, self.max_angle, num=self.num_angles)
            angles = np.concatenate((angles, np.array([0.])), axis=0)

            center = np.mean(points, axis=0, keepdims=True)

            rot_mat = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1)
            rot_mat = np.reshape(rot_mat, [-1, 2, 2])

            rotated = np.matmul((points - center)[np.newaxis, :, :], rot_mat) + center

            if self.artifacts:
                valid = np.arange(self.num_angles)
            else:
                valid = (rotated > -1) * (rotated < 1.)
                valid = valid.prod(axis=1).prod(axis=1)
                valid = np.where(valid)[0]

            idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
            points = rotated[idx, :, :]

        return points.astype(np.float32)
