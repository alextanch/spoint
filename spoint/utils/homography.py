import cv2
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.nn.functional as F


class Homography:
    def __init__(self, config):
        self.compose = Compose([
            Patch(**config.patch) if 'patch' in config else None,
            Perspective(**config.perspective) if 'perspective' in config else None,
            Rotation(**config.rotation) if 'rotation' in config else None,
            Scaling(**config.scaling) if 'scaling' in config else None,
            Translation(**config.translation) if 'translation' in config else None])

        self.grid = None

    def __call__(self, image, points):
        h, w = image.shape[:2]

        if self.grid is None:
            y, x = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            self.grid = np.stack((y, x, np.ones((h, w))), axis=2).reshape(-1, 3)

        H = self.compose()

        # warp image
        grid = (self.grid @ np.linalg.inv(H).T)[:, :2]
        grid = torch.from_numpy(grid).view([1, h, w, 2]).float()

        image = torch.tensor(image, dtype=torch.float32).view(1, 1, h, w)

        image = F.grid_sample(image, grid, mode='bilinear', align_corners=True)
        image = image.squeeze().numpy().astype(np.uint8)

        # warp points
        points = points.astype(np.float)

        S = np.array([[0, 2. / w, -1], [2. / h, 0, -1], [0, 0, 1]])
        S = np.linalg.inv(S) @ H @ S

        points = np.column_stack((points, np.ones(len(points))))
        points = (points @ S.T)[:, :2]

        mask = (0 <= points) * (points < image.shape)
        mask = np.prod(mask, axis=1) == 1

        points = points[mask].astype(np.int)

        return image, points


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self):
        points1 = np.stack([[-1, -1], [-1, 1], [1, -1], [1, 1]], axis=0).astype(np.float32)
        points2 = points1

        for t in self.transforms:
            points2 = points2 if (t is None) else t(points2)

        return cv2.getPerspectiveTransform(points1.astype(np.float32), points2.astype(np.float32))


class Patch:
    def __init__(self, ratio=.9):
        self.ratio = ratio

    def __call__(self, points):
        center = np.mean(points, axis=0, keepdims=True)
        points = (points - center) * self.ratio + center

        return points


class Perspective:
    def __init__(self, prob=.5, dx=0.05, dy=0.05, std=2, artifacts=False):
        self.prob = prob
        self.dx = dx
        self.dy = dy
        self.std = std
        self.artifacts = artifacts

    def __call__(self, points):
        if np.random.rand() < self.prob:
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
        if np.random.rand() < self.prob:
            angles = np.linspace(-self.max_angle, self.max_angle, num=self.num_angles)
            angles = np.concatenate((np.array([0]), angles), axis=0)

            center = np.mean(points, axis=0, keepdims=True)

            rot_mat = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)], axis=1)
            rot_mat = np.reshape(rot_mat, [-1, 2, 2])

            rotated = np.matmul((points - center)[np.newaxis, :, :], rot_mat) + center

            if self.artifacts:
                ids = np.arange(self.num_angles)
            else:
                ids = (-1 < rotated) * (rotated < 1)
                ids = np.where(ids.prod(axis=1).prod(axis=1))[0]

            points = rotated[np.random.choice(ids) if ids.size else 0]

        return points


class Scaling:
    def __init__(self, prob=0.5, scale=0.1, num_scales=5, std=2, artifacts=False):
        self.prob = prob
        self.scale = scale
        self.num_scales = num_scales
        self.std = std
        self.artifacts = artifacts

    def __call__(self, points):
        if np.random.rand() < self.prob:
            scales = truncnorm(-self.std, self.std, loc=1, scale=self.scale / 2).rvs(self.num_scales)
            scales = np.concatenate((np.array([1]), scales), axis=0)

            center = np.mean(points, axis=0, keepdims=True)
            scaled = (points - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center

            if self.artifacts:
                ids = np.arange(self.num_scales)
            else:
                ids = (-1 < scaled) * (scaled < 1)
                ids = np.where(ids.prod(axis=1).prod(axis=1))[0]

            points = scaled[np.random.choice(ids) if ids.size else 0]

        return points


class Translation:
    def __init__(self, prob=0.5, overflow=0, artifacts=False):
        self.prob = prob
        self.overflow = overflow
        self.artifacts = artifacts

    def __call__(self, points):
        if np.random.rand() < self.prob:
            dx1, dy1 = 1 + points.min(axis=0)
            dx2, dy2 = 1 - points.max(axis=0)

            dx = np.random.uniform(-dx1, dx2, 1)
            dy = np.random.uniform(-dy1, dy2, 1)

            if self.artifacts:
                dx += self.overflow
                dy += self.overflow

            points += np.array([dx, dy]).T

        return points
