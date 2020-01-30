import cv2
import numpy as np

cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader


def get_random_color(background_color):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """

    color = np.random.randint(256)

    if abs(color - background_color) < 60:  # not enough contrast
        color = (color + 128) % 256

    return color


def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """
    mask = (points[:, 0] >= 0) & (points[:, 0] < size[1]) & (points[:, 1] >= 0) & (points[:, 1] < size[0])
    return points[mask, :]


def get_different_color(previous_colors, min_dist=50, max_count=20):
    """ Output a color that contrasts with the previous colors
    Parameters:
      previous_colors: np.array of the previous colors
      min_dist: the difference between the new color and
                the previous colors must be at least min_dist
      max_count: maximal number of iterations
    """
    color = np.random.randint(256)
    count = 0

    while np.any(np.abs(previous_colors - color) < min_dist) and count < max_count:
        count += 1
        color = np.random.randint(256)

    return color


class Lines:
    """ Draw random lines and output the positions of the endpoints
        Parameters: nb_lines: maximal number of lines
    """

    def __call__(self, image, background_color, nb_lines=10):
        def intersect(a, b, c, d, dim):
            """ Return true if line segments AB and CD intersect """

            def ccw(a, b, c, dim):
                """ Check if the points are listed in counter-clockwise order """
                if dim == 2:
                    v = (c[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0])
                    w = (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                else:
                    v = (c[:, 1, :] - a[:, 1, :]) * (b[:, 0, :] - a[:, 0, :])
                    w = (b[:, 1, :] - a[:, 1, :]) * (c[:, 0, :] - a[:, 0, :])

                return v > w

            return np.any((ccw(a, c, d, dim) != ccw(b, c, d, dim)) & (ccw(a, b, c, dim) != ccw(a, b, d, dim)))

        num_lines = np.random.randint(1, nb_lines)
        segments = np.empty((0, 4), dtype=np.int)
        points = np.empty((0, 2), dtype=np.int)
        min_dim = min(image.shape)

        for i in range(num_lines):
            x1 = np.random.randint(image.shape[1])
            y1 = np.random.randint(image.shape[0])

            p1 = np.array([[x1, y1]])

            x2 = np.random.randint(image.shape[1])
            y2 = np.random.randint(image.shape[0])

            p2 = np.array([[x2, y2]])

            # Check that there is no overlap
            if intersect(segments[:, 0:2], segments[:, 2:4], p1, p2, 2):
                continue

            segments = np.concatenate([segments, np.array([[x1, y1, x2, y2]])], axis=0)

            col = get_random_color(background_color)
            thickness = np.random.randint(min_dim * 0.01, min_dim * 0.02)

            cv2.line(image, (x1, y1), (x2, y2), col, thickness)
            points = np.concatenate([points, np.array([[x1, y1], [x2, y2]])], axis=0)

        return image, points


class Polygon:
    """ Draw a polygon with a random number of corners and return the corner points
        Parameters: max_sides: maximal number of sides + 1
    """

    def __call__(self, image, background_color, max_sides=8):
        def angle_between_vectors(v1, v2):
            """ Compute the angle (in rad) between the two vectors v1 and v2. """
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)

            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        num_corners = np.random.randint(3, max_sides)
        min_dim = min(image.shape[0], image.shape[1])
        rad = max(np.random.rand() * min_dim / 2, min_dim / 10)

        x = np.random.randint(rad, image.shape[1] - rad)  # Center of a circle
        y = np.random.randint(rad, image.shape[0] - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * np.pi, num_corners + 1)
        angles = [slices[i] + np.random.rand() * (slices[i + 1] - slices[i]) for i in range(num_corners)]

        points = np.array([[int(x + max(np.random.rand(), 0.4) * rad * np.cos(a)),
                            int(y + max(np.random.rand(), 0.4) * rad * np.sin(a))] for a in angles])

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(points[(i - 1) % num_corners, :] - points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        points = points[mask, :]
        num_corners = points.shape[0]

        corner_angles = [angle_between_vectors(points[(i - 1) % num_corners, :] - points[i, :],
                                               points[(i + 1) % num_corners, :] - points[i, :])
                         for i in range(num_corners)]

        mask = np.array(corner_angles) < (2 * np.pi / 3)
        points = points[mask, :]
        num_corners = points.shape[0]

        if num_corners < 3:  # not enough corners
            return self(image, max_sides)

        corners = points.reshape((-1, 1, 2))
        col = get_random_color(background_color)

        cv2.fillPoly(image, [corners], col)

        return image, points


class Ellipses:
    """ Draw several ellipses
        Parameters: nb_ellipses: maximal number of ellipses
    """

    def __call__(self, image, background_color, nb_ellipses=20):
        centers = np.empty((0, 2), dtype=np.int)
        rads = np.empty((0, 1), dtype=np.int)
        min_dim = min(image.shape[0], image.shape[1]) / 4

        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))

            max_rad = max(ax, ay)

            x = np.random.randint(max_rad, image.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, image.shape[0] - max_rad)

            new_center = np.array([[x, y]])

            # Check that the ellipsis will not overlap with pre-existing shapes
            diff = centers - new_center

            if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
                continue

            centers = np.concatenate([centers, new_center], axis=0)
            rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

            col = get_random_color(background_color)
            angle = np.random.rand() * 90

            cv2.ellipse(image, (x, y), (ax, ay), angle, 0, 360, col, -1)

        return image, np.empty((0, 2), dtype=np.int)


class Star:
    """ Draw a star and output the interest points
        Parameters: nb_branches: number of branches of the star
    """

    def __call__(self, image, background_color, nb_branches=6):
        num_branches = np.random.randint(3, nb_branches)
        min_dim = min(image.shape[0], image.shape[1])

        thickness = np.random.randint(min_dim * 0.01, min_dim * 0.02)
        rad = max(np.random.rand() * min_dim / 2, min_dim / 5)

        x = np.random.randint(rad, image.shape[1] - rad)  # select the center of a circle
        y = np.random.randint(rad, image.shape[0] - rad)

        # Sample num_branches points inside the circle
        slices = np.linspace(0, 2 * np.pi, num_branches + 1)
        angles = [slices[i] + np.random.rand() * (slices[i + 1] - slices[i]) for i in range(num_branches)]

        points = np.array([[int(x + max(np.random.rand(), 0.3) * rad * np.cos(a)),
                            int(y + max(np.random.rand(), 0.3) * rad * np.sin(a))] for a in angles])

        points = np.concatenate(([[x, y]], points), axis=0)

        x0, y0 = points[0][0], points[0][1]

        for i in range(1, num_branches + 1):
            x1, y1 = points[i][0], points[i][1]
            col = get_random_color(background_color)
            cv2.line(image, (x0, y0), (x1, y1), col, thickness)

        return image, points


class Stripes:
    """ Draw stripes in a distorted rectangle and output the interest points
        Parameters:
            max_nb_cols: maximal number of stripes to be drawn
            min_width_ratio: the minimal width of a stripe is min_width_ratio * smallest dimension of the image
            transform_params: set the range of the parameters of the transformations
    """

    def __call__(self, image, background_color, max_nb_cols=13, min_width_ratio=0.04, transform_params=(0.1, 0.1)):
        # Create the grid
        board_size = (int(image.shape[0] * (1 + np.random.rand())), int(image.shape[1] * (1 + np.random.rand())))
        col = np.random.randint(5, max_nb_cols)  # number of cols
        cols = np.concatenate([board_size[1] * np.random.rand(col - 1), np.array([0, board_size[1] - 1])], axis=0)
        cols = np.unique(cols.astype(int))

        # Remove the indices that are too close
        min_dim = min(image.shape)
        min_width = min_dim * min_width_ratio

        cols = cols[(np.concatenate([cols[1:], np.array([board_size[1] + min_width])], axis=0) - cols) >= min_width]
        col = cols.shape[0] - 1  # update the number of cols
        cols = np.reshape(cols, (col + 1, 1))
        cols1 = np.concatenate([cols, np.zeros((col + 1, 1), np.int32)], axis=1)
        cols2 = np.concatenate([cols, (board_size[0] - 1) * np.ones((col + 1, 1), np.int32)], axis=1)

        points = np.concatenate([cols1, cols2], axis=0)

        # Warp the grid using an affine transformation and an homography
        # The parameters of the transformations are constrained
        # to get transformations not too far-fetched
        # Prepare the matrices
        alpha_affine = np.max(image.shape) * (transform_params[0] + np.random.rand() * transform_params[1])
        center_square = np.float32(image.shape) // 2
        square_size = min(image.shape) // 3

        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size, [center_square[0] - square_size, center_square[1] + square_size]])

        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_transform = cv2.getAffineTransform(pts1[:3], pts2[:3])

        pts2 = pts1 + np.random.uniform(-alpha_affine / 2, alpha_affine / 2, size=pts1.shape).astype(np.float32)
        transform = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the affine transformation
        points = np.transpose(np.concatenate((points, np.ones((2 * (col + 1), 1))), axis=1))
        warp_points = np.transpose(np.dot(affine_transform, points))

        # Apply the homography
        warped_col0 = np.add(np.sum(np.multiply(warp_points, transform[0, :2]), axis=1), transform[0, 2])
        warped_col1 = np.add(np.sum(np.multiply(warp_points, transform[1, :2]), axis=1), transform[1, 2])
        warped_col2 = np.add(np.sum(np.multiply(warp_points, transform[2, :2]), axis=1), transform[2, 2])

        warped_col0 = np.divide(warped_col0, warped_col2)
        warped_col1 = np.divide(warped_col1, warped_col2)

        warp_points = np.concatenate([warped_col0[:, None], warped_col1[:, None]], axis=1)
        warp_points = warp_points.astype(int)

        # Fill the rectangles
        color = get_random_color(background_color)

        for i in range(col):
            color = (color + 128 + np.random.randint(-30, 30)) % 256
            cv2.fillConvexPoly(image, np.array([(warp_points[i, 0], warp_points[i, 1]),
                                                (warp_points[i + 1, 0], warp_points[i + 1, 1]),
                                                (warp_points[i + col + 2, 0], warp_points[i + col + 2, 1]),
                                                (warp_points[i + col + 1, 0], warp_points[i + col + 1, 1])]), color)

        # Draw lines on the boundaries of the stripes at random
        nb_rows = np.random.randint(2, 5)
        nb_cols = np.random.randint(2, col + 2)

        thickness = np.random.randint(min_dim * 0.01, min_dim * 0.015)

        for _ in range(nb_rows):
            row_idx = np.random.choice([0, col + 1])
            col_idx1 = np.random.randint(col + 1)
            col_idx2 = np.random.randint(col + 1)
            color = get_random_color(background_color)

            cv2.line(image,
                     (warp_points[row_idx + col_idx1, 0], warp_points[row_idx + col_idx1, 1]),
                     (warp_points[row_idx + col_idx2, 0], warp_points[row_idx + col_idx2, 1]),
                     color, thickness)

        for _ in range(nb_cols):
            col_idx = np.random.randint(col + 1)
            color = get_random_color(background_color)

            cv2.line(image,
                     (warp_points[col_idx, 0], warp_points[col_idx, 1]),
                     (warp_points[col_idx + col + 1, 0], warp_points[col_idx + col + 1, 1]),
                     color, thickness)

        # Keep only the points inside the image
        points = keep_points_inside(warp_points, image.shape[:2])

        return image, points


class Cube:
    """ Draw a 2D projection of a cube and output the corners that are visible
            Parameters:
              min_size_ratio: min(img.shape) * min_size_ratio is the smallest achievable
                              cube side size
              min_angle_rot: minimal angle of rotation
              scale_interval: the scale is between scale_interval[0] and
                              scale_interval[0]+scale_interval[1]
              trans_interval: the translation is between img.shape*trans_interval[0] and
                              img.shape*(trans_interval[0] + trans_interval[1])
    """

    def __call__(self, image, background_color, min_size_ratio=0.2,
                 min_angle_rot=np.pi / 10, scale_interval=(0.4, 0.6),
                 trans_interval=(0.5, 0.2)):
        # Generate a cube and apply to it an affine transformation
        # The order matters!
        # The indices of two adjacent vertices differ only of one bit (as in Gray codes)

        min_dim = min(image.shape[:2])
        min_side = min_dim * min_size_ratio

        lx = min_side + np.random.rand() * 2 * min_dim / 3  # dimensions of the cube
        ly = min_side + np.random.rand() * 2 * min_dim / 3
        lz = min_side + np.random.rand() * 2 * min_dim / 3

        cube = np.array([[0, 0, 0],
                         [lx, 0, 0],
                         [0, ly, 0],
                         [lx, ly, 0],
                         [0, 0, lz],
                         [lx, 0, lz],
                         [0, ly, lz],
                         [lx, ly, lz]])

        rot_angles = np.random.rand(3) * 3 * np.pi / 10. + np.pi / 10.

        rotation_1 = np.array([[np.cos(rot_angles[0]), -np.sin(rot_angles[0]), 0],
                               [np.sin(rot_angles[0]), np.cos(rot_angles[0]), 0],
                               [0, 0, 1]])

        rotation_2 = np.array([[1, 0, 0],
                               [0, np.cos(rot_angles[1]), -np.sin(rot_angles[1])],
                               [0, np.sin(rot_angles[1]), np.cos(rot_angles[1])]])

        rotation_3 = np.array([[np.cos(rot_angles[2]), 0, -np.sin(rot_angles[2])],
                               [0, 1, 0],
                               [np.sin(rot_angles[2]), 0, np.cos(rot_angles[2])]])

        scaling = np.array([[scale_interval[0] + np.random.rand() * scale_interval[1], 0, 0],
                            [0, scale_interval[0] + np.random.rand() * scale_interval[1], 0],
                            [0, 0, scale_interval[0] + np.random.rand() * scale_interval[1]]])

        a = np.random.randint(-image.shape[1] * trans_interval[1], image.shape[1] * trans_interval[1])
        b = np.random.randint(-image.shape[0] * trans_interval[1], image.shape[0] * trans_interval[1])

        trans = np.array([image.shape[1] * trans_interval[0] + a, image.shape[0] * trans_interval[0] + b, 0])

        cube = trans + np.transpose(
            np.dot(scaling, np.dot(rotation_1, np.dot(rotation_2, np.dot(rotation_3, np.transpose(cube))))))

        # The hidden corner is 0 by construction
        # The front one is 7
        cube = cube[:, :2]  # project on the plane z=0
        cube = cube.astype(int)

        points = cube[1:, :]  # get rid of the hidden corner

        # Get the three visible faces
        faces = np.array([[7, 3, 1, 5], [7, 5, 4, 6], [7, 6, 2, 3]])

        # Fill the faces and draw the contours
        col_face = get_random_color(background_color)

        for i in [0, 1, 2]:
            cv2.fillPoly(image, [cube[faces[i]].reshape((-1, 1, 2))], col_face)

        thickness = np.random.randint(min_dim * 0.003, min_dim * 0.015)

        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3]:
                # color that constrats with the face color
                col_edge = (col_face + 128 + np.random.randint(-64, 64)) % 256
                cv2.line(image,
                         (cube[faces[i][j], 0], cube[faces[i][j], 1]),
                         (cube[faces[i][(j + 1) % 4], 0], cube[faces[i][(j + 1) % 4], 1]), col_edge, thickness)

        # Keep only the points inside the image
        points = keep_points_inside(points, image.shape[:2])

        return image, points


class Checkerboard:
    """ Draw a checkerboard and output the interest points
        Parameters:
              max_rows: maximal number of rows + 1
              max_cols: maximal number of cols + 1
              transform_params: set the range of the parameters of the transformations
    """

    def __call__(self, image, background_color, max_rows=7, max_cols=7, transform_params=(0.05, 0.15)):
        # Create the grid
        rows = np.random.randint(3, max_rows)  # number of rows
        cols = np.random.randint(3, max_cols)  # number of cols

        s = min((image.shape[1] - 1) // cols, (image.shape[0] - 1) // rows)  # size of a cell

        x_coord = np.tile(range(cols + 1), rows + 1).reshape(((rows + 1) * (cols + 1), 1))
        y_coord = np.repeat(range(rows + 1), cols + 1).reshape(((rows + 1) * (cols + 1), 1))

        points = s * np.concatenate([x_coord, y_coord], axis=1)

        # Warp the grid using an affine transformation and an homography
        # The parameters of the transformations are constrained
        # to get transformations not too far-fetched
        alpha_affine = np.max(image.shape) * (transform_params[0] + np.random.rand() * transform_params[1])
        center_square = np.float32(image.shape) // 2
        min_dim = min(image.shape)
        square_size = min_dim // 3

        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size,
                           [center_square[0] - square_size, center_square[1] + square_size]])

        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_transform = cv2.getAffineTransform(pts1[:3], pts2[:3])

        pts2 = pts1 + np.random.uniform(-alpha_affine / 2, alpha_affine / 2, size=pts1.shape).astype(np.float32)
        transform = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the affine transformation
        points = np.transpose(np.concatenate((points, np.ones(((rows + 1) * (cols + 1), 1))), axis=1))
        warp_points = np.transpose(np.dot(affine_transform, points))

        # Apply the homography
        warp_col0 = np.add(np.sum(np.multiply(warp_points, transform[0, :2]), axis=1), transform[0, 2])
        warp_col1 = np.add(np.sum(np.multiply(warp_points, transform[1, :2]), axis=1), transform[1, 2])
        warp_col2 = np.add(np.sum(np.multiply(warp_points, transform[2, :2]), axis=1), transform[2, 2])

        warp_col0 = np.divide(warp_col0, warp_col2)
        warp_col1 = np.divide(warp_col1, warp_col2)

        warp_points = np.concatenate([warp_col0[:, None], warp_col1[:, None]], axis=1)
        warp_points = warp_points.astype(int)

        # Fill the rectangles
        colors = np.zeros((rows * cols,), np.int32)

        for i in range(rows):
            for j in range(cols):
                # Get a color that contrast with the neighboring cells
                if i == 0 and j == 0:
                    col = get_random_color(background_color)
                else:
                    neighboring_colors = []

                    if i != 0:
                        neighboring_colors.append(colors[(i - 1) * cols + j])
                    if j != 0:
                        neighboring_colors.append(colors[i * cols + j - 1])

                    col = get_different_color(np.array(neighboring_colors))

                colors[i * cols + j] = col

                # Fill the cell
                cv2.fillConvexPoly(image, np.array(
                    [(warp_points[i * (cols + 1) + j, 0], warp_points[i * (cols + 1) + j, 1]),
                     (warp_points[i * (cols + 1) + j + 1, 0], warp_points[i * (cols + 1) + j + 1, 1]),
                     (warp_points[(i + 1) * (cols + 1) + j + 1, 0], warp_points[(i + 1) * (cols + 1) + j + 1, 1]),
                     (warp_points[(i + 1) * (cols + 1) + j, 0], warp_points[(i + 1) * (cols + 1) + j, 1])]), col)

        # Draw lines on the boundaries of the board at random
        nb_rows = np.random.randint(2, rows + 2)
        nb_cols = np.random.randint(2, cols + 2)

        thickness = np.random.randint(min_dim * 0.01, min_dim * 0.015)

        for _ in range(nb_rows):
            row_idx = np.random.randint(rows + 1)
            col_idx1 = np.random.randint(cols + 1)
            col_idx2 = np.random.randint(cols + 1)
            col = get_random_color(background_color)

            cv2.line(image,
                     (warp_points[row_idx * (cols + 1) + col_idx1, 0],
                      warp_points[row_idx * (cols + 1) + col_idx1, 1]),
                     (warp_points[row_idx * (cols + 1) + col_idx2, 0],
                      warp_points[row_idx * (cols + 1) + col_idx2, 1]),
                     col, thickness)

        for _ in range(nb_cols):
            col_idx = np.random.randint(cols + 1)
            row_idx1 = np.random.randint(rows + 1)
            row_idx2 = np.random.randint(rows + 1)
            col = get_random_color(background_color)

            cv2.line(image,
                     (warp_points[row_idx1 * (cols + 1) + col_idx, 0],
                      warp_points[row_idx1 * (cols + 1) + col_idx, 1]),
                     (warp_points[row_idx2 * (cols + 1) + col_idx, 0],
                      warp_points[row_idx2 * (cols + 1) + col_idx, 1]),
                     col, thickness)

        # Keep only the points inside the image
        points = keep_points_inside(warp_points, image.shape[:2])

        return image, points


class Background:
    """ Render a customized background image
            Parameters:
              size: size of the image
              nb_blobs: number of circles to draw
              min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
              max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
              min_kernel_size: minimal size of the kernel
              max_kernel_size: maximal size of the kernel
            """
    def __call__(self, size=(960, 1280), nb_blobs=100,
                 min_rad_ratio=0.02, max_rad_ratio=0.031,
                 min_kernel_size=150, max_kernel_size=500):

        image = np.zeros(size, dtype=np.uint8)

        cv2.randu(image, 0, 255)
        cv2.threshold(image, np.random.randint(256), 255, 0, image)

        background_color = int(np.mean(image))

        r1 = np.random.randint(0, size[1], size=(nb_blobs, 1))
        r2 = np.random.randint(0, size[0], size=(nb_blobs, 1))

        blobs = np.concatenate([r1, r2], axis=1)
        dim = max(size)

        for i in range(nb_blobs):
            center = blobs[i][0], blobs[i][1]
            radius = np.random.randint(int(dim * min_rad_ratio), int(dim * max_rad_ratio))
            color = get_random_color(background_color)

            cv2.circle(image, center, radius, color, -1)

        kernel_size = np.random.randint(min_kernel_size, max_kernel_size)
        cv2.blur(image, (kernel_size, kernel_size), image)

        return image


class Primitives:
    PRIMITIVES = {
        'lines': Lines(),
        'polygon': Polygon(),
        'ellipses': Ellipses(),
        'star': Star(),
        'stripes': Stripes(),
        'cube': Cube(),
        'checkerboard': Checkerboard()
    }

    def __init__(self, config):
        self.config = config

    def __call__(self):
        try:
            if 'background' in self.config:
                image = Background()(**self.config.background)
            else:
                image = Background()()

            background_color = int(np.mean(image))

            name = np.random.choice(self.config.primitives)

            if name in self.PRIMITIVES:
                image, points = self.PRIMITIVES[name](image, background_color)
            else:
                raise NotImplementedError(f'Not implemented primitive: {name}')

        except Exception as e:
            raise Exception(f'Primitives call failed: {repr(e)}')

        return image, points
