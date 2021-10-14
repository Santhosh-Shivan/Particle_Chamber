import numpy as np
import deeptrack as dt
import scipy

IMAGE_SIZE = 100


"""class MultiParticle(dt.Feature):
    def get(self, image, n_particles, **kwargs):

        position = np.zeros((n_particles, 2), dtype=float)
        radius = np.zeros((n_particles, 1), dtype=float)

        X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        mask = np.zeros(np.shape(X))

        for i in range(n_particles):
            radius[i, :] = 3 + np.random.normal(
                loc=0, scale=0.5, size=1
            )  # (3 + np.random.rand() * 1,)
            position[i, :] = 5 + np.random.rand(2) * 90  # np.random.rand(2) * (100)

            mask = mask + (
                (X - int(position[i, 0])) ** 2 + (Y - int(position[i, 1])) ** 2
                <= radius[i] ** 2
            ) * (
                0.4 + np.random.normal(loc=0, scale=0.02, size=1)
            )  # (0.45 + np.random.rand() * 0.05)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] > 0.45:
                    mask[i, j] = 0.47  # mask[i, j] / np.max(mask)
                if mask[i, j] == 0:
                    mask[i, j] = (
                        0.15 + np.random.rand() * 0.15
                    )  # np.random.normal(loc=0, scale=0.03, size=1)

        mask = -1 * (mask - 1)

        return (np.expand_dims(mask, axis=2), position, radius)
"""


class MultiParticle(dt.Feature):
    def get(self, image, n_particles, **kwargs):

        position = np.zeros((n_particles, 2), dtype=float)
        radius = np.zeros((n_particles, 1), dtype=float)

        min_rad = 3

        X, Y = np.meshgrid(
            np.arange(image.shape[0]), np.arange(image.shape[1]))
        mask = np.zeros(np.shape(X))

        for i in range(n_particles):
            radius[i, :] = min_rad + np.random.normal(
                loc=0, scale=0.5, size=1
            )  # (3 + np.random.rand() * 1,)
            # np.random.rand(2) * (100)
            flag = True
            particle_vec = (5 + np.random.rand(2) * 90)
            if i >= 1:

                while True:
                    particle_vec = (5 + np.random.rand(2) * 90)
                    for j in range(i):
                        sqrd_dist_vec = np.square(
                            particle_vec - position[j, :])
                        dist = np.sqrt(sqrd_dist_vec[0]+sqrd_dist_vec[1])
                        if dist < min_rad + 3:
                            flag = True
                            break
                        flag = False
                    if flag == False:
                        break

            position[i, :] = particle_vec

            mask = mask + (
                (X - int(position[i, 0])) ** 2 + (Y - int(position[i, 1])) ** 2
                <= radius[i] ** 2
            ) * (
                0.4 + np.random.normal(loc=0, scale=0.02, size=1)
            )  # (0.45 + np.random.rand() * 0.05)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] > 0.45:
                    mask[i, j] = 0.47  # mask[i, j] / np.max(mask)
                if mask[i, j] == 0:
                    mask[i, j] = (
                        0.15 + np.random.rand() * 0.15
                    )  # np.random.normal(loc=0, scale=0.03, size=1)

        mask = -1 * (mask - 1)

        return (np.expand_dims(mask, axis=2), position, radius)


"""class Updater(dt.Feature):

    __distributed__ = False

    def __init__(self, feature=None, **kwargs):
        self.feature = self.add_feature(feature)
        super().__init__(**kwargs)

    def get(self, image, features, **kwargs):
        return self.feature.resolve(np.ones((IMAGE_SIZE, IMAGE_SIZE)))"""
