import numpy as np
import deeptrack as dt

IMAGE_SIZE = 100


class MultiParticle(dt.Feature):
    def get(self, image, n_particles, **kwargs):

        position = np.zeros((n_particles, 2), dtype=float)
        radius = np.zeros((n_particles, 1), dtype=float)

        X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        mask = np.zeros(np.shape(X))

        for i in range(n_particles):
            radius[i, :] = (2 + np.random.rand() * 1,)
            position[i, :] = np.random.rand(2) * (100)

            mask = mask + (
                (X - int(position[i, 0])) ** 2 + (Y - int(position[i, 1])) ** 2
                <= radius[i] ** 2
            ) * (0.4 + np.random.rand() * 0.2)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] > 0.6:
                    mask[i, j] = mask[i, j] / np.max(mask)
                if mask[i, j] == 0:
                    mask[i, j] = 0.2 + np.random.rand() * 0.05

        mask = -1 * (mask - 1)

        return (np.expand_dims(mask, axis=0), position, radius)  #


class Updater(dt.Feature):

    __distributed__ = False

    def __init__(self, feature=None, **kwargs):
        self.feature = self.add_feature(feature)
        super().__init__(**kwargs)

    def get(self, image, features, **kwargs):
        return self.feature.resolve(np.ones((IMAGE_SIZE, IMAGE_SIZE)))
