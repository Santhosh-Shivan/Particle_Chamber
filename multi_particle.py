import deeptrack as dt

from deeptrack.models.yolo import utils
import numpy as np
import matplotlib.pyplot as plt

from simulation import *

sim = MultiParticle(
    n_particles=lambda: int(np.random.rand() * 10) + 15,
)
# print('a', np.array(sim[1]) )
particle = Updater(sim[0])

pos = Updater(sim[1])

rad = Updater(sim[2])


gradient = dt.IlluminationGradient(
    gradient=[0.5e-3, 0e-3],  # lambda: 1e-3 + np.random.randn(2) * 1e-3,
)


class Normalize(dt.Feature):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, image, **kwargs):
        image = image / np.max(image)
        return image


normalization = Normalize()

gauss = dt.Gaussian(mu=0, sigma=0.01)

import scipy

# Smoothing
kernel = np.ones((3, 3)) / 15
smoothing = dt.Lambda(
    lambda: lambda image: scipy.ndimage.filters.convolve(image, kernel)
)

particle = particle >> smoothing >> gauss >> normalization


def get_label(a, b):
    a = a.resolve()
    b = b.resolve()
    n = np.shape(a)[0]
    label = np.zeros((n, 4))

    for i in range(n):
        label[i, :] = [int(a[i, 0]), int(a[i, 1]), b[i], b[i]]

    return label  # (np.asarray(a.resolve()), np.asarray(b.resolve()))


NUMBER_OF_IMAGES = 20

for _ in range(NUMBER_OF_IMAGES):
    particle.update()
    a = pos.update()
    b = rad.update()
    image_of_particle = particle.resolve()

    label = get_label(a, b)
    n = np.shape(label)[0]
    position_of_particles = label[:, 0:2]
    radius_of_particles = label[:, 2]

    plt.imshow(image_of_particle, cmap="gray", vmin=0)
    plt.colorbar()
    ax = plt.gca()
    for i in range(n):
        rect = plt.Rectangle(
            (label[i, 0] - label[i, 2], label[i, 1] - label[i, 2]),
            2 * label[i, 2],
            2 * label[i, 2],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    # plt.scatter(position_of_particles[:,0], position_of_particles[:,1], marker='+', edgecolors="r", linewidth=2)
    plt.show()


"""generator = dt.generators.ContinuousGenerator(
    particle,
    get_label,
    min_data_size=int(2e+4),
    max_data_size=int(3e+4),
    batch_size=64
    
)

from deeptrack.models.yolo.yolo import YOLOv3

model = YOLOv3(
    (256, 256, 1),
    2,
)

model.compile(optimizer="adam")

with generator:
    model.fit(generator, epochs=50)"""
