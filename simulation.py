import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt

from particle import *

particle = MultiParticle(
    
    radius=lambda: 1 + np.random.rand()*1,
    n_particles = lambda: int(np.random.rand()*20) + 5
    
    
)

particle = Updater(particle)


gradient = dt.IlluminationGradient(
    gradient=[.5e-3, 0e-3],#lambda: 1e-3 + np.random.randn(2) * 1e-3,
)

class Normalize(dt.Feature):
    def __init__(
        self, **kwargs
    ):
        super().__init__( **kwargs)

    def get(self, image, **kwargs):
        image = image / np.max(image)
        return image

normalization = Normalize()

gauss = dt.Gaussian(mu = 0, sigma = .02)

import scipy
#Smoothing
kernel = np.ones((5, 5)) / 30
smoothing = dt.Lambda(lambda: lambda image: scipy.ndimage.filters.convolve(image, kernel)) 

particle +=  smoothing + gradient +gauss + normalization 

NUMBER_OF_IMAGES = 20

for _ in range(NUMBER_OF_IMAGES):
    particle.update()
    image_of_particle = particle.resolve()
    """label = get_label(image_of_particle)
    position_of_particle = np.array([label[0], label[1]])
    position_of_particle = position_of_particle * IMAGE_SIZE """
    
    plt.imshow(image_of_particle, cmap="gray", vmin = 0)
    plt.colorbar()
    #plt.scatter(position_of_particle[0], position_of_particle[1], marker='+', edgecolors="r", linewidth=2)
    plt.show()