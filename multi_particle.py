import deeptrack as dt
import numpy as np
import matplotlib.pyplot as plt

from particle import *

def pos(n_particles):
    position = np.zeros((n_particles,2))
    return position

sim= MultiParticle(
    
    
    n_particles = lambda: int(np.random.rand()*10) + 15,
        
    
)
#print('a', np.array(sim[1]) )
particle = Updater(sim[0])

pos = Updater(sim[1])

rad = Updater(sim[2])


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
kernel = np.ones((5, 5)) / 15
smoothing = dt.Lambda(lambda: lambda image: scipy.ndimage.filters.convolve(image, kernel)) 

particle +=  smoothing + gradient +gauss + normalization 


def get_label(a, b):
    a = a.resolve()
    b = b.resolve()
    n = np.shape(a)[0]
    label = np.zeros((n,4))

    for i in range(n):
        label[i,:] = [int(a[i, 0]), int(a[i, 1]), b[i], b[i]]

    
    return label #(np.asarray(a.resolve()), np.asarray(b.resolve()))


NUMBER_OF_IMAGES = 20

for _ in range(NUMBER_OF_IMAGES):
    particle.update()
    a = pos.update()
    b = rad.update()
    image_of_particle = particle.resolve()
    

    label = get_label(a, b)
    n = np.shape(label)[0]
    position_of_particles = label[:,0:2]
    radius_of_particles = label[:,2] 
    
    plt.imshow(image_of_particle, cmap="gray", vmin = 0)
    plt.colorbar()
    ax = plt.gca()
    for i in range(n):
        rect = plt.Rectangle((label[i,0]-label[i,2],label[i,1]-label[i,2]),2*label[i,2],2*label[i,2],linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    
    #plt.scatter(position_of_particles[:,0], position_of_particles[:,1], marker='+', edgecolors="r", linewidth=2)
    plt.show()