import numpy as np
import deeptrack as dt

IMAGE_SIZE = 100

class MultiParticle(dt.Feature):

    def get(self, image, n_particles, radius, **kwargs):
        
        position = np.zeros((n_particles , 2))

        X, Y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        mask = np.zeros(np.shape(X))
        
        for i in range(n_particles):
            position[i,:] = np.random.rand(2) * (100)
            
            mask = mask + ((X - int(position[i, 0])) ** 2 + (Y - int(position[i, 1])) ** 2 <= radius ** 2)*(0.3+np.random.rand()*.3)

        mask = mask / np.max(mask)

        mask  = -1*(mask - 1)
        
        return mask

class Updater(dt.Feature):
    
    __distributed__ = False
    
    def __init__(self, feature=None, **kwargs):
        super().__init__(features=feature, **kwargs)
        
    def get(self, image, features, **kwargs):
        return features.resolve(
            np.ones(
                (IMAGE_SIZE,IMAGE_SIZE)
                )
            )
