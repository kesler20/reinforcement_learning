import numpy as np 
from numpy import ndarray


class Kernel(object):

    def __init__(self, x_m, x_n):
        self.x_m: ndarray = x_m
        self.x_n: ndarray = x_n
    
    @property
    def gaussian_kernel(self, sigma: int = 0.3):
        return np.exp(-np.linalg.norm(self.x_m - self.x_n, axis=1) ** 2 / (2 * (sigma ** 2)))
    
    @property
    def laplace_kernel(self, alpha: int =0.8):
        return np.exp(-np.linalg.norm(self.x_m - self.x_n, axis=1) * alpha)
    
    @property
    def linear_kernel(self):
        return np.dot(self.x_m, self.x_n.T)

    @property
    def polynomial_kernel(self, p: int =10):
        return (1 + np.dot(self.x_m, self.x_n.T)) ** p

