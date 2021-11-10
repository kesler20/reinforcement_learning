import numpy as np 
from numpy import ndarray
import pandas as pd

class Kernel(object):

    def __init__(self, x_m, x_n):
        self.x_m: ndarray = x_m/np.argmax(x_m)
        self.x_n: ndarray = x_n/np.argmax(x_n)
        self.result : ndarray = {}
    
    def __repr__(self):
        return f'''
        {
            {self.result}
        }'''
    
    @property
    def gaussian_kernel(self, sigma: int = 0.3):
        self.result = np.exp(-np.linalg.norm(self.x_m - self.x_n, axis=1) ** 2 / (2 * (sigma ** 2)))
        return self.result
    
    @property
    def laplace_kernel(self, alpha: int =0.8):
        self.result = np.exp(-np.linalg.norm(self.x_m - self.x_n, axis=1) * alpha)
        return self.result
    
    @property
    def linear_kernel(self):
        self.result = np.dot(self.x_m, self.x_n.T)
        return self.result

    @property
    def polynomial_kernel(self, p: int =10):
        self.result = (1 + np.dot(self.x_m, self.x_n.T)) ** p
        return self.result
    
    def softmax(self, z_vector):
        exponential = np.exp(z_vector)
        probabilities = exponential / np.sum(exponential)
        return probabilities
    
    def normalize(self):
        z_vector = np.sqrt((self.x_m)**2 - (self.x_n)**2)
        dataframe = pd.DataFrame(self.softmax(z_vector))
        print(dataframe)
        probabilities = dataframe.loc[dataframe.iloc[:,0]<0.0005,:]
        return probabilities


