import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix as mx
from kernels import Kernel
from numpy import ndarray 

def create_dataset(N, K=2):
    N = 100  # number of points per class
    D = 2
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K)  # class labels

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y

class SupportVectorMachine:

    def __init__(self, X: ndarray, y: ndarray, C: int =1,):
        self.C = C
        self.y = y
        self.X = X
        self.m, n = self.X.shape
        self.kernel = self.initialise_kernel()
    
    def initialise_kernel(self):
        kernel : ndarray = np.zeros((self.m, self.m))
        for i in range(self.m):
            k = Kernel(X[i, np.newaxis], self.X)
            kernel[i, :] = k.gaussian_kernel

        return kernel

    
    def fit(self):

        # solve convex optimization in order to get the values of alpha
        P = mx(np.outer(y, y) * self.kernel)
        q = mx(-np.ones((self.m, 1)))
        G = mx(np.vstack((np.eye(self.m) * -1, np.eye(self.m))))
        h = mx(np.hstack((np.zeros(self.m), np.ones(self.m) * self.C)))
        A = mx(y, (1, self.m), "d")
        b = mx(np.zeros(1))

        cvxopt.solvers.options["show_progress"] = False

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(solution["x"])

        return self.alphas

    def predict(self, X: ndarray):
        y_predict = np.zeros((X.shape[0]))
        sv = self.get_params(self.alphas)

        for i in range(X.shape[0]):
            #y_predict[i] = np.sum(self.alphas[sv]*self.y[sv, np.newaxis]*self.kernel(X[i], self.X[sv])[:, np.newaxis])
            y_predict[i] = 0 if sv[i] else 1

        return np.sign(y_predict + self.b)

    def get_params(self, alphas):
        cut_off = 1e-5

        sv = ((alphas > cut_off) * (alphas < self.C)).flatten()
        self.w = np.dot(X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
        self.b = np.mean(self.y[sv, np.newaxis] - self.alphas[sv] * self.y[sv, np.newaxis] * self.kernel[sv, sv][:, np.newaxis]
        )
        return sv


np.random.seed(1)
X, y = create_dataset(N=50)

print(X, y)

svm = SupportVectorMachine(X, y)
parameters = svm.fit()
print(parameters)

y_pred = svm.predict(X)

print(f"Accuracy: {100*sum(y==y_pred)/y.shape[0]} %")