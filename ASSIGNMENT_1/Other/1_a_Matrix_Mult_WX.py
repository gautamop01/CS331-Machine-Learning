import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class MatrixMultiplyLayer:
    def forward(self, X, W):
        self.X = X
        self.W = W
        return np.dot(X, W)

    def backward(self, grad_output):
        grad_X = np.dot(grad_output, self.W.T)
        grad_W = np.dot(self.X.T, grad_output)
        return grad_X, grad_W
