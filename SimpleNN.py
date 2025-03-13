import numpy as np

# Neural Network Implementation
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # y: True labels 
    def backward(self, X, y ):
        # Backward pass
        m = y.shape[0] # the number of samples in the batch
        dz2 = self.a2 - y  # gradient of the loss with respect to z2
        self.dw2 = np.dot(self.a1.T, dz2) / m # gradient of the loss with respect to w2
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m # gradient of the loss with respect to b2
        dz1 = np.dot(dz2, self.w2.T) * (1 - np.tanh(self.z1) ** 2) # gradient of the loss with respect to z1 (hidden layer pre-activation)
        self.dw1 = np.dot(X.T, dz1) / m # gradient of the loss with respect to w1 (weights of the hidden layer)
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m # gradient of the loss with respect to b1 (biases of the hidden layer)

    def get_parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def get_gradients(self):
        return [self.dw1, self.db1, self.dw2, self.db2]
