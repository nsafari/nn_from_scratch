# Simplified SGD optimizer
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self, grads):
        for param, grad in zip(self.parameters, grads):
            param -= self.lr * grad