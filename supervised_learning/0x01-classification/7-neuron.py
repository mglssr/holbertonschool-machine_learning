#!/usr/bin/env python3
"""classification"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """Neuron class"""
    def __init__(self, nx):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter funtion for private attr W"""
        return (self.__W)

    @property
    def b(self):
        """getter funtion for private attr b"""
        return (self.__b)

    @property
    def A(self):
        """getter funtion for private attr A"""
        return (self.__A)

    def sigmoid_act(self, X):
        """sigmoid activate function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        self.__A = self.sigmoid_act(np.matmul(self.__W, X) + self.__b)
        return (self.__A)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        err = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return (err)

    def evaluate(self, X, Y):
        """evaluates neuron predictions"""
        B = self.forward_prop(X)
        cost = self.cost(Y, B)
        return (np.round(B).astype(int), cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        dz = A - Y
        self.__W = self.__W - alpha * (np.matmul(dz, X.T) / X.shape[1])
        self.__b = self.__b - alpha * (np.sum(dz) / X.shape[1])

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        i = 0
        costs = []
        iter = []
        while (i < iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            cost = self.cost(Y, A)
            costs.append(cost)
            iter.append(i)
            if verbose is True:
                if type(step) != int:
                    raise TypeError("step must be an integer")
                if step <= 0:
                    raise ValueError("step must be positive and <= iterations")
                if i % step == 0:
                    print(f"Cost after {i} iterations: {cost}")
            if graph is True:
                if i % step == 0:
                    plt.plot(iter, costs)
                    plt.xlabel("iteration")
                    plt.ylabel("cost")
                    plt.title("Training Cost")
                    plt.show()
            i += 1

        plt.plot(iter, costs)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("Training Cost")
        plt.show()
        return self.evaluate(X, Y)
