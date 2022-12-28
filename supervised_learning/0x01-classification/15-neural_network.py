#!/usr/bin/env python3
"""classification"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork():
    """defines a neural network with one
    hidden layer performing binary classification"""
    def __init__(self, nx, nodes):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter"""
        return (self.__W1)

    @property
    def b1(self):
        """getter"""
        return (self.__b1)

    @property
    def A1(self):
        """getter"""
        return (self.__A1)

    @property
    def W2(self):
        """getter"""
        return (self.__W2)

    @property
    def b2(self):
        """getter"""
        return (self.__b2)

    @property
    def A2(self):
        """getter"""
        return (self.__A2)

    def sigmoid_act(self, X):
        """sigmoid activate function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__A1 = self.sigmoid_act(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self.sigmoid_act(np.matmul(self.__W2, self.__A1)
                                     + self.__b2)
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return (cost)

    def evaluate(self, X, Y):
        """evaluates the neural network’s predictions"""
        B = self.forward_prop(X)[1]
        cost = self.cost(Y, B)
        return (np.round(B).astype(int), cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
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
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cost = self.cost(Y, A2)
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
