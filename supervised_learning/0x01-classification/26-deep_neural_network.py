#!/usr/bin/env python3
"""classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork():
    """defines a deep neural network with one
    hidden layer performing binary classification"""

    def __init__(self, nx, layers):
        """class constructor"""
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for ly in range(self.__L):
            if type(layers[ly]) != int or layers[ly] < 0:
                raise TypeError("layers must be a list of positive integers")
            s_l = str(ly + 1)
            self.__weights["b" + s_l] = np.zeros((layers[ly], 1))
            aux = nx
            if (ly > 0):
                aux = layers[ly - 1]
            self.__weights["W" + s_l] = np.random.randn(layers[ly], aux)
            self.__weights["W" + s_l] *= np.sqrt(2/aux)

    @property
    def L(self):
        """comment"""
        return (self.__L)

    @property
    def cache(self):
        """comment"""
        return (self.__cache)

    @property
    def weights(self):
        """comment"""
        return (self.__weights)

    def sigmoid_act(self, X):
        """sigmoid activate function"""
        return (1 / (1 + np.exp(-X)))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for ly in range(1, self.L + 1):
            aux = self.cache["A" + str(ly - 1)]
            Z = np.matmul(self.weights["W" + str(ly)], aux)
            Z += self.weights["b" + str(ly)]
            self.__cache["A" + str(ly)] = self.sigmoid_act(Z)
        return self.cache["A" + str(ly)], self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        cost = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)).mean()
        return (cost)

    def evaluate(self, X, Y):
        """evaluates neuron predictions"""
        B, _ = self.forward_prop(X)
        cost = self.cost(Y, B)
        return (np.round(B).astype(int), cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ gradient """
        m = Y.shape[1]
        dZ = cache["A" + str(self.L)] - Y
        for lay in range(self.L, 0, -1):
            A = self.cache["A" + str(lay - 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dW = np.matmul(dZ, A.T) / m
            dZ = np.matmul(self.weights["W" + str(lay)].T, dZ) * (A * (1 - A))
            self.__weights["W" + str(lay)] -= dW * alpha
            self.__weights["b" + str(lay)] -= db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ train """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if graph or verbose:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            if (i % step) == 0:
                cost = self.cost(Y, A)
                costs.append([cost, i])
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))
            self.gradient_descent(Y, cache, alpha)
        A, cost = self.evaluate(X, Y)
        if verbose:
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            costs.append([cost, iterations])
            costs = np.array(costs).T
            plt.plot(costs[1], costs[0])
            plt.show()
        return A, cost

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        ext = ".pkl"
        if filename.endswith(ext):
            pass
        else:
            filename = filename + ext
        filename = open(filename, 'wb')
        pickle.dump(self, filename)
        filename.close()

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if filename is None:
            return
        else:
            file = open(filename, 'rb')
            new = pickle.load(file)
            file.close()
            return (new)