import numpy as np


class Adaline(object):
    def __init__(self, iterations: int = 100, random_state: int = 1, learning_rate: float =0.01):
        self.iterations = iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    def net_input(self, x):
        """
        Combines the input signals with respective weights and gets weighted sum
        :param x: the net input
        :return: the weighted sum
        """
        weighted_sum = np.dot(x, self.coefficient_[1:]) + self.coefficient_[0]
        return weighted_sum

    @staticmethod
    def activation_function(x):
        """
        Output from the activation funciton is same as input to it
        :param x: net input
        :return: x (the net input)
        """
        return x

    def predict(self, x):
        """
        Returns a prediction of of 1 if the output of the activation funciton is is >= 0
        else it returns 0
        :param x: the net input
        :return: bool: 1 or 0
        """
        return np.where(self.activation_function(self.net_input(x)) >= 0.0, 1, 0)

    def fit(self, x, y):
        """
        Does the batch gradient descent
        :param x: net_input
        :param y: expected value
        :return:
        """
        rgen = np.random.RandomState(self.random_state)
        self.coefficient_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        for _ in range(self.iterations):
            activation_function_output = self.activation_function(self.net_input(x))
            errors = y - activation_function_output
            self.coefficient_[1:] = self.coefficient_[1:] + self.learning_rate * x.T.dot(errors)
            self.coefficient_[0] = self.coefficient_[0] + self.learning_rate * errors.sum()

    def score(self, x, y):
        misclassified_data_count = 0
        for xi, target in zip(x, y):
            output = self.predict(xi)
            if target != output:
                misclassified_data_count += 1
        total_data_count = len(x)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_
