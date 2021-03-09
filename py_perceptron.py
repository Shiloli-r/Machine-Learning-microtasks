"""
    This library is an implementation of the perceptron algorithm in ANN
"""
import numpy as np


class Perceptron:
    def __init__(self, datasource: str, learning_rate: float = 0.1, threshold: float = 0.5):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.data = np.genfromtxt(datasource)
        self.weights = self.set_weights()
        self.data_set = self.load_data(datasource)
        self.error_count = 0

    def data_description(self):
        print("Description of the Data")
        print("rows: ", self.data.shape[0])
        print("columns: ", self.data.shape[1])
        print("shape: ", self.data.shape)
        return self.data.shape

    def product(self, input_vector):
        """
        Calculates the dot product of the input vector and the weights
        :param input_vector:
        :return:
        """
        return np.dot(input_vector, self.weights)

    def load_data(self, datasource: str):
        """
        The function prepares the data into list. Each list entry represents a row of the dataset
        split into two i.e ((index 0....n-1, index n)
        :param datasource: a filename, given as a string
        :return: a list of the data, whose entry is described above
        """
        data = np.genfromtxt(datasource)
        dataset = []
        for row in data:
            dataset.append((tuple(row[0:data.shape[1] - 1]), row[-1]))
        self.data_set = dataset
        return dataset

    def set_weights(self, weights: list = None):
        """
        This function sets the weights if given, or sets them to default of 0
        :return: a list of weights
        """
        if weights:
            if len(weights) == self.data.shape[1] - 1:
                self.weights = weights
                return self.weights
        return list(np.random.randint(1, size=self.data.shape[1] - 1))

    def epoch(self):
        """
        This function does a single epoch
        :return: error_count
        """
        self.error_count = 0
        for input_vector, desired_output in self.data_set:
            print(self.weights)
            dot_product = self.product(input_vector)
            result = dot_product > self.threshold
            error = desired_output - result
            if error != 0:
                self.error_count += 1
                for index, value in enumerate(input_vector):
                    self.weights[index] += self.learning_rate * error * value
        return self.error_count

    def train(self):
        """
        Trains the model by continuously calling the epoch function until the error is 0
        :return:
        """
        count = 0
        error = self.epoch()
        while error != 0:
            print("=" * 15, "Epoch Number  ",  count, "=" * 20)
            count += 1
            error = self.epoch()
