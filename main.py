# import py_perceptron as pyp
#
# perceptron = pyp.Perceptron(datasource='heart_data.txt')
# perceptron.data_description()
# perceptron.set_weights([0.001, 0.001, 0.001])
# perceptron.train()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data_set = np.loadtxt("diabetes_data.txt")
print(data_set)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

