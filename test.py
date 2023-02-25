import numpy as np

y = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
x = np.array([])


def haha(y):
    y[1, 1] = 10000

haha(y)

print(y)