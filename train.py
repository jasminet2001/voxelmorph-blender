import numpy as np

shape_A = np.zeros((32, 32, 32))
shape_A[8:24, 8:24, 8:24] = 1

shape_B = np.zeros((32, 32, 32))
shape_B[10:26, 10:26, 8:24] = 1
