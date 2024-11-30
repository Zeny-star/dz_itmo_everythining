import matplotlib.pyplot as plt
import numpy as np

mat=[[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]

A = np.array([[1, 5, 7], [2, 4, 6], [3, 8, 9]])
prices = np.array([130.5, 132.1, 131.4, 135.7, 136.5, 138.2, 139.0])


foo = []
for i in range(0, 3):
    foo = []
    for j in range(0, 3):
        foo.append(A[j][i])
    print(max(foo))

print('mean_price:', np.mean(prices))
print('std_price:', np.std(prices))
