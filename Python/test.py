import numpy as np

a = np.array([1,2,3,4,5,6])
print(a.reshape(3,2).transpose())
print(a.reshape(2,-1))
print(a.ravel())

