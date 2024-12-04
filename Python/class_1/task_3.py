import numpy as np

mat_1=[[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]


mat_2=np.random.randint(1, 11, (4, 4))

A = np.array([[1, 5, 7], [2, 4, 6], [3, 8, 9]])
prices = np.array([130.5, 132.1, 131.4, 135.7, 136.5, 138.2, 139.0])


def mat(mat_0):
    for i in range(0, 4):
        if (mat_0[i]!=mat_0[(i+1)%4]).all() and (mat_0[i]!=mat_0[(i+2)%4]).all() and (mat_0[i]!=mat_0[(i+3)%4]).all():
            for j in range(0, 4):
                print(mat_0[j])
        else:
           return np.random.randint(0, 10, (4, 4))
mat(mat_0=np.random.randint(0, 10, (4, 4))
)

foo = []
for i in range(0, 3):
    foo = []
    for j in range(0, 3):
        foo.append(A[j][i])
    print(max(foo))

sum = 0
for i in range(0, 4):
    sum += mat_2[i][i]


print(sum)
print(np.sum(mat_2)-sum)
print('mean_price:', np.mean(prices))
print('median_price:', np.median(prices))
print('std_price:', np.std(prices))


