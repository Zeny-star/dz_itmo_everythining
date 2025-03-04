import numpy as np

H = np.array([215])
delta_H = np.array([5])
h = np.array([200])
delta_h = np.array([5])


gamma = H/(H-h)
gamma_mean = np.mean(gamma)
delta_gamma = H/(H-h)*np.sqrt((delta_H/H)**2 + (delta_h/h)**2)
print(gamma_mean, delta_gamma)

