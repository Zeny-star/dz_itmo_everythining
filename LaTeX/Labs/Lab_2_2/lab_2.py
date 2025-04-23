import numpy as np

delta_H = np.array([5])/1000
delta_h = np.array([5])/1000
H2 = np.array([144, 145, 140, 142, 144, 145, 142, 143, 140, 142])
H1 = np.array([163, 160, 165, 163, 162, 160, 161, 164, 164, 162])
H = np.array([19, 15, 25, 21, 18, 15, 19, 21, 24, 20])/1000
h1 = np.array([152, 153, 153, 153, 153, 153, 153, 153, 153, 153])
h2 = np.array([154, 155, 153, 154, 152, 153, 153, 154, 152, 153])/1000
h = np.array([2, 2, 0, 1, -1, 0, 0, 1, -1, 0])/1000


gamma = H/(H-h)
gamma_mean = np.mean(gamma)
delta_gamma = H/(H-h)*np.sqrt((delta_H/H)**2 + (delta_h)**2)
delta_gamma = np.mean(delta_gamma)

print(gamma_mean, delta_gamma)

