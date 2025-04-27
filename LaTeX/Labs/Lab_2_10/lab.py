import numpy as np
import matplotlib.pyplot as plt



time = np.array([50.27, 107.03, 164.35, 219.44, 278.39, 335.15, 6.25, 440.84, 493.44, 548.15, 598.74, 652.87, 703.9, 757.15, 810.72, 14.19, 316.15, 16.11, 1024.06, 1072.29])
c = 4200
R = 82
U = 232
tao = 120
Delta_t = 10
M = 970/1000


N_okr = c*M*(Delta_t/tao)
N_full = U**2/R
P = N_full-N_okr
A = P*Delta_t


