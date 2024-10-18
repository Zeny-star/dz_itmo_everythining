import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



h1 = np.array([202, 202, 202, 202, 202,
      213, 213, 213, 213, 213,
      222, 222, 222, 222, 222,
      231, 231, 231, 231, 231,
      242, 242, 242, 242, 242
      ])
#h
h2 = np.array([192, 192, 192, 192, 192,
      193, 193, 193, 193, 193,
      194, 194, 194, 194, 194,
      195, 195, 195, 195, 195,
      195, 195, 195, 195, 195,
      ])
#h'
t1 = np.array([1.4, 1.3, 1.3, 1.3, 1.3,
      0.9, 0.9, 0.9, 0.9, 0.9,
      0.7, 0.7, 0.7, 0.7, 0.7,
      0.6, 0.6, 0.6, 0.6, 0.6,
      0.6, 0.5, 0.5, 0.6, 0.5,
      ])
t2 = np.array([4.5, 4.4, 4.4, 4.4, 4.4,
      3.0, 3.0, 3.0, 3.0, 3.0,
      2.5, 2.5, 2.5, 2.5, 2.5,
      2.1, 2.1, 2.1, 2.1, 2.1,
      1.9, 1.9, 1.9, 1.9, 1.9,
      ])
x_c = 0.22
x_c_1 = 1.00
h_0 = 192
h_0_1 = 192
delta_x_c=0.005
delta_h=0.0005
x1 = 0.15
x2 = 1.10



n_series = 5
sin_alpha = []
for i in range(5):
    sin_alpha.append((h1[i*5]-h2[i*5])/((x_c_1-x_c)*1000))
sin_alpha = np.array(list(sin_alpha))

t1_mean = np.array(np.mean(t1[i:i+5]) for i in range(0, len(t1), 5))
t2_mean = np.array(np.mean(t2[i:i+5]) for i in range(0, len(t2), 5))

t1_delta = np.array(np.std(t1[i:i+5]) for i in range(0, len(t1), 5))
t2_delta = np.array(np.std(t2[i:i+5]) for i in range(0, len(t2), 5))

a = [2 * (x1 - x2) / (t2_mean**2 - t1_mean**2)]
delta_a = []
print(delta_a)
