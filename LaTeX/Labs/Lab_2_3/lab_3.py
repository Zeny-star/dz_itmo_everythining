import numpy as np


T_0 = 25+273
w = 1500
mu = 29/1000
R = 8.314
i = 5

x_1 = np.array([118, 115, 120, 118, 113])
x_2 = np.array([224, 227, 235, 226, 230])
x_3 = np.array([341, 337, 348, 344, 342])
x_4 = np.array([472, 471, 475, 463, 468])
x_5 = np.array([581, 586, 583, 570, 572])
x_1 = x_1/1000
x_2 = x_2/1000
x_3 = x_3/1000
x_4 = x_4/1000
x_5 = x_5/1000

l_12 = x_2 - x_1
l_23 = x_3 - x_2
l_34 = x_4 - x_3
l_45 = x_5 - x_4

l_mean = np.array([np.mean(l_12), np.mean(l_23), np.mean(l_34), np.mean(l_45)])
l = np.mean(l_mean)
v = 2*l*w
gamma = v**2*mu/(R*T_0)
gamma_th = (i+2)/i
print(gamma, gamma_th)
