import numpy as np
import matplotlib.pyplot as plt
p_0 = 104
V_c = np.array([50, 60, 70, 80, 90, 100, 110, 120])
def MNK(x, y:np.array, q:bool):
  x_av = sum(x)/len(x)
  y_av = sum(y)/len(y)
  up_sum = 0
  down_sum = 0
  if q:
    for i in range(len(x)):
      up_sum += (x[i] - x_av)*(y[i] - y_av)
      down_sum += (x[i] - x_av)**2
    b = up_sum/down_sum
    a = y_av - b*x_av
    d_sum = 0
    for i in range(len(x)):
      d_sum += (y[i] - (a + b*x[i]))**2
    Delta_a = 2*((1/len(x) + x_av**2/down_sum)*(d_sum/(len(x)-2)))**0.5
    Delta_b = 2*((d_sum/(len(x)-2))/down_sum)**0.5
    return [[b, Delta_b, Delta_b/b],[a, Delta_a, Delta_a/a]]
  else:
    for i in range(len(x)):
      up_sum += x[i]*y[i]
      down_sum += x[i]**2
    b = up_sum/down_sum
    d_sum = 0
    for i in range(len(x)):
      d_sum += (y[i] - b*x[i])**2
    Delta_b = 2*(d_sum/(down_sum*(len(x) - 1)))**0.5
    return [b, Delta_b, Delta_b/b]

Delta_P_11 = np.array([35.6, 14.6, 2.9, -9.2, -27.7, -38.9, -48.1, -52.1])
Delta_P_21 = np.array([35.6, 0.3, -13, -25.2, -33.7, -41.5, -47.3, -52.1])
Delta_P_12 = np.array([18.2, 1.9, -12.1, -23.3, -32, -39.6, -51.3, -55.1])
Delta_P_22 =  np.array([12.9, -6.1, -18.6, -28.9, -37.4, -44.9, -50.2, -55.1])
Delta_P_13 = np.array([12.6, -4.2, -15.9, -27.9, -35.5, -42.5, -49.3, -53.2])
Delta_P_23 = np.array([12.5, -1.3, -15.2, -27, -35.4, -42.8, -48.7, -53.2])
Delta_P_14 = np.array([15.4, -1.7, -14.7, -25.4, -34.8, -41.4, -50.8, -55.3])
Delta_P_24 = np.array([11.7, -3.9, -18.5, -28.6, -37.2, -44.7, -50, -55.4])
Delta_P_15 = np.array([13.6, -4.5, -15.5, -27.6, -36.2, -42.4, -49, -53.3])
Delta_P_25 = np.array([13.6, -0.2, -16.7, -26.5, -35.5, -42.4, -48.1, -53.3])



#Delta_P_11 = np.array([42.3, 23.8, 5.6, -7.1, -16.8, -24.8, -31.8, -37.5])
#Delta_P_21 = np.array([42.5, 21.6, 6.8, -5.5, -16.9, -24.9, -31.8, -37.6])
P_1 = p_0 + (Delta_P_11 + Delta_P_21)/2
LC_1 = 1/P_1
P_2 = p_0 + (Delta_P_12 + Delta_P_22)/2
LC_2 = 1/P_2
P_3 = p_0 + (Delta_P_13 + Delta_P_23)/2
LC_3 = 1/P_3
P_4 = p_0 + (Delta_P_14 + Delta_P_24)/2
LC_4 = 1/P_4
P_5 = p_0 + (Delta_P_15 + Delta_P_25)/2
LC_5 = 1/P_5


fig =plt.figure(dpi = 300)
ax = fig.gca()
ax.set_xlabel('$1/p, \; кПа^{-1}$')
ax.set_ylabel('$V_{ц}, \; мл $')
x_ = np.arange(0.006, 0.025, step = 0.00001)
ax.scatter(LC_1, V_c, s = 3, marker = '.', color ='yellow', zorder = 3)
ax.scatter(LC_2, V_c, s = 3, marker = '.', color ='green', zorder = 3)
ax.scatter(LC_3, V_c, s = 3, marker = '.', color ='red', zorder = 3)
ax.scatter(LC_4, V_c, s = 3, marker = '.', color ='orange', zorder = 3)
ax.scatter(LC_5, V_c, s = 3, marker = '.', color ='blue', zorder = 3)
ax.plot(x_, x_*MNK(LC_1, V_c, 0)[0], color = 'yellow', zorder = 2, label = '$V_{ц}(1/p)_1$', linewidth = 0.8)
ax.plot(x_, x_*MNK(LC_2, V_c, 0)[0], color = 'green', zorder = 2, label = '$V_{ц}(1/p)_2$', linewidth = 0.8)
ax.plot(x_, x_*MNK(LC_3, V_c, 0)[0], color = 'red', zorder = 2, label = '$V_{ц}(1/p)_3$', linewidth = 0.8)
ax.plot(x_, x_*MNK(LC_4, V_c, 0)[0], color = 'orange', zorder = 2, label = '$V_{ц}(1/p)_4$', linewidth = 0.8)
ax.plot(x_, x_*MNK(LC_5, V_c, 0)[0], color = 'blue', zorder = 2, label = '$V_{ц}(1/p)_5$', linewidth = 0.8)
ax.set_title('$Графики \; зависимостей \; V_{ц}(1/p)$')
ax.legend()
plt.show()


LC = [LC_1, LC_2, LC_3, LC_4, LC_5]
K = []
T = np.array([23.3, 32.2, 42.3, 52.0, 62.6])
T = np.array([62.6, 52.0, 42.3, 32.2, 23.3])
for i in range(len(LC)):
  K.append(MNK(LC[i]/1000, V_c/10**6, False)[0])
print(K)
print(MNK(T, K, True)[1][0]/MNK(T, K, True)[0][0])
for i in range(len(K)):
  print(i+1, '& $ ', T[i], '$ & $', round(K[i], 2), '\\\\')
  print('\\hline')
  
print(MNK(T, K, True))
print(((MNK(T, K, True)[0][2])**2 + (MNK(T, K, True)[1][2])**2)**0.5)
fig =plt.figure(dpi = 300)
ax = fig.gca()
ax.set_xlabel('$t, \; ^{\circ} C$')
ax.set_ylabel('$K, \; Дж $')
x_ =  np.arange(20, 66, step = 0.001)
ax.scatter(T, K, s = 3, marker = '.', color ='black', zorder = 3)
ax.plot(x_, x_*MNK(T, K, True)[0][0] + MNK(T, K, True)[1][0], color = 'yellow', zorder = 2, label = '$K(t)$', linewidth = 0.8)
ax.grid(which='both', linewidth=0.5, linestyle='-')
ax.set_title('$График \; зависимости \; K(t)$')
ax.legend()
plt.show()


P = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
T_star = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(8):
  for j in range(len(LC)):
    P[i][j] = 1/LC[j][i]
print(P)
for i in range(5):
  print('$ ', T[i], '$ & $ ', P[0][i], '$ & $' , P[1][i], '$ & $' , P[2][i], '$ & $' , P[3][i], '$ & $' , P[4][i], '$ & $' , P[5][i], '$ & $' , P[6][i], '$ & $' , P[7][i], '$ \\\\' ,)
  print('\\hline')
fig =plt.figure(dpi = 300)
ax = fig.gca()
ax.set_xlabel('$T, \; ^{\circ} C$')
ax.set_ylabel('$p, \; кПа$')
x_ =  np.arange(20, 66, step = 0.001)
ax.scatter(T, P[0], s = 3, marker = '.', color ='black', zorder = 3)
ax.scatter(T, P[4], s = 3, marker = '.', color ='black', zorder = 3)
ax.scatter(T, P[7], s = 3, marker = '.', color ='black', zorder = 3)
ax.plot(x_, x_*MNK(T, P[0], True)[0][0] + MNK(T, P[0], True)[1][0], color = 'yellow', zorder = 2, label = '$p(t)_{50}$', linewidth = 0.8)
ax.plot(x_, x_*MNK(T, P[4], True)[0][0] + MNK(T, P[4], True)[1][0], color = 'green', zorder = 2, label = '$p(t)_{90}$', linewidth = 0.8)
ax.plot(x_, x_*MNK(T, P[7], True)[0][0] + MNK(T, P[7], True)[1][0], color = 'blue', zorder = 2, label = '$p(t)_{120}$', linewidth = 0.8)
ax.grid(which='both', linewidth=0.5, linestyle='-')
ax.set_title('$Графики \; зависимостей \; p(t)$')
ax.legend()
plt.show()
for i in range(len(P)):
  T_star[i] = -1*MNK(T, P[i], True)[1][0]/MNK(T, P[i], True)[0][0]
  print(round(T_star[i],2), end = '$ & $')
print(T_star)
for i in range(len(V_c)):
    inv_V = 1 / V_c[i]
    t_star = T_star[i]
    print(f"{V_c[i]:^9} | {inv_V:.6f}    | {t_star:.2f}")
print("----------------------------------------------")
#
# T_star_ = 
# print(MNK(1/V_c, T_star, True))
# print(MNK(V_star, T_star_, True))
# fig =plt.figure(dpi = 1000)
# ax = fig.gca()
# ax.set_xlabel('$1/V_{ц}, \; мл^{-1}$')
# ax.set_ylabel('$t_{*}, \; ^{\circ} C $')
# x_ =  np.arange(-0.0010, 0.020, step = 0.00001)
# ax.scatter(1/V_c, T_star, s = 3, marker = '.', color ='black', zorder = 3)
# ax.scatter(V_star, T_star_, s = 3, marker = '.', color = 'orange', zorder = 3)
# ax.plot(x_, x_*MNK(1/V_c, T_star, True)[0][0] + MNK(1/V_c, T_star, True)[1][0], color = 'blue', zorder = 2, label = '$t_{*}(1/V_{ц})$', linewidth = 0.8)
# ax.plot(x_, x_*MNK(V_star, T_star_, True)[0][0] + MNK(V_star, T_star_, True)[1][0], color = 'red', zorder = 2, label = '$t_{**}(1/V_{ц})''$', linewidth = 0.8)
# ax.set_title('$График \; зависимостей \; t_{*}(1/V_{ц}) \; и \; t_{**}(1/V_{ц})$')
# ax.grid(which='both', linewidth=0.5, linestyle='-')
# ax.legend()
# plt.show()

