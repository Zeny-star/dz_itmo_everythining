import numpy as np
import matplotlib.pyplot as plt




m_karetki=0.047
delta_m_karetki=0.0005
m_shaibi = 0.22
delta_m_shaibi = 0.0005
m_gruzov=0.408
delta_m_gruzov=0.0005
l_ot_riski_do_oci=0.057
delta_l_ot_riski_do_oci=0.0005
l_mezdu_riski=0.025
delta_l_mezdu_riski=0.0002
diameter_ctypichi=0.046
delta_diameter_ctypichi=0.0005
diameter_gruza_na_krestovine=0.04
delta_diameter_gruza_na_krestovine=0.0005
h_gruza_na_krestovine=0.04
delta_h_gruza_na_krestovine=0.0005
h=0.7
delta_h=0.0005
g=9.8


t1=[4.47, 4.47, 4.53,#для m1
    3.25, 3.31, 3.32,#для m2
    2.56, 2.50, 2.63,#для m3
    2.32, 2.24, 2.32,#для m4
    ]
t2=[5.38, 5.35, 5.50,#для m1
    3.94, 3.94, 3.81,#для m2
    3.15, 3.13, 3.07,#для m3
    2.75, 2.72, 2.84,#для m4
    ]
t3=[6.28, 6.37, 6.32,#для m1
    4.53, 4.57, 4.53,#для m2
    3.65, 3.54, 3.66,#для m3
    3.29, 3.32, 3.36,#для m4
    ]
t4=[7.41, 7.49, 7.40,#для m1
    5.41, 5.34, 5.35,#для m2
    4.44, 4.37, 4.38,#для m3
    3.81, 3.81, 3.83,#для m4
    ]
t5=[8.55, 8.50, 5.58,#для m1
    6.09, 6.12, 6.09,#для m2
    5.06, 5.07, 5.13,#для m3
    4.19, 4.25, 4.24,#для m4
    ]
t6=[10.03, 9.92, 9.94,#для m1
    6.88, 6.84, 6.81,#для m2
    5.60, 5.56, 5.54,#для m3
    4.75, 4.72, 4.76,#для m4
    ]



#Поиск среднего
print(
    'Для m1 и расстояния l0', np.array(t1[:3]).mean(),
    'Для m2 и расстояния l0', np.array(t1[3:6]).mean(),
    'Для m3 и расстояния l0', np.array(t1[6:9]).mean(),
    'Для m4 и расстояния l0', np.array(t1[9:]).mean(),
    'Для m1 и расстояния l0+R', np.array(t2[:3]).mean(),
    'Для m2 и расстояния l0+R', np.array(t2[3:6]).mean(),
    'Для m3 и расстояния l0+R', np.array(t2[6:9]).mean(),
    'Для m4 и расстояния l0+R', np.array(t2[9:]).mean(),
    'Для m1 и расстояния l0+2R', np.array(t3[:3]).mean(),
    'Для m2 и расстояния l0+2R', np.array(t3[3:6]).mean(),
    'Для m3 и расстояния l0+2R', np.array(t3[6:9]).mean(),
    'Для m4 и расстояния l0+2R', np.array(t3[9:]).mean(),
    'Для m1 и расстояния l0+3R', np.array(t4[:3]).mean(),
    'Для m2 и расстояния l0+3R', np.array(t4[3:6]).mean(),
    'Для m3 и расстояния l0+3R', np.array(t4[6:9]).mean(),
    'Для m4 и расстояния l0+3R', np.array(t4[9:]).mean(),
    'Для m1 и расстояния l0+4R', np.array(t5[:3]).mean(),
    'Для m2 и расстояния l0+4R', np.array(t5[3:6]).mean(),
    'Для m3 и расстояния l0+4R', np.array(t5[6:9]).mean(),
    'Для m4 и расстояния l0+4R', np.array(t5[9:]).mean(),
    'Для m1 и расстояния l0+5R', np.array(t6[:3]).mean(),
    'Для m2 и расстояния l0+5R', np.array(t6[3:6]).mean(),
    'Для m3 и расстояния l0+5R', np.array(t6[6:9]).mean(),
    'Для m4 и расстояния l0+5R', np.array(t6[9:]).mean(),
    sep='\n'
)
print('Погрешность для m1 и расстояния l0', np.std(np.array(t1[:3])))

#Поиск ускорения 
a =[]
epsilon = []
moment = []
for i in range(0,4):
    a.append(np.array(t1[i*3:(i+1)*3]).mean())
for i in range(0,4):
    a.append(np.array(t2[i*3:(i+1)*3]).mean())
for i in range(0,4):
    a.append(np.array(t3[i*3:(i+1)*3]).mean())
for i in range(0,4):
    a.append(np.array(t4[i*3:(i+1)*3]).mean())
for i in range(0,4):
    a.append(np.array(t5[i*3:(i+1)*3]).mean())
for i in range(0,4):
    a.append(np.array(t6[i*3:(i+1)*3]).mean())
#Ускорение
for i in range(len(a)):
    a[i]=2*h/a[i]**2
#Угловое ускорение
for i in range(len(a)):
    epsilon.append(2*a[i]/diameter_ctypichi)
#Момент
for i in range(0, len(a), 4):
    moment.append((m_karetki+(m_shaibi*1))*diameter_ctypichi*(g-a[i])/2)
for i in range(1, len(a), 4):
    moment.append((m_karetki+(m_shaibi*2))*diameter_ctypichi*(g-a[i])/2)
for i in range(2, len(a), 4):
    moment.append((m_karetki+(m_shaibi*3))*diameter_ctypichi*(g-a[i])/2)
for i in range(3, len(a), 4):
    moment.append((m_karetki+(m_shaibi*4))*diameter_ctypichi*(g-a[i])/2)

print(moment)




