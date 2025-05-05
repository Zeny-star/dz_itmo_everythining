import math



def sum_pr(n: int) -> float:

    x = []
    summa = 0
    for i in range (0, n+1):
        x.append(0.96*i/n)
    for j in range (1, n+1):
        summa=summa +1/math.sqrt(math.cos(x[j-1]) - math.cos(0.96))*0.96/n
    return summa

print((2*math.sqrt(2)/3.23)*sum_pr(105200))
#214503.53412220217 - 100000
#21491327.764522463 - 1000000
#2104.657276554137 - 100
