def polynomial_evaluate(a, x):
    result = 0
    for i in range(len(a)):
        result += a[i] * x ** i
    return result

def polynomial_add(a, b):
    result = []

    for i in range(max(len(a), len(b))):
        if i < len(a):
            a_k = a[i]
        else:
            a_k = 0

        if i < len(b):
            b_k = b[i]
        else:
            b_k = 0

        result.append(a_k + b_k)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result


def polynomial_derivative(a):
    result = []

    for i in range(1, len(a)):
        result.append(a[i] * i)

    while len(result) > 0 and result[-1] == 0:
        result.pop()

    if result == []:
        result = [0]
    return result

#print(polynomial_derivative([1, 2, 3, 0, 1, 2]))


def polynomial_multiply(a, b):
    result = [0]*(len(a) + len(b)-1)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i+j] += a[i] * b[j]

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result


def polynomial_find_root(a):
    step = 0.00005
    beg = -100
    end = 100
    while beg <= end:
        for i in range(len(a)):
            sum = a[i]*beg**i
            if abs(sum)<0.0001:
                return beg
        beg += step


def ackermann(m, n):
    if m == 0:
        return n + 1
    elif m > 0 and n == 0:
        return ackermann(m - 1, 1)
    elif m > 0 and n > 0:
        return ackermann(m - 1, ackermann(m, n - 1))

def polynomial_to_string(a):
    result = ""
    if a[-1] > 0:
        if a[-1]==1:
            result += "x^"+str(len(a)-1)+" "
        else:
            result += str(a[-1])+'x^'+str(len(a)-1)+' '
    elif a[-1] < 0:
        if a[-1]==-1:
            result += '-'+"x^"+str(len(a)-1)+" "
        else:
            result += '- '+str(abs(a[-1]))+'x^'+str(len(a)-1)+' '
    else:
        result += ''
    for i in range(len(a)-2, 0, -1):
        if a[i] <= 0:
            if a[i] == -1:
                if i == 1:
                    result += "- "+ "x" + ' '
                else:
                    result += "- "+ "x^" + str(i) + ' '
            elif a[i] == 0:
                result += ""
            else:
                if i == -1:
                    result += "- "+str(abs(a[i])) + "x" + ' '
                else:
                    result += '-'+str(abs(a[i])) + "x^" + str(i) + ' '
        elif a[i] > 0:
            if a[i] == 1:
                if i == 1:
                    result += "+ "+ "x" + ' '
                else:
                    result += "+ "+ "x^" + str(i) + ' '
            else:
                if i == 1:
                    result += "+ "+str(a[i]) + "x" + ' '
                else:
                    result += "+ "+str(a[i]) + "x^" + str(i) + ' '
    if a[0] > 0:
        result += "+ "+str(a[0])
    elif a[0] < 0:
        result += "- "+str(abs(a[0]))
    else:
        result = result[:-1]
    return result


def happy_new_year(seg):
    print(' '*seg + '<>')
    i = 1
    while i < seg+1:
        print(' '*seg + '/\\ ')
        k=1
        while k <= i :
            if k < i:
                print(' '*(seg-k) + '/'+ ' '*(k-1) + ' ' +' '*(k-1) + ' ' + '\\ ' )
                k+=1
            if k == i:
                print(' '*(seg-k) + '/' + '_'*k*2 + '\\ ')
                break
        i+=1
    print(' '*seg + '||')
#print(polynomial_add([1, 2], [0, 2, 3]))


