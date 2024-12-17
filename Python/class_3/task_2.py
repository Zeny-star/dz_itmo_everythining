def repeat_sum(l):
    lst = []
    a = []
    res = 0
    for i in l:
        for j in i:
            if j in lst and j not in a:
                res += j
                a.append(j)
        lst = lst + i
    return res
print(repeat_sum([[1, 2, 3],[2, 8, 9],[7, 123, 8]])) # --> 10
print(repeat_sum([[1], [2], [3, 4, 4, 4], [123456789]])) # --> 0
print(repeat_sum([[1, 8, 8], [8, 8, 8], [8, 8, 8, 1]])) # --> 9

def bracket_pairs(s):
    f = []
    pairs = {}
    for i, j in enumerate(s):
        if j == '(':
            f.append(i)
        elif j == ')':
            if not f:
                return False
            pairs[f.pop()] = i
    return pairs

# Пример использования
print(bracket_pairs("len(list)")) # --> {3:8}
print(bracket_pairs("string")) # --> {}
print(bracket_pairs("")) # --> {}
print(bracket_pairs("def f(x")) # --> False
print(bracket_pairs(")(")) # --> False
print(bracket_pairs("(a(b)c()d)")) # --> {0:9,2:4,6:7}
print(bracket_pairs("f(x[0])")) # --> {1:6}

