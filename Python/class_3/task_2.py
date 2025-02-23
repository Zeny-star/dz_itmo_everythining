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
def bracket_pairs(s):
    f = []
    pairs = {}
    for i, j in enumerate(s):
        if j == '(':
            f.append(i)
        elif j == ')':
            try:
                pairs[f[-1]] = i
                f.pop()
            except:
                return False
    return pairs

# Пример использования
print(bracket_pairs("len(list)")) # --> {3:8}
print(bracket_pairs("string")) # --> {}
print(bracket_pairs("")) # --> {}
print(bracket_pairs("def f(x")) # --> False
print(bracket_pairs(")(")) # --> False
print(bracket_pairs("(a(b)c()d)")) # --> {0:9,2:4,6:7}
print(bracket_pairs("f(x[0])")) # --> {1:6}



def cats_and_mice(mapping, moves):
    cat_cord_y = 0
    cat_cord_x = 0
    mouse_cord_y = 0
    mouse_cord_x = 0
    y_cord =0

    if 'C' in mapping and 'm' in mapping:
        for y in mapping.split():
            y_cord +=1
            if 'C' in y:
                cat_cord_x=y.index('C')
                cat_cord_y=y_cord
            if 'm' in y:
                mouse_cord_x=y.index('m')
                mouse_cord_y=y_cord
        if abs(mouse_cord_y-cat_cord_y)+abs(mouse_cord_x-cat_cord_x) < moves:
            return 'caught'
        else:
            return 'run'
    else:
        return 'We need two animals!'


print(cats_and_mice("""\
            ..C......
            .........
            ....m....""", 6)) # должно вернуть Caught!
print(cats_and_mice("""\
            .C.......
            .........
            ......m..""", 6)) # должно вернуть Run!
print(cats_and_mice("""\
            ..C......
            .........
            .........""", 6)) # должно вернуть We need two animals!
