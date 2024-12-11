def floyd_line(n):
    sum = 0
    for i in range(0, n):
        sum += i
        if sum >= n:
            return i

print(floyd_line(3)) # --> 2
print(floyd_line(17)) # --> 6
print(floyd_line(22)) # --> 7
print(floyd_line(499502)) # --> 1000


def fruit_and_animals(s):
    a = s.split()
    gave = []
    took = []
    has = 10
    for i in range(0, len(a)):
        if a[i] == 'gave':
            gave.append(int(a[i+1]))
    for i in range(0, len(a)):
        if a[i] == 'has':
            has = int(a[i+1])
        elif a[i] == 'had':
            has = int(a[i+1])
    for d in range(0 , len(a)):
        if a[d] == 'took':
            took.append(int(a[d+1]))
    return has + int(sum(took))-(sum(gave))

# Проверьте себя
print(fruit_and_animals('The monkey has 10 apples and gave 3')) # должно вернуть 7
print(fruit_and_animals('The cat has 13 oranges and took 6')) # должно вернуть 19
print(fruit_and_animals('Kangaroo had 254 bananas and gave 1 banana')) # должно вернуть 253


def repeat_sum(l):
    sum = []
    for i in range(0, len(l)):
        for j in range(0, len(l[i])):
            sum.append(l[i][j])
            for k in range(i, len(l)):
                for d in range(0, len(l[k])):
                    if l[k][d] == l[i][j] and k != i:
                        sum.append(l[k][d])
            return sum

#print(repeat_sum([[1, 2, 3],[2, 8, 9],[7, 123, 8]])) # --> 10
#print(repeat_sum([[1], [2], [3, 4, 4, 4], [123456789]])) # --> 0
#print(repeat_sum([[1, 8, 8], [8, 8, 8], [8, 8, 8, 1]])) # --> 9


def new_list(lst):
    if sum(lst[1:-1]) == lst[-1]+lst[0]:
        return lst
    else:
        try:
            lst.pop()
            new_list(lst)
        except:
            return []

# Проверьте себя
print(new_list([1,2,3,4,5])) # должно вернуть []
print(new_list([1,-1])) # должно вернуть [1,-1]
print(new_list([100,0,-100])) # должно вернуть [100,0,-100]

def count_salutes(s):
    count = 0
    for i in range(0, len(s)):
        if s[i] == '>':
            for j in range(i, len(s)):
                if s[j] == '<' and i<j:
                    count += 1
    return count*2


print(count_salutes('>--->---<--<')) # --> 8
print(count_salutes('<----<---<-->')) # --> 0
print(count_salutes('>-<->-<')) # --> 6
print(count_salutes('<---->---<---<-->')) # --> 4
print(count_salutes('------')) # --> 0
print(count_salutes('>>>>>>>>>>>>>>>>>>>>>----<->')) # --> 42
print(count_salutes('<<----<>---<')) # --> 2
print(count_salutes('>')) # --> 0
