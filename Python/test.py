def foo(*args):
    print(sum(args))

foo(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
def bar(**kwargs):
    print(sum(kwargs.values()))

bar(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=10)
