def whoami():
    print('Турчанин Евгений Павлович')
def unique(a):
    filtered = []
    for i in a:
        if i not in filtered:
            filtered.append(i)
    return filtered

