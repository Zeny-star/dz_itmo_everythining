#на вход строка  и список слов, вернуть словарь где в ключ - эл-т списка
words = ["apples"," ", "0"]
phrase = "John likes apples"
dict_of_words = {}
def foo(words, phrase):
    for i in words:
        if i in phrase:
            dict_of_words[i] = phrase.find(i)
        else:
            dict_of_words[i] = -1
    return dict_of_words
print(foo(words, phrase))





