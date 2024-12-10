class Restaurant():
    def __init__(self, name, cusine):
        self.name=name
        self.cusine=cusine
        self.info=[]
        self.is_open = False
        self.rating = []


    def set_info(self, working_hours, owner, adress, http, rating):
        self.info = [working_hours, owner, adress, http, rating]

    def get_info(self):
        print('Часы работы:', self.info[0],'; ', 'Владелец:',self.info[1], '; ','Адрес:', self.info[2], '; ','Веб страница:', self.info[3],'; ','Рейтинг:', self.info[4])

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def get_rating(self):
        sum = 0
        for i in self.rating:
            sum+=i
        return sum/len(self.rating)

    def set_rating(self, mark):
        self.rating.append(mark)

res = Restaurant(name='Kit', cusine='the best')
res.set_rating(5)
res.set_rating(3)
print(res.get_rating())
res.set_rating(3)
print(res.get_rating())
res.set_info(10, 'Anton', 'Saint-Petesburg', 'google.com', res.get_rating())
res.get_info()




