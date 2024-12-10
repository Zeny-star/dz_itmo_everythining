class Building:
    def __init__(self, type_of_building, address):
        self.type_of_building = type_of_building
        self.address = address
        self.info = {}
        self.open_status = False

    def set_info(self, online, site, floor, schedule, owner):
        self.info = {
            "онлайн": online,
            "сайт": site,
            "этаж": floor,
            "расписание": schedule,
            "владелец": owner}

    def get_info(self):
        return (
            f"Кухня: {self.type_of_building}, адрес: {self.address}, онлайн: {self.info['онлайн']}, сайн: {self.info['сайт']}, этаж: {self.info['этаж']}, расписание: {self.info['расписание']}, владелец: {self.info['владелец']}"
        )

    def open(self):
        self.open_status = True
        return f"{self.type_of_building} открыто"

    def close(self):
        self.open_status = False
        return f"{self.type_of_building} закрыто"

    def is_open(self):
        return self.open_status



class Restaurant(Building):
    def __init__(self, address, cuisine_type):
        super().__init__(type_of_building="ресторан", address=address)
        self.cuisine_type = cuisine_type


building = Building("Кофе", "Адрес Григорий Горбушкина")
building.set_info(True, "www.cafe123.com", 2, "с 8 утра, до 8 утра", "Я")
print(building.get_info())
print(building.open())
print(building.is_open())
print(building.close())
print(building.is_open())

restaurant = Restaurant("St Pet", "Italian")
restaurant.set_info(True, "www.coc.com", 1000, "от заката, до рассвета", "Григорий Горбушкин")
print(restaurant.get_info())
print(restaurant.open())
