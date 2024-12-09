import random
class User:
    def __init__(self, username, email, age, country, is_online):
        self.username = username
        self.email = email
        self.age = age
        self.country = country
        self.is_online = is_online

    def display_info(self):
        return print("имя:", self.username, "почта:", self.email, "возраст:", self.age, "страна:", self.country, "онлайн:", self.is_online)

    def change_is_online(self):
        self.is_online = not self.is_online

    def send_greeting(self):
        return print(random.choice(["Вечер в хату", "Привет", " Доброе утро"]), self.username)

class Admin(User):
    def __init__(self, username, email, age, country, is_online):
        super().__init__(username, email, age, country, is_online)
        self.ban_list = set()
        self.admin_list= set(username)

    def display_ban_list(self):
        return print("стена банов - ", self.ban_list)

    def block_user(self, user):
        self.ban_list.add(user)
        return print(user, "был пуплично посрамлен")

    def add_admin(self, new_admin):
        self.admin_list.add(new_admin)
        return print(self.username, "пригласил в палаты:", {new_admin.username})

user = User("evgenii", "evgenii@gmail.com", 52, "RU", True)
admin = Admin("super_admin", "admin@gmail.com", 3, "RU", True)
admin.block_user(user="evgenii")
admin.display_ban_list()
user.display_info()

