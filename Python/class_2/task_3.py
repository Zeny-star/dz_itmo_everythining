class Building:
    def __init__(self, type_of_building, address):
        self.type_of_building = type_of_building
        self.address = address
        self.info = {}
        self.open_status = False

    def set_info(self, online, site, floor, schedule, owner_name):
        self.info = {
            "online": online,
            "site": site,
            "floor": floor,
            "schedule": schedule,
            "owner_name": owner_name
        }

    def get_info(self):
        if not self.info:
            return "Information about this building is not set."
        return (
            f"Type: {self.type_of_building}, Address: {self.address}, Online: {self.info['online']}, "
            f"Website: {self.info['site']}, Floor: {self.info['floor']}, "
            f"Schedule: {self.info['schedule']}, Owner: {self.info['owner_name']}"
        )

    def open(self):
        self.open_status = True
        return f"The {self.type_of_building} at {self.address} is now open."

    def close(self):
        self.open_status = False
        return f"The {self.type_of_building} at {self.address} is now closed."

    def is_open(self):
        return self.open_status


class Restaurant(Building):
    def __init__(self, address, cuisine_type, average_price):
        super().__init__(type_of_building="Restaurant", address=address)
        self.cuisine_type = cuisine_type
        self.average_price = average_price
        self.menu = []

    def add_dish(self, dish_name, price):
        self.menu.append({"dish": dish_name, "price": price})
        return f"{dish_name} added to the menu for {price}."

    def show_menu(self):
        if not self.menu:
            return "The menu is empty."
        menu_items = "\n".join([f"{item['dish']} - {item['price']}" for item in self.menu])
        return f"Menu:\n{menu_items}"

    def calculate_average_price(self):
        if not self.menu:
            return "The menu is empty, cannot calculate average price."
        total_price = sum(item["price"] for item in self.menu)
        return f"The average price of dishes is {total_price / len(self.menu):.2f}."


building = Building("Cafe", "123 Main St")
building.set_info(True, "www.cafe123.com", 2, "9 AM - 9 PM", "John Doe")
print(building.get_info())
print(building.open())
print(building.is_open())
print(building.close())
print(building.is_open())

restaurant = Restaurant("456 Elm St", "Italian", 20)
restaurant.set_info(True, "www.italianplace.com", 1, "11 AM - 11 PM", "Mario Rossi")
print(restaurant.get_info())
print(restaurant.add_dish("Pasta Carbonara", 15))
print(restaurant.add_dish("Margherita Pizza", 12))
print(restaurant.show_menu())
print(restaurant.calculate_average_price())
print(restaurant.open())
