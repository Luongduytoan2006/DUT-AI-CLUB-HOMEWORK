class Manufacturer:
    def __init__(self, identify, location):
        self.identify = identify
        self.location = location

    def describe(self):
        print(f"Manufacturer: {self.identify}")
        print(f"Location: {self.location}")

class Device:
    def __init__(self, name, price, **kwargs):
        self.name = name
        self.price = price
        self.manufacturer = Manufacturer(**kwargs)
        
    def describe(self):
        print(f"Device Name: {self.name}")
        print(f"Price: {self.price}")
        self.manufacturer.describe()

device1 = Device ( name =" mouse ", price =2.5 , identify =9725 , location =" Vietnam ")
device1.describe()

print(20*"-")

device2 = Device ( name =" keyboard ", price =5.0 , identify =1234 , location =" USA ")
device2.describe()


