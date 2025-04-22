from abc import ABC
from Order import Order
from Menu import Menu
from  Food_item import FoodItem
from Restaurent import Restaurent 


class User(ABC):
    def __init__(self,name,phone,email,address):
        super().__init__()
        self.name=name
        self.email=email
        self.address=address
        self.phone=phone

class Customer(User):
    def __init__(self, name, phone, email, address):
        super().__init__(name, phone, email, address)
        self.cart=None
    
    def view_menu(self,restaurent):
        restaurent.menu.show_menu()
    
    def add_to_cart(self,restaurent,item_name,quantity):
        item=restaurent.menu.find_item(item_name)
        if item:
            if quantity>item.quantity:
                print("Item quantity excedded")
            else:
                item.quantity=quantity
                self.cart.add_item(item)
        else:
            print("Item not found!")
        
    def view_cart(self):
        print("Your Cart")
        print("Name\tPrice\tQuantity")
        for item, quantity in self.cart.items.items():
            print(f"{item.name}{item.price}{quantity}")
        print(f"Total price:{self.cart.total_price}")

    def pay_bill(self):
        print(f"Total {self.cart.total_price} paid successfull!")
        self.cart.clear()
class Admin(User):
    def __init__(self, name, phone, email, address):
        super().__init__(name, phone, email, address)
        
    
    def add_employee(self,restaurent,employee):
        restaurent.add_employee(employee)
    
    def view_employee(self,restaurent):
        restaurent.view_employee()
    
    def add_new_item(self,restaurent,item):
        restaurent.menu.add_menu_item(item)
        
    def remove_item(self,restaurent,item):
        restaurent.menu.remove_item(item)
    
    def view_menu(self,restaurent):
        restaurent.menu.show_menu()


class Employee(User):
    def __init__(self, name, phone, email, address,age,designation,salary):
        super().__init__(name, phone, email, address)
        self.age=age
        self.designation=designation
        self.salary=salary

            
# ad=Admin("Avijit Roy",7658865197,"avijitroy@gmail.com","Bangladesh",)
# ad.add_employee("Sagor",999,"sagorgmail","bd",23,"chief",1200)
# ad.view_employee()

# mn=Menu()
# item=FoodItem("Pizza",1200,2)

# mn.add_menu_item(item=item)
# mn.show_menu()


# mama=Restaurent("Mama")
# ad1=Admin("abba",44333,"3fgdf","3fwdf")
# ad1.add_new_item(mama,item)
# cus1=Customer("aaa",33333,"dddf","bdf")
# cus1.view_menu(mama)

# itemname=input("Enter item name:")
# itemquantity=input("Enter item quantity:")
# cus1.add_to_cart(restaurent=mama,item_name=itemname,quantity=itemquantity)
# cus1.view_cart()