from Food_item import FoodItem
from Menu import Menu
from Order import Order
from Restaurent import Restaurent
from User import Customer,Admin,Employee

mamar_dokan=Restaurent("Mamar Dokan")



def Customer_menu():
    name=input("Enter name:")
    email=input("Enter mail:")
    phone=input("Enter phone:")
    address=input("Enter address:")
    customer=Customer(name=name,phone=phone,email=email,address=address)
    
    while True:
        print(f"Welcome{customer.name}")
        print("1.View Menu")
        print("2.Add item to cart")
        print("3.View cart")
        print("4.Pay bill ")
        print("5.Exit")
        
        choice=int(input("Enter your choice:"))
        if choice==1:
            customer.view_menu(mamar_dokan)
            
        elif choice==2:
            item_name=input("Enter item name:")
            item_quantity=int(input("Enter item quantity:"))
            customer.add_to_cart(mamar_dokan,item_name,item_quantity)
        
        elif choice==3:
            customer.view_cart()
        
        elif choice==4:
            customer.pay_bill()
        
        elif choice==5:
            break
        
        else:
            print("Invalid input!")
    
    


def Admin_menu():
    name=input("Enter name:")
    email=input("Enter mail:")
    phone=input("Enter phone:")
    address=input("Enter address:")
    ad=Admin(name=name,phone=phone,email=email,address=address)
    while True:
        print(f"Welcome{ad.name}")
        print("1.Add New Item")
        print("2.Add new Employee")
        print("3.View Employee")
        print("3.View Item")
        print("4.Delete Item ")
        print("5.Exit")
        
        choice=int(input("Enter your choice:"))
        if choice==1:
            item_name=input("Enter Item Nme:")
            item_price=input("Enter Item Price:")
            item_quantity=input("Enter Item Quantity:")
            item=FoodItem(item_name,item_price,item_quantity)
            ad.add_new_item(mamar_dokan,item)
            
        elif choice==2:
            Employee_name=input("Enter Employee name:")
            Employee_Phone=input("Enter Employee phone:")
            Employee_email=input("Enter Employee email:")
            Employee_designation=input("Enter Employee designation:")
            Employee_age=input("Enter Employee age:")
            Employee_salary=input("Enter Employee salary:")
            Employee_address=input("Enter Employee address:")
            employee=Employee(Employee_name,Employee_email,Employee_Phone,Employee_address,Employee_age,Employee_designation,Employee_salary)
            ad.add_employee(mamar_dokan,employee)
            2
        elif choice==3:
            ad.view_employee(mamar_dokan)
        elif choice==4:
            ad.view_menu(mamar_dokan)
            item_name=input("Enter item name:")
            ad.remove_item(mamar_dokan,item_name)
        elif choice==5:
            break
        else:
            print("Invalid input!")
    
while True:
    print("Welcome!!")
    print("1.Customer")
    print("2.Admin")
    print("3.Exit")
    
    choice=int(input("Enter your choice:"))
    if choice==1:
        Customer_menu()
    elif choice==2:
        Admin_menu()
    elif choice==3:
        break
    else:
        print("Invalid")
    