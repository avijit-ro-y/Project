from Menu import *

class Restaurent:
    def __init__(self,name):
        self.name=name
        self.employees=[]
        self.menu=Menu()
        
    def add_employee(self,employee):
        self.employees.append(employee)
    
    def view_employee(self):
        print("Employee list : ")
        for emp in self.employees:
            print(emp.name,emp.email,emp.phone,emp.address)
