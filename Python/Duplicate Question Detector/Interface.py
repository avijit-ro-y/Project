from Core import *

print('       Duplicate Question Checking       ')
while True:
    print("\n1.Check Duplicate")
    print("2.Exit")
    inp=input("Enter your choice:")
    if inp=="1":
        q1 = input("Enter question 1: ")
        q2 = input("Enter question 2: ")

        result=xgb.predict(query_point_creator(q1,q2))

        if result[0]==1:
            print("Duplicate!!")
        else:
            print("Non Duplicate!!")
            
        print("Result is:",result)
        print("Accuracy is:",accuracy)
    elif inp=="2":
        print("Thank you!!!")
        break
    else:
        print("Invalid Choice! Please Choose correct Option Again!")

# q1 = 'Where is the capital of India?'
# q2 = 'What is the current capital of Pakistan?'
# q3 = 'Which city serves as the capital of India?'
# q4 = 'What is the business capital of India?