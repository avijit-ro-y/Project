#include <iostream>
#include <vector>
#include <string>

using namespace std;

// Structure to represent a student
struct Student {
    int id;
    string name;
    int age;
    string course;
};

// Function declarations
void addStudent(vector<Student>& students); 
void modifyStudent(vector<Student>& students);
void deleteStudent(vector<Student>& students);
void listStudents(const vector<Student>& students);

int main() {
    vector<Student> students; // Vector to store student records
    int choice;

    do {
        // Display menu
        cout << "\nStudent Management System    \n";
        cout << "   1. Add Student\n";
        cout << "   2. Modify Student\n";
        cout << "   3. Delete Student\n";
        cout << "   4. List Students\n";
        cout << "   5. Exit\n";
        cout << "   Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1:
                addStudent(students);
                break;
            case 2:
                modifyStudent(students);
                break;
            case 3:
                deleteStudent(students);
                break;
            case 4:
                listStudents(students);
                break;
            case 5:
                cout << "Exiting the program.\n";
                break;
            default:
                cout << "Invalid choice. Please try again.\n";
        }
    } while (choice != 5);

    return 0;
}

// Function to add a new student
void addStudent(vector<Student>& students) {
    Student newStudent;
    cout << "Enter Student ID: ";
    cin >> newStudent.id;
    cin.ignore(); // To clear the newline character from the buffer
    cout << "Enter Student Name: ";
    getline(cin, newStudent.name);
    cout << "Enter Student Age: ";
    cin >> newStudent.age;
    cin.ignore();
    cout << "Enter Student Course: ";
    getline(cin, newStudent.course);

    students.push_back(newStudent);
    cout << "Student added successfully!\n";
}

// Function to modify an existing student's details
void modifyStudent(vector<Student>& students) {
    int id, index = -1;
    cout << "Enter Student ID to modify: ";
    cin >> id;

    for (size_t i = 0; i < students.size(); i++) {
        if (students[i].id == id) {
            index = i;
            break;
        }
    }

    if (index != -1) {
        cout << "Enter new details for the student:\n";
        cout << "Enter Student Name: ";
        cin.ignore();
        getline(cin, students[index].name);
        cout << "Enter Student Age: ";
        cin >> students[index].age;
        cin.ignore();
        cout << "Enter Student Course: ";
        getline(cin, students[index].course);
        cout << "Student details updated successfully!\n";
    } else {
        cout << "Student with ID " << id << " not found.\n";
    }
}

// Function to delete a student by ID
void deleteStudent(vector<Student>& students) {
    int id, index = -1;
    cout << "Enter Student ID to delete: ";
    cin >> id;

    for (size_t i = 0; i < students.size(); i++) {
        if (students[i].id == id) {
            index = i;
            break;
        }
    }

    if (index != -1) {
        students.erase(students.begin() + index);
        cout << "Student with ID " << id << " deleted successfully!\n";
    } else {
        cout << "Student with ID " << id << " not found.\n";
    }
}

// Function to list all students
void listStudents(const vector<Student>& students) {
    if (students.empty()) {
        cout << "No students to display.\n";
        return;
    }

    cout << "\nList of Students:\n";
    for (const auto& student : students) {
        cout << "ID: " << student.id << ", Name: " << student.name
             << ", Age: " << student.age << ", Course: " << student.course << endl;
    }
}
