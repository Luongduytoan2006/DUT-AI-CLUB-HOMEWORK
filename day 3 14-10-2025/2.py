class Student:
    def __init__(self, name, yob, grade):
        self.name = name
        self.yob = yob
        self.grade = grade
    
    def describe(self):
        print(f"Student - Name: {self.name} - YoB: {self.yob} - Grade: {self.grade}")


class Teacher:
    def __init__(self, name, yob, subject):
        self.name = name
        self.yob = yob
        self.subject = subject
    
    def describe(self):
        print(f"Teacher - Name: {self.name} - YoB: {self.yob} - Subject: {self.subject}")


class Doctor:
    def __init__(self, name, yob, specialist):
        self.name = name
        self.yob = yob
        self.specialist = specialist
    
    def describe(self):
        print(f"Doctor - Name: {self.name} - YoB: {self.yob} - Specialist: {self.specialist}")


class Ward:
    def __init__(self, name):
        self.name = name
        self.list_person = []
    
    def addPerson(self, person):
        self.list_person.append(person)
    
    def describe(self):
        print(f"Ward Name: {self.name}")
        for person in self.list_person:
            person.describe()
    
    def countDoctor(self):
        count = 0
        for person in self.list_person:
            if isinstance(person, Doctor):
                count += 1
        return count
    
    def sortAge(self):
        self.list_person.sort(key=lambda person: person.yob, reverse=True)
    
    def aveTeacherYearOfBirth(self):
        teachers = [person for person in self.list_person if isinstance(person, Teacher)]
        if len(teachers) == 0:
            return 0
        total_yob = sum(teacher.yob for teacher in teachers)
        return total_yob / len(teachers)


student1 = Student(name="studentA", yob=2010, grade="7")
student1.describe()

teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher1.describe()

doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor1.describe()

teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")
ward1 = Ward(name="Ward1")
ward1.addPerson(student1)
ward1.addPerson(teacher1)
ward1.addPerson(teacher2)
ward1.addPerson(doctor1)
ward1.addPerson(doctor2)
ward1.describe()

print(f"Number of doctors: {ward1.countDoctor()}")

ward1.sortAge()
print("After sorting by age (ascending):")
ward1.describe()

average_yob = ward1.aveTeacherYearOfBirth()
print(f"Average year of birth of teachers: {average_yob}")