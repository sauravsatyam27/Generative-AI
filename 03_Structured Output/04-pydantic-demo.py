from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name:str 
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt =0, lt=10, default=5, description='Decimal value representing CGPA of the studemt')


new_student = {
    'name' : "Virat",
    # 'age' : 10  ## What is type coursing 
    'email' : 'abc@gmail.com',
    'cgpa' : 70
}

student = Student(**new_student)

print(student)
print(type(student))



student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()