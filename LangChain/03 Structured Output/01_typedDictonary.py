from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

new_person : Person = {
    'name': "Rocky",
    'age' : 31
}

print(new_person)



# Python doesn't enforce what keys should exist.

# With TypedDict, you can define the expected structure: