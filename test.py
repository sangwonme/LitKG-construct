import json

# Sample list of dictionaries
list_of_dicts = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# Saving the list of dictionaries as a JSON file
with open('test.json', 'w') as file:
    json.dump(list_of_dicts, file)

# The file is saved as 'data.json' in a writable directory
file_path = 'test.json'
file_path
