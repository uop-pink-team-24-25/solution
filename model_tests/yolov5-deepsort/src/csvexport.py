## This script reads a python dictionary and writes it to a CSV file.
## The dictionary is expected to have a specific structure:

import csv

dictionary = [{'key': ['1'], 'Vehicle Type': ['Car'], 'Vehicle Colour': ['Red'], 'Object Start Frame': [1, 2, 3], 'Object End Frame': [4, 5, 6], 'Objects No Longer In Scene': [0, 1, 2]},
              {'key': ['2'], 'Vehicle Type': ['Car'], 'Vehicle Colour': ['Red'], 'Object Start Frame': [1, 2, 3], 'Object End Frame': [4, 5, 6], 'Objects No Longer In Scene': [0, 1, 2]}]

csv_filename = 'data.csv'

field_names = ['key','Vehicle Type', 'Vehicle Colour', 'Object Start Frame', 'Object End Frame', 'Objects No Longer In Scene']

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader() 
    writer.writerows(dictionary)