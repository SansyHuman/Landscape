import csv
import os.path

import numpy as np


csv.field_size_limit(np.iinfo(np.int32).max)

filename = input("Enter file name to load: ")
norm_size = int(input("Enter the number of data from each theory: "))
save_url = input("Enter the file name to save data: ")

data = None
with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

field_content_index = -1
for i in range(len(data[0])):
    if data[0][i] == "Name":
        field_content_index = i

field_contents = dict()

for i in range(1, len(data)):
    field_content = data[i][field_content_index]
    if field_content not in field_contents:
        field_contents[field_content] = []
    field_contents[field_content].append(i)

with open(save_url, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data[0])

    for field_content, indices in field_contents.items():
        if len(indices) < norm_size:
            continue

        chosen = np.random.choice(indices, norm_size, replace=False)
        for index in chosen:
            writer.writerow(data[index])
