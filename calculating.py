import opendatasets as od
import numpy as np
import csv

with open('G:\my_lovely_codes\math_model\iris\Iris.csv', 'w', newline='') as iris_file:
    writer = csv.writer(iris_file)
    writer.writerow(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

print("Iris dataset downloaded successfully.")
print("Dataset location:", iris)
