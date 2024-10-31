import matplotlib.pyplot as plt
import pandas as pd
import csv
x = []
y = []

# Load data
with open('2_Measurement.csv') as File:
    plots = csv.reader(File, delimiter = ',')
    for row in plots:
        x.append(row[0])
        y.append(row[0])

plt.bar(x, y, color = 'g', width = 0.72, label = "Age") 
plt.xlabel('Names') 
plt.ylabel('Ages') 
plt.title('Ages of different persons') 
plt.legend() 
plt.show()
