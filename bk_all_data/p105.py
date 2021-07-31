import matplotlib.pyplot as plt
import numpy as np
import csv

human = []
with open("data/human_kor.csv") as f:
    data = csv.reader(f)
    for row in data:
        for cnt in row[3:]:
            human.append(int(cnt.replace(',' , '')))

plt.title("Korea Population (2020.08)")
plt.plot(range(101), human, "r")
plt.show()
