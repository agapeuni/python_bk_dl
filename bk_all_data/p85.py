import csv
import matplotlib.pyplot as plt

with open("data/sample_1973.csv") as f:
    data = csv.reader(f)
    for i in range(8):
        next(data)
            
    m02 = []
    m05 = []
    m08 = []
    m11 = []

    for row in data :
        month = row[0].split('-')[1]
        if row[-1] != '' :
            if month == '02':
                m02.append(float(row[2]))
            if month == '05':
                m05.append(float(row[2]))
            if month == '08':
                m08.append(float(row[2]))
            if month == '11':
                m11.append(float(row[2]))

plt.rc('font', family = 'Malgun Gothic')            
plt.rcParams['axes.unicode_minus'] = False
plt.title('2월, 5월, 8월, 11월 평균기온')
plt.hist(m11, bins = 100, color = 'gray', label = '11월')
plt.hist(m08, bins = 100, color = 'r', label = '8월')
plt.hist(m05, bins = 100, color = 'g', label = '4월')
plt.hist(m02, bins = 100, color = 'b', label = '2월')
plt.legend()
plt.show()
