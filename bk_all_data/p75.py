import csv
import matplotlib.pyplot as plt

with open("data/sample_1973.csv") as f:
    data = csv.reader(f)
    for i in range(8):
        next(data)

    high = []
    avg = []
    low = []
    day = []
        
    for row in data :
        if row[-1] != '' and row[-2] != '' :
            date = row[0].split('-')
            
        if date[1] == '03' and date[2] == '02' :
            high.append(float(row[-1]))
            low.append(float(row[-2]))
            avg.append(float(row[-3]))
            day.append(date[0])

plt.rc('font', family = 'Malgun Gothic')            
plt.rcParams['axes.unicode_minus'] = False
plt.title('광복절날 기준 기온 변화')

plt.plot(day, high, 'hotpink', label = '최고기온')
plt.plot(day, low, 'skyblue', label = '최저기온')
plt.plot(day, avg, 'darkgreen', label = '평균기온')

plt.legend()
plt.show()