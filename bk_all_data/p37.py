import csv

all_min_temp = 100
all_min_date = ''
all_max_temp = -100
all_max_date = ''

# CSV 파일열기
with open("data/sample.csv") as input:
    csv_reader = csv.reader(input)

    # 메타정보, 헤더
    for i in range(8):
        next(csv_reader)
        
    min_temp = 0
    max_temp = 0

    for row in csv_reader:
        min_temp = float(row[3])        
        max_temp = float(row[4])

        # 최저기온 처리
        if(all_min_temp > min_temp):
            all_min_temp = min_temp
            all_min_date = row[0]

        # 최고기온 처리
        if(all_max_temp < max_temp):
            all_max_temp = max_temp
            all_max_date = row[0]
                    
print(all_min_date, "역대 최저기온 =", all_min_temp)
print(all_max_date, "역대 최고기온 =", all_max_temp)