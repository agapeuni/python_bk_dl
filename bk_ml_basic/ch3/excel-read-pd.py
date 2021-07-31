import pandas as pd

# 엑셀 파일 열기 --- (※1)
# 첫 번째 줄부터 헤더
book = pd.read_excel("stats_104103.xlsx", "stats_104102", header=1)

# 2019년 인구로 정렬 --- (※2)
book = book.sort_values(by=2019, ascending=True)
print(book)