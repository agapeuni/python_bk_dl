# 라이브러리 읽어 들이기 --- (※1)
import urllib.request

# URL과 저장 경로 지정하기
url = "http://wisdomjin.dothome.co.kr/bbs/data/photo/1366988126/YY.jpg"
savename = "yy.jpg"

# 다운로드 --- (※2)
urllib.request.urlretrieve(url, savename)
print("저장되었습니다...!")