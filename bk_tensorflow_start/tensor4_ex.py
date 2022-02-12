# 라이브러리 읽어 들이기
from PIL import ImageEnhance
import numpy as np
from PIL import Image

# 이미지 파일 읽어 들이기
image = Image.open('./data/pict/sample.jpg', 'r')

# 이미지 파일의 픽셀값 추출하기
image_px = np.array(image)
print(image_px)

# 이미지를 1차원 배열로 변환하기
image_flatten = image_px.flatten().astype(np.float32)/255.0
print(image_flatten)

# 이미지 픽셀값(배열)의 크기 출력하기
print(len(image_flatten))

# 이미지를 그레이스케일로 변환하기
gray_image = image.convert('L')

# 이미지 파일을 픽셀값으로 변환하기
gray_image_px = np.array(gray_image)
print(gray_image_px)

# 이미지를 1차원 배열로 변환하기
gray_image_flatten = gray_image_px.flatten().astype(np.float32)/255.0
print(gray_image_flatten)

# 이미지 픽셀값(배열)의 크기 출력하기
print(len(gray_image_flatten))


# 이미지의 채도 조정하기
conv1 = ImageEnhance.Color(image)
conv1_image = conv1.enhance(0.5)


# 이미지의 명도 조정하기
conv2 = ImageEnhance.Brightness(image)
conv2_image = conv2.enhance(0.5)


# 이미지의 콘트라스트 조정하기
conv3 = ImageEnhance.Contrast(image)
conv3_image = conv3.enhance(0.5)


# 이미지의 날카로움 조정하기
conv4 = ImageEnhance.Sharpness(image)
conv4_image = conv4.enhance(2.0)


# 가공한 이미지 저장하기
conv4_image.save("./data/pict/sample_conv.jpg")
