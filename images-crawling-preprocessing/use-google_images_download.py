# -*- coding: utf-8 -*-
from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"아토피 피부염, 건선, 여드름, 백선, 어루러기, 모낭염, 헤르페스, 알레르기 접촉 피부염, 장미색 비강진, 기저 세포암, 편평 세포암, 악성 흑색종, 광선 각화증, 지루 각화증, 남성형 탈모, 백반증, 특발 물방울 모양 멜라닌 저하증, 지루피부염, 주사, 피부낭종, 흑색점, 멜라닌 세포모반, 사마귀, 피지샘 증식증, 피부섬유종, 혈관종, 환관종, 비립종, 쥐젖, 원형 탈모, 흉터, 두드러기 ", "limit":50, "print_urls":True, "format": "jpg"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images