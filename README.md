# <img align="left" src="https://user-images.githubusercontent.com/90181028/206636097-de0a1ae2-18ae-4612-89f7-1a1f5c0458a0.png" width="100px" /> SKIN-AI

> Teachable Machine을 활용한 악성흑색종 진단 AI 웹 서비스

<br>
<div align='left'>
<img src="./skin-ai-mainpage.png" height="300px"/>
</div>

# 📑 INDEX

1. [INTRODUCTION](#1-introduction)
2. [기획 배경](#2-기획-배경)
3. [시스템 구성 및 설계](#3-시스템-구성-및-설계)
4. [Modeling](#4-modeling)  
   4.1. [데이터 수집 및 전처리](#41-데이터-수집-및-전처리)  
   4.2. [Hyper Parameter Tuning](#42-hyper-parameter-tuning)  
   4.3. [Gray Scale](#43-gray-scale)
5. [결론](#5-결론)

# 1. INTRODUCTION

> 피부질환은 역사 속에서 꾸준히 인류와 함께하며 작고 큰 고통을 사람들에게 주어왔다. 또한, 바쁜 현대사회의 폐해로 잊고 지나가는 피부질환도 있었으며, 최근에 G7 선진국 대열에 들어서는 상황에서야 주위를 둘러보는 여유를 잠깐 가지게 되었다. 최근, 빠른 속도로 발전하는 의학기술의 발전으로 과거에 비해 질환이 가져다주는 위험성은 감소하였으나, 다양하고 초기대응을 놓치면 큰 질환으로 발전하게 될 피부질환과, 이에 대한 치료 시기를 놓쳐 큰 위험을 겪는 사례가 여전히 존재한다. 그 중, 악성흑색종의 경우에는 상당히 커질 때까지 별다른 증상이 없고, 점과 유사한 모습을 보여 초기 진단이 어려운 탓에 초기대응을 놓쳐 깊은 질환으로 이어진다는 문제가 있다. 본 프로젝트에서는 구글의 오픈소스 인공지능 툴, ‘Teachable Machine’을 사용하여, 악성흑색종의 자가진단법인 ‘ABCDE 규칙’에 근거하는 5가지 시각적 단서를 참고해, 사용자의 사진 기반으로 악성흑색종과 점을 분류·진단해주는 인공지능 서비스를 구현하려 한다. 세부 파라미터값을 조정하여 분류·진단 성능을 최적화하고, 제안된 모델을 통해 일반인이 자신의 피부질환 가능성에 대해 자가진단하여 더 큰 위험으로의 연결을 막고자 한다. - _‘Teachable Machine을 활용한 악성흑색종 진단 AI 웹서비스 개발’ 초록 발췌_

- 악성 흑색종 진단 AI 웹 서비스
- 아주대학교 2022-1 '인공지능기초' 수업 프로젝트
- 2022 한국스마트미디어학회 논문 승인 - ‘Teachable Machine을 활용한 악성흑색종 진단 AI 웹서비스 개발’

# 2. 기획 배경

피부는 신체구조에서 가장 바깥쪽에 있어 질환이 발생하는 경우 비교적 빠르게 시각적 인지가 가능하다. 그러나 검버섯과 기저세포암, 모낭염과 여드름, 점(모반)과 악성흑색종처럼, 일반인들의 시각과 지식으로는 구분이 어려운 질환들이 존재한다. 이 중 멜라닌 세포에 생긴 암, 악성흑색종은 다른 장기로의 전이 전 초기 진단이 매우 중요하다. 그 이유는 크기가 상당히 커질 때까지 가려움이나 통증 등의 증상이 없고, 점과 유사한 생김새 때문에 초기 진단을 놓쳐 전이가 진행되고 나서야 알아차려, 중증으로 발전 후 알게 되는 경우가 많기 때문이다. 흑색종의 확진은 대부분 조직검사가 필요하나 자가진단의 단서 ‘ABCDE 규칙’이 존재하여, 조직검사 없이 자가진단으로도 흑색종의 합리적 의심이 가능하다. 이를 통해 악성흑색종의 가능성을 분류해 내고자 한다. 그림1과 같이 시각적 단서를 발견할 수 있으면 악성흑색종의 가능성이 높아진다.

![image](https://user-images.githubusercontent.com/90181028/206637301-247a156d-9cc1-416b-9218-2e826e49e8e6.png)

- 그림 1. ABCDE 규칙

# 3. 시스템 구성 및 설계

그림 1에 근거한 시각적 단서를 이용하여 점과 흑색종을 분류·진단하고자 한다. 분류·진단을 위한 모델링은 구글의 오픈소스 소프트웨어이며, GUI기반으로 쉽게 학습모델에 접근하여 이를 TensorFlow 모델로 내보낼 수 있는 웹 기반 인공지능 플랫폼인 ‘Teachable Machine’을 사용하였다. Teachable Machine은 개인디바이스에서 학습 후 클라우드에 저장해 사용하는 MobileNet 방식이다. 웹을 통해 사용자의 사진이 업로드되면, 클라우드로부터 구현된 학습모델이 호출되어 점과 악성흑색종을분류해줄 수 있도록 아래에 나타난 실행순서로 프로그램을 설계하였다.

![image](https://user-images.githubusercontent.com/90181028/206637420-cda63dda-4a02-42ec-94f3-b3ace8fca481.png)

- 그림2. User 기반의 프로그램 실행순서

![image](https://user-images.githubusercontent.com/90181028/206637646-3c9568ae-efb7-46af-a6a2-f13c12d9eda1.png)

- 그림3. 시스템 구조도

# 4. Modeling

## 4.1. 데이터 수집 및 전처리

데이터는 Kaggle의 ‘Melanoma Detection Dataset’을사용하였다. 추가로 python의 Selenium library를 이용해 Google, Naver, DermNet을 크롤링하여 총 15000장의 점과 흑색종의 이미지 데이터를 확보하였다. Teachable Machine의 이용 규격을 맞추기 위해서 사진을 정사각형으로 자르고, 사진에 적용된 워터마크를 Hitpaw 툴을 이용해 제거하였다.

## 4.2. Hyper Parameter Tuning

각 7500개씩의 흑색종, 점 사진의 데이터를 업로드하여 모델을 만들었다. Training set와 Test set의 비율은 8:2로 설정되어 있다. Teachable Machine은 Epoch, Batch Size, Learning Rate 총 3가지 Hyper Parameter가 조정 가능하다.

Epoch = 50, Batich Size = 16으로 설정하고 Learning Rate 변화에 따른 분류 성능을 측정하였다. 성능평가 지표로는 Accuracy와 Loss값을 사용하였다. 그림4을 보면 Learning Rate = 0.00001에서 가장 높은 Accuracy와 가장 낮은 Loss값을 보인다.

![image](https://user-images.githubusercontent.com/90181028/206637702-ebd086fd-d011-4cc5-97b7-62db0159dd59.png)

- 그림4. 학습률 변화에 따른 모델 성능

## 4.3. Gray Scale

이미지 분류 문제에서 학습에 불필요한 배경, 색조를 제거하기 위해 gray scale을 적용한다. 흑색종 판별 문제에서도 유용한지 테스트를 위해 gray scale을 적용한 모델과 원본을 사용한 모델과의 성능 비교를 하였다. 두 모델의 Confusion Matrix를 만들어 보았을 때, 두 클래스 간 불균형성이 존재한다. 이런 경우 Accuracy는 신뢰도가 떨어지며 대신 정밀도(Precision)와 재현율(Recall)의 조화평균인 F1Score를 활용할 수 있다[2]. 식(1)과 그림5의 Confusion Matrix를 사용해 두 모델의 F1Score를 구하였다. 좌측의 F1Score는 0.93, 우측은 0.88 로 Gray Scale을 사용하였을 때 오히려 성능이 낮아짐을 확인할 수 있다. 이는 서론에서 언급하였던 흑색종의 단서‘ABCDE 규칙’중 Color variegation을 위배하였기 때문으로 보인다.

![image](https://user-images.githubusercontent.com/90181028/206636413-9b31852c-8a7c-4f78-a81e-dbdf0a79df8f.png)

![image](https://user-images.githubusercontent.com/90181028/206637822-7af55050-b7c6-46a5-95df-c32b58f95adb.png)

- 그림5. 좌: Color 우: Gray Confusion Matrix

# 5. 결론

사용자가 간편하게 사진을 통해 자가진단해 볼 수 있는 웹서비스를 설계 및 구현하였다. 충분한 데이터양을 확보, Open Source Software 인공지능툴‘Teachable Machine’으로 유의미한 분류 성능을 보이는모델을 만들 수 있었고, 제안된 모델을 통해 일반인이 지레짐작 아닌 데이터에 근거한 자가 진단이 가능하다. 점과 악성흑색종 이외의 다른 유사질환(e.g. 검버섯과 기저세포암, 모낭염과 여드름 등)의 경우에도 데이터 수를 많이 확보한다면, 높은 성능을 가진 모델을 만들 수 있을 것으로 기대된다.
