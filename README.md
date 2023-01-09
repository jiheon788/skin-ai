# SKIN-AI

> Teachable Machine을 활용한 악성흑색종 진단 AI 웹 서비스

## 👨🏻‍💻 Team

|                   박지헌                   |               홍준기               |    
| :----------------------------------------: | :--------------------------------: | 
| [@jiheon788](https://github.com/jiheon788) | none | 
|                웹 개발 및 모델링               |             모델링 및 발표             |  

## 🤔 Background

- Problem: 피부 질환은 중 일반인들의 시각과 지식으로는 구분이 어려운 질환들이 존재한다.
- Solution: 데이터를 근거로 자가진단을 도울 수 있는 인공지능 서비스
- Vision: 일반인이 자신의 피부질환 가능성에 대해 자가 진단하여 더 큰 위험으로의 연결을 막고자 한다.

## ✨ Features & Pages

### 1. Home

![127 0 0 1_5500_web_index html](https://user-images.githubusercontent.com/90181028/211202836-b2318be0-9f87-4f7f-9b52-061cd9145715.png)

- Lottie 를 통한 애니메이션

### 2. 진단 전

![127 0 0 1_5500_web_diagnosis html](https://user-images.githubusercontent.com/90181028/211202840-c050779a-b1c6-4fa9-9a32-e1bf3b47d7ef.png)

- 드래그 앤 드랍으로 환부 사진을 등록 가능 하다

### 3. 진단 후

![image](https://user-images.githubusercontent.com/90181028/211202827-1d7e30e9-86f6-4245-aa7f-41c1cc807995.png)

- 진단 모델에서 사진을 분류한 후 결과를 출력한다.

## 🛠 Tech Stack

`HTML`, `CSS`, `JavaScript`, `Teachable Machine`, `Google Cloud`, `TensorFlowJs`, `Python`, `Hitpaw`

## ⚙️ System Architecture

![image](https://user-images.githubusercontent.com/90181028/206637646-3c9568ae-efb7-46af-a6a2-f13c12d9eda1.png)

## 👾 Modeling

### 데이터 수집 및 전처리
- Google, Naver, DermNet을 크롤링하여 총 15000장의 점과 흑색종의 이미지 데이터를 확보
- Teachable Machine의 이용 규격을 맞춰 crop
- 사진에 적용된 워터마크를 Hitpaw 툴을 이용해 제거
- Gray Scale
- 이미지 증식 기법 적용

### 학습 및 Hyper Parameter Tuning

- 구글의 오픈소스 인공지능 툴, `Teachable Machine`을 사용
- Epoch, Batch Size, Learning Rate 총 3가지 Hyper Parameter 변화에 따른 분류 성능을 측정 후 최적화 작업 진행
- 성능평가 지표로는 Accuracy와 Loss값을 사용해 최적화 작업 진행

![image](https://user-images.githubusercontent.com/90181028/211203581-7dc48d20-15eb-496a-9e49-b97c617bd960.png)

![image](https://user-images.githubusercontent.com/90181028/211203245-da35d8a3-e2fe-437d-8e88-3fc463c67fb1.png)





