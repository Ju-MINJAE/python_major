## 📍 Use
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white">
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

## 2024_py


#### project1.py

Rogistic Regression 

1. 데이터 시각화
2. y = a*X + b
3. z = a*x1 + b*x2 + c

-----

#### project2_2.py

### Rogistic Regression (Z = a*feat1 + b*feat2 + c*feat3 + d)

로지스틱 회귀 모델을 사용하여 테스트 데이터에 대한 최적의 매개변수(a, b, c, d)를 찾고, 모델의 손실과 정확도를 출력합니다. 혼동 행렬을 시각화하고, 정확도, 정밀도, 재현율을 계산하여 출력하며, 과적합을 방지합니다.

주요기능
* 로지스틱 회귀 모델 구현
* 최적 매개변수(a, b, c, d) 찾기
* 모델 손실 및 정확도 출력
* 혼동 행렬
* 정확도, 정밀도, 재현율 계산
* 과적합 방지

-----


#### project2_3.py

### Decision Tree

의사결정 트리 분류기를 구현하여 데이터를 훈련하고, 트리를 구축하며, 새로운 데이터를 분류합니다. 혼동 행렬을 통해 모델 성능을 평가하고, 과적합 방지 방법을 적용합니다.

주요기능
* 의사결정 트리 분류기 구현
* 데이터 훈련 및 의사결정 트리 구축
* 새로운 데이터 분류
* 혼동 행렬 생성 및 정확도, 정밀도, 재현율 계산
* 과적합 방지

----

#### project3.py

### Fashion MNIST 분류 프로젝트

이 프로젝트는 Fashion MNIST 데이터셋을 전처리하고, scikit-learn의 다양한 머신러닝 알고리즘을 사용하여 분류하는 작업을 포함합니다. 주요 목표는 파라미터 조정 및 과적합 문제를 해결하여 모델 성능을 분석하고 향상시키는 것입니다.

#### 기능

1. Fashion MNIST 데이터 전처리:
   * Fashion MNIST 데이터셋의 물리적 특징을 추출하여 효과적으로 전처리합니다.
     
2. 사용된 분류기:
   * 로지스틱 회귀 (Logistic Regression)
   * 의사 결정 트리 (Decision Tree)
   * 다층 퍼셉트론 (MLP, Multi-layer Perceptron)
     
3. 파라미터 튜닝:
   * 로지스틱 회귀: C 값에 따른 정확도 변화를 분석합니다.
   * 의사 결정 트리: max_depth 값에 따른 정확도 변화를 분석합니다.
   * MLP: hidden_layer_size 값에 따른 정확도 변화를 분석합니다.
     
4. 과적합 방지:
   * 적합을 방지하고 모델이 보지 못한 데이터에 잘 일반화되도록 하기 위한 기술을 구현합니다.
     
5. 혼동 행렬:
   * 각 분류기의 성능을 평가하기 위해 혼동 행렬을 생성하고 분석합니다. 
