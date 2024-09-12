## 📍 Use
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=Numpy&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=OpenCV&logoColor=white">
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

# 2022_py

## interpolation.py (Only Numpy)

### 🚀 기능 (Features)

- Homogeneous 변환 행렬을 이용한 이미지 회전 (Rotation of images using Homogenous Transformation Matrices)
- 후진 사상(`np.where()`)을 이용한 이미지 빈 영역 채우기 (Image filling using Backward Mapping (np.where()))
- 선형 보간법을 이용한 이미지 빈 영역 채우기 (Image filling using Bilinear Interpolation)
- `subplot()`과 `axis()`를 이용한 두 방법의 시각적 비교 (Visual comparison of the two methods using subplot() and axis())

### 💻 요구 사항 (Requirements)

- `NumPy`
- `Matplotlib`

### 🖼️ 이미지 변환 및 보간법 (Image Transformation and Interpolation)

Homogeneous 변환 행렬을 사용하여 이미지를 회전시키고, `np.where()`와 선형 보간법을 이용하여 빈 픽셀을 채우는 Python 프로젝트입니다.

A Python project that performs image rotation using a homogenous transformation matrix and fills in empty pixels using backward warping with np.where() and bilinear interpolation.

### 📚 문제 & 결과 (Problem & Result)

자신의 학번 끝 두 자리에 2를 곱한 값만큼 **Homogeneous 변환 행렬**을 사용하여 이미지를 회전시키세요.

Rotate an image based on the last two digits of your student ID, doubled, using a **Homogenous Transformation Matrix**.

![4-1](https://github.com/user-attachments/assets/1963dbf4-25d6-4377-9412-d41e87a7ceef)
![4-2](https://github.com/user-attachments/assets/27b7d62b-4eb5-46cc-8f6b-5d0b3a746069)

image scale이 변형되며 픽셀과 픽셀 사이의 값이 비워집니다. **Backward Mapping**을 이용하여 회전한 이미지의 빈 영역을 `np.where()`로 채우세요.

Use **Backward Mapping** to fill the empty spots in the rotated image by utilizing np.where().

![4-2(2)](https://github.com/user-attachments/assets/eb311ce2-9bfa-4f47-8fcd-9c32eab7c32f)

**1차 보간법**을 이용하여 회전 후 생긴 빈 영역을 메우세요.

Use **Bilinear Interpolation** to fill in the empty spots left after the rotation.

![4-3(2)](https://github.com/user-attachments/assets/ae32f55e-be81-4f77-afa3-d8bb7ccb64ae)

위 두 방법으로 보정한 이미지를 `subplot()`을 이용해 한 그림에 나란히 배치하고 확대하여 비교하세요. 비교 시 `axis()`를 사용해 이미지를 확대하여 성능을 평가하세요.

Compare the two methods visually by displaying both corrected images on a **subplot** within a single figure. The images should be enlarged to allow for a visual comparison of their effectiveness.

![4-4](https://github.com/user-attachments/assets/904ce5fb-8f3a-4262-ae4b-b18795bd8d93)


## image_processing.py

### 🚀 기능 (Features)

- 흑백 사진을 그레이 스케일로 변환 및 크기 축소 (Convert grayscale images and resize them to specified dimensions)
- `ginput(30)`을 사용하여 이미지에서 좌표 클릭 (Use `ginput(30)` to select and round coordinates from an image)
- 세 번 접은 후 왜곡된 이미지 처리 및 좌표 보정 (Process images distorted by folding and adjust coordinates)
- 좌표 기반 변형 및 워핑된 영상 생성 (Generate a warped image based on estimated transformation coefficients)


### 🖼️ 이미지 변형 및 좌표 변환 프로젝트 (Image Transformation and Interpolation)

Python을 사용하여 이미지를 축소, 좌표 클릭, 그리고 좌표 기반 변형을 수행하는 프로젝트입니다. 이 프로젝트는 세 번 접은 후 왜곡된 영상을 처리하여 변형하는 과정을 다룹니다.

A Python project that performs image scaling, coordinate clicking, and coordinate-based transformations. The project also handles image distortions caused by folding and generates a warped image based on calculated coefficients.

### 📚 문제 & 결과 (Problem & Result)

위에서 촬영한 자신의 흑백 사진을 A4 크기로 인쇄하고 이를 다시 읽어들여 **그레이 스케일**로 변환한 후, **800x600** 크기로 축소하세요.
`ginput(30)`을 사용하여 이미지에서 **30개의 좌표**를 클릭한 후, 각각의 좌표를 반올림하세요.

Print a grayscale image of yourself, captured from above, filling an A4 sheet. Then, read it back in grayscale and resize it to **800x600**.
Use `ginput(30)` to click on **30 points** in the image and round each of the selected coordinates.

![1](https://github.com/user-attachments/assets/4b85a59a-1c63-4e98-8653-ac7d326570ce)

A4 이미지를 세 번 접었다가 펼친 후, 동일한 방법으로 사진을 촬영하고 읽어들여 **800x600** 크기로 축소하세요.
위와 동일한 방식으로 좌표를 클릭하고 반올림하세요.

Fold the A4 image three times and unfold it. Capture the image again, read it in grayscale, and resize it to **800x600**.
Repeat the steps from **Problem 4-4 (2)** to click on **30 points** and round the coordinates.


![2](https://github.com/user-attachments/assets/291e1193-c470-47f5-b6e7-3ae515aa42f3)

현재 4개의 모서리를 갖는 5x4의 직사각형으로 구성된 영상 **I1**, **I2**가 있습니다. 이 두 영상이 세 번 접는 과정에서 왜곡되었습니다. **a0 ~ a3**, **b0 ~ b3** 계수를 추정하여 변형된 영상 **I3**을 생성하고 출력하세요.

Given two 5x4 rectangular images **I1** and **I2**, calculate the distortion caused by folding the image three times. Estimate the coefficients **a0 ~ a3** and **b0 ~ b3**, then generate a warped image **I3** based on these coefficients.

![3](https://github.com/user-attachments/assets/6eb6ec83-2779-4cf7-bd23-f4c609296e52)

최근점 이웃 보간법으로 보간

![4](https://github.com/user-attachments/assets/b2c965e5-dc0d-4e75-a5ea-2ccf184a2b5a)


---


# 2024_py


### project1.py

Rogistic Regression 

1. 데이터 시각화
2. y = a*X + b
3. z = a*x1 + b*x2 + c

-----

### project2_2.py

### Rogistic Regression (Z = a*feat1 + b*feat2 + c*feat3 + d)

로지스틱 회귀 모델을 사용하여 테스트 데이터에 대한 최적의 매개변수(a, b, c, d)를 찾고, 모델의 손실과 정확도를 출력합니다. 혼동 행렬을 시각화하고, 정확도, 정밀도, 재현율을 계산하여 출력하며, 과적합을 방지합니다.

주요기능
* 로지스틱 회귀 모델 구현
* 최적 매개변수(a, b, c, d) 찾기
* 모델 손실 및 정확도 출력
* 혼동 행렬
* 정확도, 정밀도, 재현율 계산
* 과적합 방지

![1](https://github.com/user-attachments/assets/4401f0be-e999-4809-a088-12065d5cc1a6)
![2](https://github.com/user-attachments/assets/80259c99-94d9-4618-88b4-5b7d08a3254c)

-----


### project2_3.py

### Decision Tree

의사결정 트리 분류기를 구현하여 데이터를 훈련하고, 트리를 구축하며, 새로운 데이터를 분류합니다. 혼동 행렬을 통해 모델 성능을 평가하고, 과적합 방지 방법을 적용합니다.

주요기능
* 의사결정 트리 분류기 구현
* 데이터 훈련 및 의사결정 트리 구축
* 새로운 데이터 분류
* 혼동 행렬 생성 및 정확도, 정밀도, 재현율 계산
* 과적합 방지

![3](https://github.com/user-attachments/assets/6b5df93d-3ae4-46e0-beab-fc6f9dde70f9)
![5](https://github.com/user-attachments/assets/0250908c-d575-456b-b7f2-eaddc617326e)

![4](https://github.com/user-attachments/assets/a0205264-2ae5-4b1f-8d00-36ff61a35f84)


----

### project3.py

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
  

![1](https://github.com/user-attachments/assets/07353e6f-54d1-40e5-87e4-37fdfa26407b)
![4](https://github.com/user-attachments/assets/f9cc5ab1-d5bc-4631-bb28-b01d2f208e0f)

![2](https://github.com/user-attachments/assets/c26ad08e-3403-4a46-bb8f-92a19ef7a55e)
![5](https://github.com/user-attachments/assets/3ff0ba3e-553d-41d1-a84e-f4e6fca200ee)

![3](https://github.com/user-attachments/assets/0f4a838b-7b76-4b31-a7b3-4bd3b6a71508)
![6](https://github.com/user-attachments/assets/460c836e-1f25-4f51-8a8d-0e2ddaa1d0e3)
