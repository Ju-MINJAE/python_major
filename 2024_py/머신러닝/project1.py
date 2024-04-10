import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv(
    '/Users/juminjae/Desktop/python/2024_py/머신러닝/data.tsv', delimiter='\t')
# print(data.head())

######### 1 ##########
# plt.scatter(data[data['label'] == 0]['feat1'],
#             data[data['label'] == 0]['feat2'], c='blue', label='Label 0', s=10)

# plt.scatter(data[data['label'] == 1]['feat1'],
#             data[data['label'] == 1]['feat2'], c='red', label='Label 1', s=10)
# plt.xlim(-100, 100)
# plt.xlabel('Feat1')
# plt.ylabel('Feat2')
# plt.legend()
# plt.show()

########## 2 ###########
# 선형모델을 도입하여 두 클래스를 분류하는 결정경계를 찾아보자. y = ax+b 로 모델을 설정하고
# 위 데이터를 사용해 가장 오류를 적게 만드는 a, b를 찾는 코드를 작성하라. a, b, ξ는
# 임의로 설정하고 Loss는 1-(맞춘 개수/ 200)을 사용하라.

feat1 = data['feat1'].values
feat2 = data['feat2'].values
label = data['label'].values
n = len(data)
a = 0.5
b = 1
learning_rate = 0.0005
min_loss = float('inf')
best_a = a
best_b = b
best_correct_count = 0

epoch = 0
while epoch < 20000:
    correct_count = 0
    for x, y, l in zip(feat1, feat2, label):
        predicted_y = a * x + b  # 선형모델 y=ax+b
        if (predicted_y > y and l == 1) or (predicted_y <= y and l == 0):
            correct_count += 1

    loss = 1 - (correct_count / n)
    if loss < min_loss:
        min_loss = loss
        best_a = a
        best_b = b
        best_correct_count = correct_count

    if correct_count > 180:
        break

    for x, y, l in zip(feat1, feat2, label):
        predicted_y = a * x + b
        if l == 1 and predicted_y <= y:
            a += learning_rate * (y - predicted_y) * x
            b += learning_rate * (y - predicted_y)
        elif l == 0 and predicted_y > y:
            a -= learning_rate * (predicted_y - y) * x
            b -= learning_rate * (predicted_y - y)
    if epoch % 100 == 0:
        print(f'a={a},b={b},loss={loss},correct_count={correct_count}')
    epoch += 1

print(
    f"Final: a = {best_a}, b = {best_b}, \nMin loss ={min_loss}, correct ={best_correct_count}")

plt.scatter(data[data['label'] == 0]['feat1'],
            data[data['label'] == 0]['feat2'], c='blue', s=5)
plt.scatter(data[data['label'] == 1]['feat1'],
            data[data['label'] == 1]['feat2'], c='red', s=5)
x_values = np.array(plt.xlim())
y_values = best_a * x_values + best_b
plt.plot(x_values, y_values, '-g')
plt.xlim(-100, 80)
plt.xlabel('feat1')
plt.ylabel('feat2')
plt.show()


############ 3 ##############
# 로지스틱 회귀 모델을 도입하여 두 클래스를 분류하는 결정경계를 찾아보자. z = ax1+bx2+c 로
# 모델을 설정하고 위 데이터를 사용해 가장 오류를 적게 만드는 a, b, c를 찾는 코드를 작성하라.
# a, b, c, ξ는 임의로 설정하고 Loss는 log Loss를 사용하라.

# feat1 = data['feat1'].values
# feat2 = data['feat2'].values
# label = data['label'].values
# n = len(data)
# a, b, c = 3, 2, 1
# learning_rate = 0.005
# epochs = 10000
# min_loss = float('inf')
# best_a, best_b, best_c = None, None, None
# epoch = 0


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# def compute_log_loss(a, b, c):
#     log_loss_sum = 0
#     for x1, x2, l in zip(feat1, feat2, label):
#         z = a * x1 + b * x2 + c  # Log odds
#         hz = sigmoid(z)
#         # log_loss_sum이 log0으로 출력되기 때문에 10^-15를 더해줌
#         log_loss = -(l * np.log(hz + 1e-15) + (1 - l) * np.log(1 - hz + 1e-15))
#         log_loss_sum += log_loss
#     return log_loss_sum / n


# while epoch < epochs:
#     gradient_a = 0
#     gradient_b = 0
#     gradient_c = 0

#     for x1, x2, l in zip(feat1, feat2, label):
#         z = a * x1 + b * x2 + c
#         hz = sigmoid(z)  # 예측값
#         gradient_a += (hz - l) * x1  # (hz-l)예측값-실제값:오차
#         gradient_b += (hz - l) * x2
#         gradient_c += hz - l

#     a -= learning_rate * gradient_a / n
#     b -= learning_rate * gradient_b / n
#     c -= learning_rate * gradient_c / n
#     total_loss = compute_log_loss(a, b, c)
#     if total_loss < min_loss:
#         min_loss = total_loss
#         best_a, best_b, best_c = a, b, c

#     if total_loss <= 0.2:
#         break
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}: Total Loss = {total_loss}")
#     epoch += 1

# print(
#     f"Best Parameters: a = {best_a}, b = {best_b} \nc = {best_c}, Min Loss = {min_loss}")

# plt.scatter(data[data['label'] == 0]['feat1'],
#             data[data['label'] == 0]['feat2'], c='blue', s=10)

# plt.scatter(data[data['label'] == 1]['feat1'],
#             data[data['label'] == 1]['feat2'], c='red', s=10)

# x_values = np.array(plt.xlim())
# y_values = (-best_a * x_values - best_c)/best_b
# plt.plot(x_values, y_values, '-g')
# plt.xlim(-70, 60)
# plt.ylim(-80, 80)
# plt.xlabel('feat1')
# plt.ylabel('feat2')
# plt.show()
