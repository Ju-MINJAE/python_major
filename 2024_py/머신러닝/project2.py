import numpy as np
import matplotlib.pyplot as plt

########### 1 ###########
# 주어진 train.txt, valid.txt, test.txt를 읽어 들여 위와 같이 출력하라.
#########################
# train_data = np.loadtxt(
#     '/Users/juminjae/Desktop/python/2024_py/머신러닝/train.txt')
# test_data = np.loadtxt('/Users/juminjae/Desktop/python/2024_py/머신러닝/test.txt')
# valid_data = np.loadtxt(
#     '/Users/juminjae/Desktop/python/2024_py/머신러닝/valid.txt')

# train_features, train_label = train_data[:, :3], train_data[:, 3]
# test_features, test_label = test_data[:, :3], test_data[:, 3]
# valid_features, valid_label = valid_data[:, :3], valid_data[:, 3]
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')


# def plot_3d_scatter(features, labels, ax):
#     unique_labels = np.unique(labels)
#     colors = ['r', 'b']
#     for label, color in zip(unique_labels, colors):
#         indices_to_keep = labels == label
#         if label == 1:
#             color = 'b'
#         ax.scatter(features[indices_to_keep, 0], features[indices_to_keep, 1],
#                    features[indices_to_keep, 2], c=color, s=15, marker='*')


# plot_3d_scatter(train_features, train_label, ax)
# plot_3d_scatter(test_features, test_label, ax)
# plot_3d_scatter(valid_features, valid_label, ax)
# ax.set_ylim(-150, 100)
# ax.set_zlim(-75, 125)
# plt.show()

########### 2-1 ###########
# train.txt를 사용하여 로지스틱 회귀 모델을 도입, 두 클래스를 분류하는 결정경계를 찾아보자.
# z=ax1+bx2+cx3+d 로 모델을 설정하고 위 데이터를 사용해 가장 오류를 적게 만드는 a, b, c, d, ξ를
# 찾는 코드를 작성하라. a, b, c, d, ξ는 임의로 설정 후 시작하고 Loss는 log Loss를 사용하되 epoch
# 단위로 변수를 갱신하자. ξ는 임의로 설정한 최소값에서 시작하여 최대값까지 값을 증가시켜 가며
# 최적점을 찾고 최종적으로 찾은 acc를 표로 기록하라.
#########################
train_data = np.loadtxt(
    '/Users/juminjae/Desktop/python/2024_py/머신러닝/train.txt')
feat1, feat2, feat3, label = train_data[:,
                                        0], train_data[:, 1], train_data[:, 2], train_data[:, 3]
n = len(train_data)
a, b, c, d = 0.3, 0.1, 0.1, 1
learning_rate = 0.1


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_log_loss(a, b, c, d):
    z = a * feat1 + b * feat2 + c * feat3 + d
    hz = sigmoid(z)
    log_loss = -(label * np.log(hz + 1e-15) +
                 (1 - label) * np.log(1 - hz + 1e-15))
    return np.mean(log_loss)


def compute_gradients(a, b, c, d):
    z = a * feat1 + b * feat2 + c * feat3 + d
    hz = sigmoid(z)
    error = hz - label
    grad_a = np.dot(error, feat1) / n
    grad_b = np.dot(error, feat2) / n
    grad_c = np.dot(error, feat3) / n
    grad_d = np.mean(error)
    return grad_a, grad_b, grad_c, grad_d


epochs = 5000
best_accuracy = -1
for epoch in range(epochs):
    grad_a, grad_b, grad_c, grad_d = compute_gradients(a, b, c, d)
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    z = a * feat1 + b * feat2 + c * feat3 + d
    predictions = sigmoid(z) > 0.5
    accuracy = np.mean(predictions == label)
    print(compute_log_loss(a, b, c, d))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_a, best_b, best_c, best_d = a, b, c, d
loss = compute_log_loss(best_a, best_b, best_c, best_d)
accuracy = 1 - loss
print(
    f"Best Parameters (a, b, c, d): {best_a:.4f}, {best_b:.4f}, {best_c:.4f}, {best_d:.4f}")
print(f"Log Loss : {loss:.4f}, Best Accuracy: {best_accuracy:.4f}")
print(f"Accuracy: {accuracy:.4f}")
# 3d 시각화


def plot_3d_scatter(features, labels, ax):
    unique_labels = np.unique(labels)
    colors = ['r', 'b']
    for label, color in zip(unique_labels, colors):
        index = labels == label
        if label == 1:
            color = 'b'
        ax.scatter(features[index, 0], features[index, 1],
                   features[index, 2], c=color, s=15, marker='*')


non_zero_indices = np.where((feat1 != 0) | (feat2 != 0) | (feat3 != 0))
feat1_filtered = feat1[non_zero_indices]
feat2_filtered = feat2[non_zero_indices]
feat3_filtered = feat3[non_zero_indices]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
plot_3d_scatter(np.vstack((feat1_filtered, feat2_filtered,
                feat3_filtered)).T, label[non_zero_indices], ax)
x_vals = np.linspace(min(feat1_filtered), max(feat1_filtered), 100)
y_vals = np.linspace(min(feat2_filtered), max(feat2_filtered), 100)
z_vals = np.linspace(min(feat3_filtered), max(feat3_filtered), 100)
x_mesh, y_mesh, z_mesh = np.meshgrid(x_vals, y_vals, z_vals)
z_boundary = -(best_a * x_mesh + best_b * y_mesh + best_d)/best_c
ax.plot_surface(x_mesh[:, :, 0], y_mesh[:, :, 0],
                z_boundary[:, :, 0], color='green', alpha=0.5)
plt.show()


########### 2-2 ###########
# 위에서 찾은 최적의 파라미터로 학습된 인식기에 test.txt를 대입하여 성능을 검증하라.
# 이때 결과를 보여주기 위해 해당 파라미터와, confusion matrix, acc, precision, recall을 각각 표로 제시하라.
#########################
# test_data = np.loadtxt(
#     '/Users/juminjae/Desktop/python/2024_py/머신러닝/test.txt')
# feat1, feat2, feat3, label = test_data[:,
#                                        0], test_data[:, 1], test_data[:, 2], test_data[:, 3]
# n = len(test_data)
# a, b, c, d = 0.3, 0.1, 0.1, 1
# learning_rate = 0.1


# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# def compute_log_loss(a, b, c, d):
#     z = a * feat1 + b * feat2 + c * feat3 + d
#     hz = sigmoid(z)
#     log_loss = -(label * np.log(hz + 1e-15) +
#                  (1 - label) * np.log(1 - hz + 1e-15))
#     return np.mean(log_loss)


# def compute_gradients(a, b, c, d):
#     z = a * feat1 + b * feat2 + c * feat3 + d
#     hz = sigmoid(z)
#     error = hz - label
#     grad_a = np.dot(error, feat1) / n
#     grad_b = np.dot(error, feat2) / n
#     grad_c = np.dot(error, feat3) / n
#     grad_d = np.mean(error)
#     return grad_a, grad_b, grad_c, grad_d


# epochs = 10000
# best_accuracy = -1
# for epoch in range(epochs):
#     grad_a, grad_b, grad_c, grad_d = compute_gradients(a, b, c, d)
#     a -= learning_rate * grad_a
#     b -= learning_rate * grad_b
#     c -= learning_rate * grad_c
#     d -= learning_rate * grad_d

#     z = a * feat1 + b * feat2 + c * feat3 + d
#     predictions = sigmoid(z) > 0.5
#     accuracy = np.mean(predictions == label)

#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_a, best_b, best_c, best_d = a, b, c, d
# loss = compute_log_loss(best_a, best_b, best_c, best_d)
# print(
#     f"Best Parameters (a, b, c, d): {best_a:.4f}, {best_b:.4f}, {best_c:.4f}, {best_d:.4f}")
# print(f"Loss : {loss:.4f}, Best Accuracy: {best_accuracy: .4f}")

# # 3d 시각화


# def plot_3d_scatter(features, labels, ax):
#     unique_labels = np.unique(labels)
#     colors = ['r', 'b']
#     for label, color in zip(unique_labels, colors):
#         index = labels == label
#         if label == 1:
#             color = 'b'
#         ax.scatter(features[index, 0], features[index, 1],
#                    features[index, 2], c=color, s=15, marker='*')


# non_zero_indices = np.where((feat1 != 0) | (feat2 != 0) | (feat3 != 0))
# feat1_filtered = feat1[non_zero_indices]
# feat2_filtered = feat2[non_zero_indices]
# feat3_filtered = feat3[non_zero_indices]

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# plot_3d_scatter(np.vstack((feat1_filtered, feat2_filtered,
#                 feat3_filtered)).T, label[non_zero_indices], ax)
# x_vals = np.linspace(min(feat1_filtered), max(feat1_filtered), 200)
# y_vals = np.linspace(min(feat2_filtered), max(feat2_filtered), 200)
# z_vals = np.linspace(min(feat3_filtered), max(feat3_filtered), 200)
# x_mesh, y_mesh, z_mesh = np.meshgrid(x_vals, y_vals, z_vals)
# z_boundary = -(best_a * x_mesh + best_b * y_mesh + best_d)/best_c
# ax.plot_surface(x_mesh[:, :, 0], y_mesh[:, :, 0],
#                 z_boundary[:, :, 0], alpha=0.5, color='green')
# plt.show()

#
# z_test = a * feat1 + b * feat2 + c * feat3 + d
# predictions_test = sigmoid(z_test) > 0.5
# TP = np.sum((predictions_test == True) & (label == 1))
# FN = np.sum((predictions_test == False) & (label == 1))
# FP = np.sum((predictions_test == True) & (label == 0))
# TN = np.sum((predictions_test == False) & (label == 0))
# print(TP, FN, FP, TN)
# confusion_matrix = np.array([[TP, FN], [FP, TN]])
# acc = (TP+TN)/(TP+TN+FP+FN)
# precision = TP / (TP+FP)
# recall = TP / (TP+FN)
#
# plt.figure(figsize=(8, 6))
# plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.xticks([0, 1], ['Predicted Negative', 'Predicted Positive'])
# plt.yticks([0, 1], ['Actual Negative', 'Actual Positive'])

# # Add text annotations
# for i in range(2):
#     for j in range(2):
#         plt.text(j, i, str(
#             confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center', color='black')

# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.show()

# # Print accuracy, precision, and recall
# print(f"Accuracy: {acc:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")

########### 2-3 ###########
# 1)의 학습 과정을 다시 진행하되 파라미터 갱신 때마다 valid.txt를 대입하여 acc를 계산하여 과도적합 여부를 점검하라.
# 아래의 그림처럼 train.txt와 valid.txt의 오차를 비교하여 2 세트에서 동시에 최저가 되는 parameter를 확인하라.
#########################
