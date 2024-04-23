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
learning_rate = 0.08


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

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_a, best_b, best_c, best_d = a, b, c, d
print(f"Best Accuracy: {best_accuracy:.4f}")
print(
    f"Best Parameters (a, b, c, d): {best_a:.4f}, {best_b:.4f}, {best_c:.4f}, {best_d:.4f}")

# 3d 시각화


def plot_3d_scatter(features, labels, ax):
    unique_labels = np.unique(labels)
    colors = ['r', 'b']
    for label, color in zip(unique_labels, colors):
        indices_to_keep = labels == label
        if label == 1:
            color = 'b'
        ax.scatter(features[indices_to_keep, 0], features[indices_to_keep, 1],
                   features[indices_to_keep, 2], c=color, s=15, marker='*')


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
                z_boundary[:, :, 0], alpha=0.5)
plt.show()
# acc
xi_values = np.linspace(0.1, 0.9, 9)
accuracies = []
for xi in xi_values:
    z = a * feat1 + b * feat2 + c * feat3 + d
    predictions = sigmoid(z) > xi
    accuracy = np.mean(predictions == label)
    accuracies.append(accuracy)

plt.plot(xi_values, accuracies, marker='*')
plt.xlabel('Threshold (ξ)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

for xi, acc in zip(xi_values, accuracies):
    print(f"ξ = {xi:.2f}, Accuracy = {acc:.4f}")
