import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('2024_py/머신러닝/fashion_mnist/x_train.npy')
y_train = np.load('2024_py/머신러닝/fashion_mnist/y_train.npy')
x_test = np.load('2024_py/머신러닝/fashion_mnist/x_test.npy')
y_test = np.load('2024_py/머신러닝/fashion_mnist/y_test.npy')

x_train = x_train/255.0
x_test = x_test/255.0
train_unique_labels = np.unique(y_train)
test_unique_labels = np.unique(y_test)
train_label_indices = [np.where(y_train == i)[0][:1000]
                       for i in train_unique_labels]
x_train_data = np.concatenate([x_train[idx] for idx in train_label_indices])
y_train_data = np.concatenate([y_train[idx] for idx in train_label_indices])

valid_label_indices = [np.where(y_train == i)[0][1000:2000]
                       for i in train_unique_labels]
x_valid_data = np.concatenate([x_train[idx] for idx in valid_label_indices])
y_valid_data = np.concatenate([y_train[idx] for idx in valid_label_indices])

test_label_indices = [np.where(y_test == i)[0][0:200]
                      for i in test_unique_labels]
x_test_data = np.concatenate([x_test[idx] for idx in test_label_indices])
y_test_data = np.concatenate([y_test[idx] for idx in test_label_indices])

# To make 2d array
x_train_flatten = x_train_data.reshape(-1, 28*28)
x_valid_flatten = x_valid_data.reshape(-1, 28*28)
x_test_flatten = x_test_data.reshape(-1, 28*28)

label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
         'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_acc(model, x_val, y_val):
    return model.score(x_val, y_val)


def get_confusion_matrix(model, x_val, y_val):
    y_pred = model.predict(x_val)
    cm = confusion_matrix(y_pred, y_val)
    return cm


def get_metric(model, x_val, y_val, test=False):
    acc = get_acc(model, x_val, y_val)
    cm = get_confusion_matrix(model, x_val, y_val)
    if test != True:
        print('Validation Accuracy:', acc)
    else:
        print('Test Accuracy:', acc)
        return acc, cm
    sns.heatmap(cm,
                annot=True,
                xticklabels=label,
                yticklabels=label,
                fmt='d',
                cmap='Blues', )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return acc, cm


# Raw data (Logistic Regression)
# lr_raw = LR(max_iter=1000, random_state=42)
# lr_raw.fit(x_train_flatten, y_train_data)
# lr_raw_acc, lr_raw_cm = get_metric(lr_raw, x_valid_flatten, y_valid_data)

# Raw data (DecisionTree)
# dt_raw = DT(random_state=42)
# dt_raw.fit(x_train_flatten, y_train_data)
# dt_raw_acc, dt_raw_cm = get_metric(dt_raw, x_valid_flatten, y_valid_data)

# Raw data (MLP)
# mlp_raw = MLP(random_state=42, max_iter=1000)
# mlp_raw.fit(x_train_flatten, y_train_data)
# mlp_raw_acc, mlp_raw_cm = get_metric(mlp_raw, x_valid_flatten, y_valid_data)


# ## Raw Data Test Accuracy
# print('LR Raw Test Accuracy')
# lr_raw_acc, lr_raw_cm = get_metric(
#     lr_raw, x_test_flatten, y_test_data, test=True)
# print('DT Raw Test Accuracy')
# dt_raw_acc, dt_raw_cm = get_metric(
#     dt_raw, x_test_flatten, y_test_data, test=True)
# print('MLP Raw Test Accuracy')
# mlp_raw_acc, mlp_raw_cm = get_metric(
#     mlp_raw, x_test_flatten, y_test_data, test=True)
# print('')


# ## Feature Extraction
def pixel_average(img):   # 전체 pixel 비율
    return np.mean(img)


def left_right_symmetry(img):  # 좌우 대칭성
    left_half = img[:, :14]
    right_half = img[:, 14:]
    lr_symmetry = np.abs(left_half - right_half[:, ::-1]).mean()
    return lr_symmetry


def upper_bottom_symmetry(img):  # 위아래 대칭성
    upper_half = img[:14, :]
    bottom_half = img[14:, :]
    ub_symmetry = np.abs(upper_half - bottom_half[:, ::-1]).mean()
    return ub_symmetry


def bottom_zero_ratio(img):  # 아래쪽 0 비율
    bottom_quarter = img[-7:, :]
    bottom_zero = np.mean(bottom_quarter == 0)
    return bottom_zero


def upper_zero_ratio(img):  # 위쪽 0 비율
    upper_quarter = img[:7, :]
    upper_zero = np.mean(upper_quarter == 0)
    return upper_zero


def left_zero_ratio(img):    # 왼쪽 0 비율
    left_quarter = img[:, :7]
    left_zero = np.mean(left_quarter == 0)
    return left_zero


def right_zero_ratio(img):    # 오른쪽 0 비율
    right_quarter = img[:, -7:]
    right_zero = np.mean(right_quarter == 0)
    return right_zero


def vertical_change(img):    # 세로 방향 변화량
    return np.abs(img[1:] - img[:-1]).mean()


def horizontal_change(img):   # 가로 방향 변화량
    return np.abs(img[:, 1:] - img[:, :-1]).mean()


def vertical_first_nonzero_pixel_5(img, vertical_pixel=5):  # 첫 번째로 0이 안 나오는 픽셀
    first_nonzero_pixel = 1
    for i, pixel in enumerate(img[:, vertical_pixel]):
        if pixel != 0:
            first_nonzero_pixel = i / 28
            break
    return first_nonzero_pixel


# 첫 번째로 0이 안 나오는 픽셀
def vertical_first_nonzero_pixel_minus5(img, vertical_pixel=-5):
    first_nonzero_pixel = 1
    for i, pixel in enumerate(img[:, vertical_pixel]):
        if pixel != 0:
            first_nonzero_pixel = i / 28
            break
    return first_nonzero_pixel


def center_upper_avg_pixel(img):    # 중앙 위 pixel 평균
    mid_upper = np.mean(img[:5, 12:16])
    return mid_upper


def center_bottom_avg_pixel(img):   # 중앙 아래 pixel 평균
    mid_bottom = np.mean(img[-5:, 12:16])
    return mid_bottom


def corner_upper_avg_pixel(img):    # 모서리 위 pixel 평균
    corner_upper = (np.mean(img[:5, :5]) + np.mean(img[:5, -5:]))/2
    return corner_upper


def corner_bottom_avg_pixel(img):   # 모서리 아래 pixel 평균
    corner_bottom = (np.mean(img[-5:, -5:]) + np.mean(img[-5:, :5]))/2
    return corner_bottom


def pr1(img):
    pr1 = np.mean(img[:8, :8])/np.mean(img)
    return pr1


def pr2(img):
    pr2 = np.mean(img[:8, 21:28])/np.mean(img)
    return pr2


def pr3(img):
    pr3 = np.mean(img[-8:, :8])/np.mean(img)
    return pr3


def pr4(img):
    pr4 = np.mean(img[-8:, 21:28])/np.mean(img)
    return pr4


def pr5(img):
    pr5 = np.mean(img[9:21, :8])/np.mean(img)
    return pr5


def pr6(img):
    pr6 = np.mean(img[9:21, 21:28])/np.mean(img)
    return pr6


def pr7(img):
    pr7 = np.mean(img[9:21, 9:21])/np.mean(img)
    return pr7


def extract_features(*feature_function):
    train_features = []
    for img in x_train_data:
        train_features.append([func(img) for func in feature_function])
    valid_features = []
    for img in x_valid_data:
        valid_features.append([func(img) for func in feature_function])
    test_features = []
    for img in x_test_data:
        test_features.append([func(img) for func in feature_function])
    return np.array(train_features), np.array(valid_features), np.array(test_features)


# ## Feature Selection Procedure

# feature_functions = [
#     pixel_average
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# # feature data (Logistic Regression)
# lr_temp1 = LR(max_iter=10000, random_state=42)
# lr_temp1.fit(X_train_features, y_train_data)
# lr_acc1, lr_cm1 = get_metric(lr_temp1, X_valid_features, y_valid_data)

# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# # feature data (Logistic Regression)
# lr_temp2 = LR(max_iter=10000, random_state=42)
# lr_temp2.fit(X_train_features, y_train_data)
# lr_acc2, lr_cm2 = get_metric(lr_temp2, X_valid_features, y_valid_data)


# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
#     bottom_zero_ratio,
#     upper_zero_ratio,
#     left_zero_ratio,
#     right_zero_ratio,
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# # feature data (Logistic Regression)
# lr_temp3 = LR(max_iter=10000, random_state=42)
# lr_temp3.fit(X_train_features, y_train_data)
# lr_acc3, lr_cm3 = get_metric(lr_temp3, X_valid_features, y_valid_data)

# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
#     bottom_zero_ratio,
#     upper_zero_ratio,
#     left_zero_ratio,
#     right_zero_ratio,
#     vertical_change,
#     horizontal_change,
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# # feature data (Logistic Regression)
# lr_temp4 = LR(max_iter=10000, random_state=42)
# lr_temp4.fit(X_train_features, y_train_data)
# lr_acc4, lr_cm4 = get_metric(lr_temp4, X_valid_features, y_valid_data)

# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
#     bottom_zero_ratio,
#     upper_zero_ratio,
#     left_zero_ratio,
#     right_zero_ratio,
#     vertical_change,
#     horizontal_change,
#     corner_upper_avg_pixel,
#     corner_bottom_avg_pixel
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# # feature data (Logistic Regression)
# lr_temp5 = LR(max_iter=10000, random_state=42)
# lr_temp5.fit(X_train_features, y_train_data)
# lr_acc5, lr_cm5 = get_metric(lr_temp5, X_valid_features, y_valid_data)

# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
#     bottom_zero_ratio,
#     upper_zero_ratio,
#     left_zero_ratio,
#     right_zero_ratio,
#     vertical_change,
#     horizontal_change,
#     corner_upper_avg_pixel,
#     corner_bottom_avg_pixel,
#     center_upper_avg_pixel,
#     center_bottom_avg_pixel,
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)
# # feature data (Logistic Regression)
# lr_temp6 = LR(max_iter=10000, random_state=42)
# lr_temp6.fit(X_train_features, y_train_data)
# lr_acc6, lr_cm6 = get_metric(lr_temp6, X_valid_features, y_valid_data)

# feature_functions = [
#     pixel_average,
#     left_right_symmetry,
#     upper_bottom_symmetry,
#     bottom_zero_ratio,
#     upper_zero_ratio,
#     left_zero_ratio,
#     right_zero_ratio,
#     vertical_change,
#     horizontal_change,
#     corner_upper_avg_pixel,
#     corner_bottom_avg_pixel,
#     center_upper_avg_pixel,
#     center_bottom_avg_pixel,
#     vertical_first_nonzero_pixel_5,
#     vertical_first_nonzero_pixel_minus5, pr1, pr2, pr3, pr4, pr5, pr7
# ]

# X_train_features, X_valid_features, X_test_features = extract_features(
#     *feature_functions)

# feature data (Logistic Regression)
# lr_temp7 = LR(max_iter=10000, random_state=42)
# lr_temp7.fit(X_train_features, y_train_data)
# lr_acc7, lr_cm7 = get_metric(lr_temp7, X_valid_features, y_valid_data)

feature_functions = [
    pixel_average,
    left_right_symmetry, upper_bottom_symmetry,
    bottom_zero_ratio, upper_zero_ratio,
    left_zero_ratio, right_zero_ratio,
    vertical_change, horizontal_change,
    corner_upper_avg_pixel, corner_bottom_avg_pixel,
    center_upper_avg_pixel, center_bottom_avg_pixel,
    vertical_first_nonzero_pixel_5, vertical_first_nonzero_pixel_minus5,
    pr1, pr2, pr3, pr4, pr5, pr7
]

X_train_features, X_valid_features, X_test_features = extract_features(
    *feature_functions)

# Logistic Regression
# lr_train_dict = {}
# lr_valid_dict = {}
# for c in [0.001, 0.01, 0.1, 1.0, 10, 100, 300, 500]:
#     lr = LR(max_iter=10000, random_state=42, C=c)
#     lr.fit(X_train_features, y_train_data)
#     lr_train_acc, lr_train_cm = get_metric(
#         lr, X_train_features, y_train_data)
#     lr_train_dict[c] = lr_train_acc
#     lr_valid_acc, lr_valid_cm = get_metric(
#         lr, X_valid_features, y_valid_data)
#     lr_valid_dict[c] = lr_valid_acc

# plt.plot(list(map(lambda x: str(x), lr_train_dict.keys())),
#          lr_train_dict.values(), marker='*', label='train')
# plt.plot(list(map(lambda x: str(x), lr_valid_dict.keys())),
#          lr_valid_dict.values(), marker='*', label='valid')
# plt.title('Logistic Regression')
# plt.xlabel('C')
# plt.ylabel('Accuracy')
# plt.axvline(x='10', color='red', linestyle='--')
# plt.legend()
# plt.show()


# DecisionTree
# dt_train_dict = {}
# dt_valid_dict = {}
# for max_depth in [3, 5, 7, 10, 15, 20, 30]:
#     dt = DT(random_state=42, max_depth=max_depth)
#     dt.fit(X_train_features, y_train_data)
#     dt_train_acc, dt_train_cm = get_metric(
#         dt, X_train_features, y_train_data)
#     dt_train_dict[max_depth] = dt_train_acc
#     dt_valid_acc, dt_valid_cm = get_metric(
#         dt, X_valid_features, y_valid_data)
#     dt_valid_dict[max_depth] = dt_valid_acc

# plt.plot(list(map(lambda x: str(x), dt_train_dict.keys())),
#          dt_train_dict.values(), marker='*', label='train')
# plt.plot(list(map(lambda x: str(x), dt_valid_dict.keys())),
#          dt_valid_dict.values(), marker='*', label='valid')
# plt.title('Decision Tree')
# plt.xlabel('Max Depth')
# plt.ylabel('Accuracy')
# plt.axvline(x='10', color='red', linestyle='--')
# plt.legend()
# plt.show()


# MLP
# mlp_train_dict = {}
# mlp_valid_dict = {}
# for hidden_layer_size in [10, 30, 50, 100, 1000, 1200, 1400, 1600, 1800, 2000]:
#     mlp = MLP(random_state=42, max_iter=1000,
#               hidden_layer_sizes=hidden_layer_size)
#     mlp.fit(X_train_features, y_train_data)
#     mlp_train_acc, mlp_train_cm = get_metric(
#         mlp, X_train_features, y_train_data)
#     mlp_train_dict[hidden_layer_size] = mlp_train_acc
#     mlp_valid_acc, mlp_valid_cm = get_metric(
#         mlp, X_valid_features, y_valid_data)
#     mlp_valid_dict[hidden_layer_size] = mlp_valid_acc


# plt.plot(list(map(lambda x: str(x), mlp_train_dict.keys())),
#          mlp_train_dict.values(), marker='*', label='train')
# plt.plot(list(map(lambda x: str(x), mlp_valid_dict.keys())),
#          mlp_valid_dict.values(), marker='*', label='valid')
# plt.title('MLP')
# plt.xlabel('Hidden Layer Size')
# plt.ylabel('Accuracy')
# plt.axvline(x='1200', color='red', linestyle='--')
# plt.legend()
# plt.show()


# Best Param 이용해 test
# lr = LR(max_iter=10000, random_state=42, C=10)
# lr.fit(X_train_features, y_train_data)
# lr_feature_acc, lr_feature_cm = get_metric(
#     lr, X_test_features, y_test_data, test=True)
# dt = DT(random_state=42, max_depth=10)
# dt.fit(X_train_features, y_train_data)
# dt_feature_acc, dt_feature_cm = get_metric(
#     dt, X_test_features, y_test_data, test=True)
# mlp = MLP(random_state=42, max_iter=1000, hidden_layer_sizes=1200)
# mlp.fit(X_train_features, y_train_data)
# mlp_feature_acc, mlp_feature_cm = get_metric(
#     mlp, X_test_features, y_test_data, test=True)
# print('')


# def plot_confusion_matrices(cm1, cm2, labels):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

#     sns.heatmap(cm1, annot=True, xticklabels=labels, yticklabels=labels,
#                 fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_xlabel('Predicted')
#     ax1.set_ylabel('True')
#     ax1.set_title('Raw Data')

#     sns.heatmap(cm2, annot=True, xticklabels=labels, yticklabels=labels,
#                 fmt='d', cmap='Blues', ax=ax2)
#     ax2.set_xlabel('Predicted')
#     ax2.set_ylabel('True')
#     ax2.set_title('Feature Extracted Data')

#     sns.heatmap(cm2 - cm1, annot=True, xticklabels=labels, yticklabels=labels,
#                 fmt='d', cmap='Blues', ax=ax3)
#     ax3.set_xlabel('Predicted')
#     ax3.set_ylabel('True')
#     ax3.set_title('Feature - Raw')

#     plt.tight_layout()
#     plt.show()


# Logistic Regression
# plot_confusion_matrices(lr_raw_cm, lr_feature_cm, label)

# Decision Tree
# plot_confusion_matrices(dt_raw_cm, dt_feature_cm, label)

# MLP
# plot_confusion_matrices(mlp_raw_cm, mlp_feature_cm, label)
