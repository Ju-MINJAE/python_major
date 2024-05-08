import numpy as np
import matplotlib.pyplot as plt

##### Decision Tree 시각화 #####


def decision_tree(train_data, threshold=0.98, count=0):
    count += 1
    feat1, feat2, feat3, label = train_data[:,
                                            0], train_data[:, 1], train_data[:, 2], train_data[:, 3]
    data_n = len(train_data)
    n = 10
    feat1_min, feat1_max = np.min(feat1), np.max(feat1)
    feat2_min, feat2_max = np.min(feat2), np.max(feat2)
    feat3_min, feat3_max = np.min(feat3), np.max(feat3)
    division1 = np.linspace(feat1_min, feat1_max, n+2)
    division2 = np.linspace(feat2_min, feat2_max, n+2)
    division3 = np.linspace(feat3_min, feat3_max, n+2)
    feat1_div = np.round(np.delete(division1, [0, len(division1)-1]))
    feat2_div = np.round(np.delete(division2, [0, len(division2)-1]))
    feat3_div = np.round(np.delete(division3, [0, len(division3)-1]))
    impurity = []

    for i in range(n):
        f1_left_indices = feat1 <= feat1_div[i]
        f1_right_indices = feat1 > feat1_div[i]
        f1_left_labels = label[f1_left_indices]
        f1_right_labels = label[f1_right_indices]
        f1_left_class_counts = np.bincount(f1_left_labels.astype(int))
        f1_right_class_counts = np.bincount(f1_right_labels.astype(int))
        f1_left_impurity = 1 - \
            np.max(f1_left_class_counts) / len(f1_left_labels)
        f1_right_impurity = 1 - \
            np.max(f1_right_class_counts) / len(f1_right_labels)
        f1_impurity = (f1_left_impurity + f1_right_impurity) / 2

        f2_left_indices = feat2 <= feat2_div[i]
        f2_right_indices = feat2 > feat2_div[i]
        f2_left_labels = label[f2_left_indices]
        f2_right_labels = label[f2_right_indices]
        f2_left_class_counts = np.bincount(f2_left_labels.astype(int))
        f2_right_class_counts = np.bincount(f2_right_labels.astype(int))
        f2_left_impurity = 1 - \
            np.max(f2_left_class_counts) / len(f2_left_labels)
        f2_right_impurity = 1 - \
            np.max(f2_right_class_counts) / len(f2_right_labels)
        f2_impurity = (f2_left_impurity + f2_right_impurity) / 2

        f3_left_indices = feat3 <= feat3_div[i]
        f3_right_indices = feat3 > feat3_div[i]
        f3_left_labels = label[f3_left_indices]
        f3_right_labels = label[f3_right_indices]
        f3_left_class_counts = np.bincount(f3_left_labels.astype(int))
        f3_right_class_counts = np.bincount(f3_right_labels.astype(int))
        f3_left_impurity = 1 - \
            np.max(f3_left_class_counts) / len(f3_left_labels)
        f3_right_impurity = 1 - \
            np.max(f3_right_class_counts) / len(f3_right_labels)
        f3_impurity = (f3_left_impurity + f3_right_impurity) / 2

        impurity.append([f1_impurity, f2_impurity, f3_impurity])

    split_feature, split_threshold = np.unravel_index(
        np.argmin(impurity), (n, 3))
    if split_threshold == 0:
        predicted_labels = (feat1 > feat1_div[split_feature]).astype(int)
    elif split_threshold == 1:
        predicted_labels = (feat2 > feat2_div[split_feature]).astype(int)
    else:
        predicted_labels = (feat3 > feat3_div[split_feature]).astype(int)

    ##################### acc 코드 수정 ######################
    accuracy = np.sum(predicted_labels == label)/data_n
    # accuracy = np.mean(predicted_labels == label)
    if (accuracy < 0.5):
        accuracy = 1-accuracy
    # print(accuracy)
    ########################################################
    if accuracy >= threshold:
        return count

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feat1, feat2, feat3, cmap='bwr', c=label, s=15)

    if split_threshold == 0:
        yy, zz = np.meshgrid([feat2_min, feat2_max], [feat3_min, feat3_max])
        xx = np.full_like(yy, feat1_div[split_feature])
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
    elif split_threshold == 1:
        xx, zz = np.meshgrid([feat1_min, feat1_max], [feat3_min, feat3_max])
        yy = np.full_like(xx, feat2_div[split_feature])
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)
    else:
        xx, yy = np.meshgrid([feat1_min, feat1_max], [feat2_min, feat2_max])
        zz = np.full_like(xx, feat3_div[split_feature])
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.3)

    ax.scatter(feat1, feat2, feat3, cmap='bwr', c=label, s=15)
    plt.show()

    if split_threshold == 0:
        left_data = train_data[train_data[:, 0] <= feat1_div[split_feature]]
        right_data = train_data[train_data[:, 0] > feat1_div[split_feature]]
    elif split_threshold == 1:
        left_data = train_data[train_data[:, 1] <= feat2_div[split_feature]]
        right_data = train_data[train_data[:, 1] > feat2_div[split_feature]]
    else:
        left_data = train_data[train_data[:, 2] <= feat3_div[split_feature]]
        right_data = train_data[train_data[:, 2] > feat3_div[split_feature]]

    left_impurity = calculate_impurity(left_data)
    right_impurity = calculate_impurity(right_data)

    if left_impurity >= right_impurity:
        return decision_tree(left_data, threshold, count=count)
    else:
        return decision_tree(right_data, threshold, count=count)


def calculate_impurity(data):
    labels = data[:, -1]
    label_count = np.bincount(labels.astype(int))
    class_probabilities = label_count / len(labels)
    impurity = 1 - np.max(class_probabilities)
    return impurity


train_data = np.loadtxt(
    '/Users/juminjae/Desktop/python/2024_py/머신러닝/train.txt')
count = decision_tree(train_data)
print(f"Depth: {count}")


############# 3-2, 3-3 ###############


# def data_split(data):
#     return data[:, :-1], data[:, -1].astype(int)


# def calculate_accuracy(y_true, y_pred):
#     return np.bincount(y_true == y_pred)[1] / len(y_true)


# def log_loss(y_true, y_pred):
#     # 1e-15 -> 0 , 1-1e-15 -> 1
#     y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
#     loss = -np.mean(y_true * np.log(y_pred) +
#                     (1 - y_true) * np.log(1 - y_pred))
#     acc = 1 - loss
#     return acc


# def stopping_criteria(y, depth, max_depth, min_samples_split):
#     # label의 고유값이 1개 이면 stop
#     if len(np.unique(y)) == 1:
#         return True
#     # max_depth 도달시 stop
#     if depth == max_depth:
#         return True
#     # node에 있는 data가 2개이면 stop
#     if len(y) < min_samples_split:
#         return True
#     return False


# def build_tree(X, y, depth, max_depth, min_samples_split, sample_weights):
#     # 재귀함수 탈출 검사
#     if stopping_criteria(y, depth, max_depth, min_samples_split):
#         last_data = calculate_last_data(y, sample_weights)
#         return last_data

#     feature_index, threshold = best_split(
#         X, y, sample_weights, min_samples_split)
#     if feature_index is None:
#         last_data = calculate_last_data(y, sample_weights)
#         return last_data

#     left_indices = X[:, feature_index] <= threshold
#     right_indices = X[:, feature_index] > threshold

#     X_left, X_right = X[left_indices], X[right_indices]
#     y_left, y_right = y[left_indices], y[right_indices]

#     weights_left = sample_weights[left_indices]
#     weights_right = sample_weights[right_indices]

#     left_subtree = build_tree(
#         X_left, y_left, depth + 1, max_depth, min_samples_split, weights_left)
#     right_subtree = build_tree(
#         X_right, y_right, depth + 1, max_depth, min_samples_split, weights_right)

#     return (feature_index, threshold, left_subtree, right_subtree)


# def calculate_last_data(y, sample_weights):
#     labels, counts = np.unique(y, return_counts=True)
#     weighted_counts = np.zeros_like(counts, dtype=np.float64)

#     for i, label in enumerate(labels):
#         weighted_counts[i] = sample_weights[y == label].sum()

#     return dict(zip(labels, weighted_counts))


# def best_split(X, y, sample_weights, min_samples_split):
#     best_div = 0
#     best_feature, best_threshold = None, None
#     current_impurity = calculate_impurity(y, sample_weights)

#     #  각 특징점의 고유값을 확인
#     for feature_index in range(X.shape[1]):
#         unique_values = np.unique(X[:, feature_index])

#         # 샘플의 수가 min_samples_split(2) 이상인 경우에만 분할 후보 고려
#         if len(unique_values) < min_samples_split:
#             continue
#         # 고유값을 기준으로 특징점 분할
#         for threshold in unique_values:
#             left_indices = X[:, feature_index] <= threshold
#             right_indices = X[:, feature_index] > threshold

#             div = calculate_div(
#                 y, sample_weights, left_indices, right_indices, current_impurity)
#             if div > best_div:
#                 best_div = div
#                 best_feature = feature_index
#                 best_threshold = threshold

#     return best_feature, best_threshold


# def calculate_impurity(y, sample_weights):
#     total_weight = np.sum(sample_weights)
#     _, counts = np.unique(y, return_counts=True)
#     weighted_counts = np.zeros_like(counts, dtype=np.float64)

#     for i, label in enumerate(np.unique(y)):
#         weighted_counts[i] = np.sum(sample_weights[y == label])

#     prob = weighted_counts / total_weight
#     impurity = 1 - np.sum(prob**2)

#     return impurity


# def calculate_div(y, sample_weights, left_indices, right_indices, current_impurity):
#     left_weight = np.sum(sample_weights[left_indices])
#     right_weight = np.sum(sample_weights[right_indices])
#     total_weight = left_weight + right_weight

#     weights_left = sample_weights[left_indices]
#     weights_right = sample_weights[right_indices]

#     left_impurity = calculate_impurity(y[left_indices], weights_left)
#     right_impurity = calculate_impurity(y[right_indices], weights_right)

#     weighted_impurity = (left_weight / total_weight) * left_impurity
#     weighted_impurity += (right_weight / total_weight) * right_impurity
#     # 현재 불순도에서 가중 불순도를 뺀 값, 이 값이 클수록 노드 분할 후 불순도 감소가 크다
#     new_impurity = current_impurity - weighted_impurity

#     return new_impurity


# def predict_single(node, x):
#     while isinstance(node, tuple):
#         feature_index, threshold, left_subtree, right_subtree = node
#         if x[feature_index] <= threshold:
#             node = left_subtree
#         else:
#             node = right_subtree

#     return max(node, key=node.get)


# def predict(root, X):
#     preds = [predict_single(root, x) for x in X]
#     return np.array(preds)


# def fit(X, y, max_depth=10,  min_samples_split=2, sample_weights=None):
#     if sample_weights is None:
#         sample_weights = np.ones(len(y))
#     sample_weights = np.array(sample_weights)

#     root = build_tree(X, y, 0, max_depth, min_samples_split, sample_weights)
#     return root


# def calculate_confusion_matrix(y_true, y_pred):
#     TP, TN, FP, FN = 0, 0, 0, 0

#     for true_label, pred_label in zip(y_true, y_pred):
#         if true_label == 1 and pred_label == 1:
#             TP += 1
#         elif true_label == 0 and pred_label == 0:
#             TN += 1
#         elif true_label == 0 and pred_label == 1:
#             FP += 1
#         elif true_label == 1 and pred_label == 0:
#             FN += 1

#     return TP, TN, FP, FN


# def calculate_metrics(TP, TN, FP, FN):
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     return accuracy, precision, recall


# train_set = np.loadtxt("2024_py/머신러닝/train.txt")
# valid_set = np.loadtxt("2024_py/머신러닝/valid.txt")
# test_set = np.loadtxt("2024_py/머신러닝/test.txt")

# X_train, y_train = data_split(data=train_set)
# X_test, y_test = data_split(data=test_set)
# X_valid, y_valid = data_split(data=valid_set)

# depth_range = range(1, 11)
# acc_train = []
# acc_valid = []
# acc_test = []
# confusion_matrices = []

# for depth in depth_range:
#     tree = fit(X_train, y_train, max_depth=depth)
#     y_train_pred = predict(tree, X_train)
#     y_valid_pred = predict(tree, X_valid)
#     y_test_pred = predict(tree, X_valid)
#     train_acc = log_loss(y_true=y_train, y_pred=predict(tree, X_train))
#     valid_acc = log_loss(y_true=y_valid, y_pred=predict(tree, X_valid))
#     test_acc = log_loss(y_true=y_test, y_pred=predict(tree, X_test))

#     acc_train.append(train_acc)
#     # print(f"{train_acc:.4f}")
#     acc_valid.append(valid_acc)
#     acc_test.append(test_acc)

#     TP_train, TN_train, FP_train, FN_train = calculate_confusion_matrix(
#         y_train, y_train_pred)
#     TP_valid, TN_valid, FP_valid, FN_valid = calculate_confusion_matrix(
#         y_valid, y_valid_pred)
#     TP_test, TN_test, FP_test, FN_test = calculate_confusion_matrix(
#         y_test, y_test_pred)

#     confusion_matrices.append({
#         'depth': depth,
#         'train': {'TP': TP_train, 'TN': TN_train, 'FP': FP_train, 'FN': FN_train},
#         'valid': {'TP': TP_valid, 'TN': TN_valid, 'FP': FP_valid, 'FN': FN_valid},
#         'test': {'TP': TP_test, 'TN': TN_test, 'FP': FP_test, 'FN': FN_test},
#     })

# plt.plot(list(depth_range), acc_train, label="train acc")
# plt.plot(list(depth_range), acc_valid, label="valid acc")
# plt.xlabel("Depths")
# plt.ylabel("Acc")
# plt.legend()
# plt.grid()
# max_valid_idx = np.argmax(acc_valid)
# plt.axvline(x=max_valid_idx + 2, color='r', linestyle='--')
# plt.tight_layout()
# plt.show()

# for confusion_matrix in confusion_matrices:
#     if confusion_matrix['depth'] == max_valid_idx + 2:
#         print(f"Depth: {confusion_matrix['depth']}")
#         print("Train:")
#         print(confusion_matrix['train'])
#         train_acc, train_precision, train_recall = calculate_metrics(
#             confusion_matrix['train']['TP'], confusion_matrix['train']['TN'],
#             confusion_matrix['train']['FP'], confusion_matrix['train']['FN'])
#         print(
#             f"Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

#         # print("Valid:")
#         # print(confusion_matrix['valid'])
#         # valid_acc, valid_precision, valid_recall = calculate_metrics(
#         #     confusion_matrix['valid']['TP'], confusion_matrix['valid']['TN'],
#         #     confusion_matrix['valid']['FP'], confusion_matrix['valid']['FN'])
#         # print(
#         #     f"Accuracy: {valid_acc:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}")

#     print("Test:")
#     print(confusion_matrix['test'])
#     test_acc, test_precision, test_recall = calculate_metrics(
#         confusion_matrix['test']['TP'], confusion_matrix['test']['TN'],
#         confusion_matrix['test']['FP'], confusion_matrix['test']['FN'])
#     print(
#         f"Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
#     print("\n")
