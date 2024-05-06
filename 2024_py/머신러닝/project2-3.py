import numpy as np
import matplotlib.pyplot as plt


def decision_tree(train_data, threshold=0.95):
    feat1, feat2, feat3, label = train_data[:,
                                            0], train_data[:, 1], train_data[:, 2], train_data[:, 3]
    n = 7
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
    accuracy = np.mean(predicted_labels == label)
    if (accuracy < 0.5):
        accuracy = 1-accuracy
    print(accuracy)

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
        decision_tree(train_data[train_data[:, 0] <=
                      feat1_div[split_feature]], threshold)
        # decision_tree(train_data[train_data[:, 0] >
        #               feat1_div[split_feature]], threshold)
    elif split_threshold == 1:
        decision_tree(train_data[train_data[:, 1] <=
                      feat2_div[split_feature]], threshold)
        # decision_tree(train_data[train_data[:, 1] >
        #               feat2_div[split_feature]], threshold)
    else:
        decision_tree(train_data[train_data[:, 2] <=
                      feat3_div[split_feature]], threshold)
        # decision_tree(train_data[train_data[:, 2] >
        #               feat3_div[split_feature]], threshold)

    if accuracy >= threshold:
        return


train_data = np.loadtxt(
    '/Users/juminjae/Desktop/python/2024_py/머신러닝/train.txt')
decision_tree(train_data)
