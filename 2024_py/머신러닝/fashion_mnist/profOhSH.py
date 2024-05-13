import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('2024_py/머신러닝/fashion_mnist/x_train.npy')
y_train = np.load('2024_py/머신러닝/fashion_mnist/y_train.npy')
x_test = np.load('2024_py/머신러닝/fashion_mnist/x_test.npy')
y_test = np.load('2024_py/머신러닝/fashion_mnist/y_test.npy')
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) #(60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# print(np.bincount(y_train)) # [6000 6000 6000 6000 6000 6000 6000 6000 6000 6000]
# print(np.bincount(y_test)) # [1000 1000 1000 1000 1000 1000 1000 1000 1000 1000]

# pixel 0~1로 만들기
x_train = x_train/255.0
x_test = x_test/255.0
train_unique_labels = np.unique(y_train)  # [0 1 2 3 4 5 6 7 8 9]
test_unique_labels = np.unique(y_test)  # [0 1 2 3 4 5 6 7 8 9]
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
for i in range(10):
    plt.subplot(5, 2, i+1)
    plt.imshow(x_train_data[1000*i])
plt.show()
