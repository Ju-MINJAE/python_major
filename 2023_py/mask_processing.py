import cv2
import numpy as np
import matplotlib.pyplot as plt
from Common.otsu import otsu

#image = cv2.imread('/Users/juminjae/Desktop/vscode_file/2023_py/1858031_mask.jpeg',cv2.IMREAD_GRAYSCALE)
image = cv2.imread('/Users/juminjae/Desktop/vscode_file/2023_py/mask2.jpeg',cv2.IMREAD_GRAYSCALE)

def calcHist(img):
    height,width = img.shape[:2]
    hist = np.zeros((256,))
    for row in range(height):
        for col in range(width):
            intensity = img[row,col]
            hist[intensity] += 1
    return hist

def window_sliding(img):
    window_size = np.zeros((609,256))
    i = 0
    # window size에 따라 반복문 변경
    # 29,21 (100,100) / 59,42 (50,50)
    for row in range(29):
        for col in range(21):
            hist = calcHist(img[row*25:row*25+100,col*25:col*25+100])
            window_size[i] = hist
            i+=1

    return window_size

def distance_hist():
    window_size = window_sliding(image)
    dst = np.zeros((609,609))

    for row in range(609):
        for col in range(609):
            dst[row,col] = np.linalg.norm(window_size[row]-window_size[col])

    return dst

for i in range(256):
    result = otsu(image)

blue_dot = np.where(result==255)
#print(blue_dot[1].size) # 58809
plt.imshow(image,cmap='gray')
#plt.scatter(blue_dot[1],blue_dot[0],c='b',s=0.1)
plt.scatter(blue_dot[1],blue_dot[0],c='b',s=0.1,alpha=0.1)
plt.show()
