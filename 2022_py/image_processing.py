import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

me1 = mpimg.imread(
    '/Users/juminjae/Desktop/vscode_file/2022-2_py/img/Pro4_4.jpeg')
image1 = me1[:, :, 0]*0.2989 + me1[:, :, 1]*0.5870 + me1[:, :, 2]*0.1140
'''
############ 매번 클릭을 줄이고자 I1을 고정시킴 #################
plt.imshow(image1,cmap='gray')
I1 = plt.ginput(30,0)
plt.show()
I1 = np.round(I1)'''

I1 = np.array([[78, 63], [197, 65], [318, 67], [435, 71], [553, 73], [74, 204], [202, 203],
               [315, 199], [432, 205], [554, 205], [72, 337], [
                   195, 338], [319, 338], [434, 339],
               [554, 339], [69, 475], [193, 478], [317, 477], [
                   434, 477], [557, 478], [62, 616],
               [190, 616], [315, 617], [436, 616], [
    562, 618], [60, 758], [185, 757], [315, 758],
    [436, 755], [564, 756]])

me2 = mpimg.imread(
    '/Users/juminjae/Desktop/vscode_file/2022-2_py/img/Pro4_4.jpeg')
image2 = me2[:, :, 0]*0.2989 + me2[:, :, 1]*0.5870 + me2[:, :, 2]*0.1140
plt.imshow(image2, cmap='gray')
I2 = plt.ginput(30, 0)
plt.show()
I2 = np.round(I2)

'''
I2 = np.array([[67,81],[189,83],[308,77],[429,81],[552,83],[59,216],[201,213],
            [306,209],[430,215],[557,217],[63,356],[187,355],[310,357],[424,354],
            [548,357],[65,492],[188,490],[308,492],[426,492],[547,492],[62,626],
            [183,631],[305,634],[426,629],[548,630],[68,745],[186,749],[306,751],
            [420,750],[538,747]])
            '''

output = np.zeros((800, 600))

for i in range(5):
    for j in range(4):
        [x1, y1] = np.array([[I1[i*5+j][0], I1[i*5+j+1][0], I1[(i+1)*5+j+1][0], I1[(i+1)*5+j][0]],
                             [I1[i*5+j][1], I1[i*5+j+1][1], I1[(i+1)*5+j+1][1], I1[(i+1)*5+j][1]]])

        [x2, y2] = np.array([[I2[i*5+j][0], I2[i*5+j+1][0], I2[(i+1)*5+j+1][0], I2[(i+1)*5+j][0]],
                             [I2[i*5+j][1], I2[i*5+j+1][1], I2[(i+1)*5+j+1][1], I2[(i+1)*5+j][1]]])

        A = np.array([[x2[0], x2[1], x2[2], x2[3]],
                     [y2[0], y2[1], y2[2], y2[3]]])
        B = np.array([[x1[0]*y1[0], x1[1]*y1[1], x1[2]*y1[2], x1[3]*y1[3]],
                      [x1[0], x1[1], x1[2], x1[3]],
                      [y1[0], y1[1], y1[2], y1[3]],
                      [1, 1, 1, 1]])
        C = A.dot(np.linalg.inv(B))  # a0 a1 a2 a3 / b0 b1 b2 b3 (2,4)

        for l in range(np.min(y1), np.max(y1)):
            for k in range(np.min(x1), np.max(x1)):
                H = np.array([[k*l], [k], [l], [1]])
                D = C.dot(H).astype(int)
                output[D[1], D[0]] = image1[l, k]
                if (output[l][k] == 0):
                    output[l][k] = (output[l-1][k]+output[l+1][k]+output[l][k-1]+output[l][k+1] +
                                    output[l-1][k-1]+output[l-1][k+1]+output[l+1][k-1]+output[l+1][k+1])/8

plt.imshow(output, cmap='gray')
plt.show()
