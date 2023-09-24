import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

me = mpimg.imread(
    '/Users/juminjae/Desktop/vscode_file/2022-2_py/img/1858031.jpg')
image = me[:, :, 0]*0.2989 + me[:, :, 1] * \
    0.5870 + me[:, :, 2]*0.1140  # gray scale
angle = math.radians(62)
a = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
H = np.array([[np.cos(angle), -np.sin(angle), 705],
             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

height = image.shape[0]  # 800
width = image.shape[1]  # 600
corner = np.array([[0, 0, width-1, width-1], [0, height-1, 0, height-1]])
new_points = a.dot(corner).astype(int)
new_points[0] = new_points[0] + 705
''' 705 0 986 281
    0 375 528 903 '''
output = np.zeros(
    (np.max(new_points[1]+1), np.max(new_points[0])+1))  # (904,987)

for i in range(height):
    for j in range(width):
        xy0 = np.array([[j], [i], [1]])
        xy1 = H.dot(xy0).astype(int)
        output[xy1[1], xy1[0]] = image[i, j]
        output2 = output

xcoord = np.arange(np.max(new_points[0])+1)
ycoord = np.arange(np.max(new_points[1])+1)
X, Y = np.meshgrid(xcoord, ycoord)
output = np.where((X < 281/528*Y+705) & (X < -705/375*(Y-528)+986)
                  & (X > 281/528*(Y-375)) & (X > -705/375*Y+705), output, 100)
output2 = np.where((X < 281/528*Y+705) & (X < -705/375*(Y-528)+986)
                   & (X > 281/528*(Y-375)) & (X > -705/375*Y+705), output2, 100)
black_spot_y, black_spot_x = np.where(output == 0)


for k in range(black_spot_x.size):  # 60786
    A = np.array([[black_spot_x[k]], [black_spot_y[k]], [1]])
    B = np.linalg.inv(H).dot(A).astype(int)
    output[black_spot_y[k], black_spot_x[k]] = image[B[1], B[0]]

    a1 = (output2[black_spot_y[k]-1, black_spot_x[k]] +
          output2[black_spot_y[k]+1, black_spot_x[k]])/2
    b1 = (output2[black_spot_y[k], black_spot_x[k]-1] +
          output2[black_spot_y[k], black_spot_x[k]+1])/2
    c1 = (a1+b1)/2
    output2[black_spot_y[k], black_spot_x[k]] = c1


output = np.where((X < 281/528*Y+705) & (X < -705/375*(Y-528)+986)
                  & (X > 281/528*(Y-375)) & (X > -705/375*Y+705), output, 0)
output2 = np.where((X < 281/528*Y+705) & (X < -705/375*(Y-528)+986)
                   & (X > 281/528*(Y-375)) & (X > -705/375*Y+705), output2, 0)
plt.subplot(1, 2, 1)
plt.axis([500, 600, 300, 400])
plt.imshow(output, cmap='gray')

plt.subplot(1, 2, 2)
plt.axis([500, 600, 300, 400])
plt.imshow(output2, cmap='gray')
plt.show()
