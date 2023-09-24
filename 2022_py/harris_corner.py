import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

lena = mpimg.imread('/Users/juminjae/Desktop/2022vscode/2022_py/img/Lenna.png')
lena_gray = lena[:,:,0]*0.2989+lena[:,:,1]*0.5870+lena[:,:,2]*0.1140
pad_image = np.pad(lena_gray,((3,3),(3,3)))
height, width = lena_gray.shape
A = np.zeros((height,width))
W = np.zeros((7,7))
q = np.arange(-3,4)
X,Y = np.meshgrid(q,q)
for y in range(0,7):
    for x in range(0,7):
        W[y,x] = np.exp(-1*(X[y,x]**2+Y[y,x]**2)/(0.7**2))
#print(W)
k = 0.04

for i in range(height-1):
    for j in range(width-1):
        I = pad_image[i:i+7,j:j+7]
        I2 = pad_image[i:i+7,j+1:j+8]
        I3 = pad_image[i+1:i+8,j:j+7]
        ix = I2 - I
        iy = I3 - I
        
        Ix = ix**2
        Ixy = ix*iy
        Iy = iy**2

        Wx = W*Ix
        Wxy = W*Ixy
        Wy = W*Iy
        M = np.array([[np.sum(Wx),np.sum(Wxy)],[np.sum(Wxy),np.sum(Wy)]])
        R = np.linalg.det(M) - k*np.trace(M)**2
        if(R>0.0001):
            plt.plot(j,i,'r.')

plt.imshow(pad_image,cmap='gray')
plt.axis([3,512,512,3])
plt.show()
