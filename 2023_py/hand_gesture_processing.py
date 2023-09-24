import numpy as np
import math
import cv2
import matplotlib.image as mi
import matplotlib.pyplot as plt
import time
import os
import natsort 

t1 = time.time()

def showBinarizedImages(images):
    n_images = len(images)

    plt.figure(figsize=(10, 5))
    for i in range(n_images):
        plt.subplot(10, 5, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    
    plt.show()

def imgRead(fL, dir1):
    total = []
    
    newtmpl = np.zeros((200,2000))
    for ii in dir1:      
        addr = ii + '//'
        os.chdir(addr)
        file1 = os.listdir()
        imgDB = np.zeros((len(file1),200,200))
        tmpl = np.zeros((200,200))
        fL.append(len(file1))
        total.append(imgDB)
        
        for jj in range(len(file1)):
            image = cv2.imread(file1[jj], cv2.IMREAD_GRAYSCALE)
            filter_image = cv2.GaussianBlur(image,(0,0),0.07)
            binary_image = cv2.adaptiveThreshold(filter_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 467, 32)
            
            list = []
            
            for kk in range(0, binary_image.shape[0]):
                num = np.count_nonzero(binary_image[kk,:] == 255)
                list.append(num)
                
            Larray = np.array(list)
            list_maxValue = np.argmax(Larray)
            hand_location = np.asarray(np.where(binary_image[list_maxValue] == 255))
            hand_center = (hand_location[0][0] + hand_location[0][-1]) // 2
            last = np.nonzero(list)
            last1 = np.asarray(last)
            wrist_loc = np.asarray(np.where(binary_image[last1[0][-1]] == 255))
            wrist_center_loc = (wrist_loc[0][0] + wrist_loc[0][-1]) //2

            angle = abs(math.atan2(list_maxValue - last1[0][-1], hand_center - wrist_center_loc))
            degree = math.degrees(angle)

            matrix = cv2.getRotationMatrix2D((int(hand_center),int(list_maxValue)), 90-degree, 1)
            rotate_image = cv2.warpAffine(binary_image, matrix, (200,200))

            A = np.argmax(np.sum(rotate_image,axis=1)).astype(int)
            maxValue = image[A]
            radius = np.count_nonzero(maxValue) // 2

            if(radius > 30):
                radius = 30
            rotate_image[200-radius:200] = 0
            cut_image = rotate_image
            
            counters, _ = cv2.findContours(cut_image,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pointx = []
            pointy = []

            for i in range(len(counters)):
                for j in range(len(counters[i])):
                    pointx.append(counters[i][j][0][0])
                    pointy.append(counters[i][j][0][1])

            xmin = np.min(pointx)
            xmax = np.max(pointx)
            ymin = np.min(pointy)
            ymax = np.max(pointy)

            x = xmin
            y = ymin
            width = xmax - xmin
            height = ymax - ymin

            scale_image = cv2.resize(cut_image[y:y+height, x:x+width], (190, 190))
            scale_image = cv2.resize(scale_image,(200,200))

            tmpl += scale_image 
            imgDB[jj:,:] = scale_image
               
        #showBinarizedImages(imgDB)
        tmpl = tmpl/(jj+1)
        newtmpl[0:200 , (int(ii)-1)*200:(int(ii)-1)*200+200] = tmpl
        os.chdir("/Users/juminjae/Desktop/vscode_file/2023_py/image")
    print("imgRead Complete!")
    return total, newtmpl
        
def tmplMake(total): #이미지를 비교할 대상
    tmpl = newtmpl
##    tmpl = np.zeros((200,2000))
##    for ii in range(10):
##        tmpl[:,ii*200:(ii+1)*200] = total[ii][0]
    
    print("tmplMake Complete!")
    return tmpl


def tmplMatch(total, tmpl):
    result = np.ones((10, 50))*-1
    for ii in range(10):
        temp=[]
        for jj in range(fL[ii]):
            X = total[ii][jj]
            X1 = np.tile(X, (1,10))
            error= np.sum(np.abs(tmpl - X1), axis = 0)
            error2 = [error[0:200].sum(),error[200:400].sum(),error[400:600].sum(),
            error[600:800].sum(), error[800:1000].sum(),error[1000:1200].sum(),
             error[1200:1400].sum(), error[1400:1600].sum(),error[1600:1800].sum(),
             error[1800:2000].sum()]
            temp = np.argmin(error2)
            result[ii, jj] = temp
    print('tmplMatch Finished!!')
    global resultS
    resultS = len(result)
    return result

def cMatMake(fL, result):
    cMat = np.zeros((10,10))

    bound = np.arange(-0.5, 9.6, 1)
    for ii in range(10):
        hist, ddd = np.histogram(result[ii,:], bound)
        cMat[ii,:] = hist

        
    print("cMatMake Complete!")
    return cMat

def cMatSimMake(cMat):
    cMatSim = np.zeros((4,10))
    label =  np.tile(np.arange(0,10).reshape(10,1), (1,50))

    for ii in range(10):
        cMatSim[0,ii] = ((label == result)&(label == ii)).sum()
        cMatSim[1,ii] = ((ii != result)&(label != ii)&(result != -1)).sum()
        cMatSim[2,ii] = ((ii == result)&(label != ii)).sum()
        cMatSim[3,ii] = ((ii != result)&(label == ii)&(result != -1)).sum()

    print('cMatSim Finished!!')        
    return cMatSim

def scoreMake(cMatSim,fL):

    score = [[0 for jj in range(resultS)] for ii in range(5)]

    for ii in range(resultS):
        accuracy = (cMatSim[0][ii]+cMatSim[1][ii])/ \
                   (cMatSim[0][ii]+cMatSim[1][ii]+cMatSim[2][ii]+cMatSim[3][ii])
        precision = (cMatSim[0][ii])/ (cMatSim[0][ii]+cMatSim[2][ii])
        recall = (cMatSim[0][ii])/ (cMatSim[0][ii]+cMatSim[3][ii])
        f1 = 2*(accuracy*recall)/(accuracy+recall)
        recog = cMatSim[0][ii]/fL[ii]

        score[0][ii] = accuracy
        score[1][ii] = precision
        score[2][ii] = recall
        score[3][ii] = f1
        score[4][ii] = recog

    score = np.hstack((np.mean(score,1).reshape(5,1), score))
    print('scoreMake Finished!!')        
    return score


if __name__ == "__main__":
    fL = []
    os.chdir("/Users/juminjae/Desktop/vscode_file/2023_py/image")
    dir1 = os.listdir()
    dir1 = natsort.natsorted(dir1)

    total,newtmpl = imgRead(fL, dir1)
    tmpl = tmplMake(total)
    result = tmplMatch(total, tmpl)
    cMat = cMatMake(fL, result)
    cMatSim = cMatSimMake(cMat)
    score = scoreMake(cMatSim,fL)

    print(np.array(score))

    t2 = time.time()

    print((t2-t1))
 
    





    
