from random import random
import cv2
from cv2 import threshold
import matlab.engine
from matplotlib import pyplot as plt
import numpy as np

img1_name = '03'
img2_name = '04'
res2 = None
threshold = 100
max_iter = 100000
inst = [[3451.5, 0.0, 2312],
       [0.0 ,3451.5,  1734],
       [0.0 ,     0.0 ,   1.0]]
inst = np.array(inst)
img1 = cv2.imread('./Data/sfm' + img1_name + '.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./Data/sfm' + img2_name + '.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# ratio test 적용
good_kp1 = []
good_kp2 = []
good = []

for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_kp1.append(kp1[m.queryIdx].pt)
        good_kp2.append(kp2[m.trainIdx].pt)
        good.append(m)
max_idx = len(good_kp1)
print(max_idx)
good_kp1 = np.array(good_kp1)
good_kp2 = np.array(good_kp2)

#5-pts algorithms with RANSAC
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0)
#for i in range(max_iter):
print('step A')
E_est = None
s_est = 1e+200
for iter in range(max_iter):
    random_pt = np.random.randint(0, max_idx, size=5)
    good_kp1_rand = good_kp1[random_pt].T
    good_kp1_rand = np.append(good_kp1_rand, np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), axis = 0)
    good_kp1_rand_in = good_kp1_rand.tolist()
    good_kp1_rand_in = matlab.double(good_kp1_rand_in)
    good_kp2_rand = good_kp2[random_pt].T
    good_kp2_rand = np.append(good_kp2_rand, np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), axis = 0)
    good_kp2_rand_in = good_kp2_rand.tolist()
    good_kp2_rand_in = matlab.double(good_kp2_rand_in)
    E = eng.calibrated_fivepoint(good_kp1_rand_in, good_kp2_rand_in)
    E = np.array(E)
    E_min = None
    s_min = 1e+200
    for i in range(len(E[0])):
        Ei = E[:, i].reshape(3, 3)
        F = inst.T@Ei@inst.T
        det = np.linalg.det(F)
        sum_loss = 0
        for pt in range(max_idx):
            nkp1 = np.append(good_kp1[pt], 1.0)
            nkp2 = np.append(good_kp2[pt], 1.0)
            loss = np.abs(nkp1.T@inst.T@Ei@inst@nkp2)
            if loss < threshold:
                sum_loss += loss
                
        if s_min > s:
            E_min = Ei
            s_min = s
    loss = None
    for pt in range(max_idx):
        
        
        
            
    if s_est > s_min:
        E_est = E_min
        s_est = s_min
print(E_est)
print(s_est)

U, s, V = np.linalg.svd(E, full_matrices = True)
#W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
"""res2 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,res2,flags=2)        
res2 = cv2.resize(res2, dsize=(0, 0), fx=0.5, fy=0.5)
cv2.imshow("BF with SIFT",res2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""