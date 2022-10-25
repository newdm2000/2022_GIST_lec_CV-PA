from time import sleep
import cv2
import matlab.engine
import matplotlib. pyplot as plt
#%matplotlib inline
import numpy as np
import time

img1_name = '03'
img2_name = '04'
res2 = None
threshold = 0.01
max_iter = 500
inst = [[3451.5, 0.0, 2312.0],
       [0.0,3451.5, 1734.0],
       [0.0, 0.0, 1.0]]
W = [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]
print('step A')

inst = np.array(inst)
inst_1 = np.linalg.inv(inst)
W = np.array(W)
img1 = cv2.imread('./Data/sfm' + img1_name + '.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('./Data/sfm' + img2_name + '.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good_kp1 = []
good_kp2 = []
good = []

for m,n in matches:
    if m.distance < 0.70*n.distance:
        good.append(m)
    
good = sorted(good, key=lambda x: x.distance)
query_idx = [match.queryIdx for match in good]
train_idx = [match.trainIdx for match in good]
good_kp1 = np.float32([kp1[ind].pt for ind in query_idx])
good_kp2 = np.float32([kp2[ind].pt for ind in train_idx])
max_idx = len(good_kp1)
print(max_idx)

"""res2 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,res2, flags=2)
res2 = cv2.resize(res2, dsize=(0, 0), fx=0.2, fy=0.2)
cv2.imshow('SIFT', res2)
cv2.waitKey()
cv2.destroyAllWindows()
time.sleep(10)"""
one = np.ones((1, max_idx))
good_kp1 = np.array(good_kp1).reshape(-1, 2)
q1 = good_kp1.T
q1 = np.append(q1, one, axis=0)

good_kp2 = np.array(good_kp2).reshape(-1, 2)
q2 = good_kp2.T
q2 = np.append(q2, one, axis=0)

#5-pts algorithms with RANSAC
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0)
print('step B')
E_est = None
cnt_max = 0
inlinear = None
for iter in range(max_iter):
    random_pt = np.random.randint(0, max_idx, size=5)
    good_kp1_rand = good_kp1[random_pt].T
    good_kp1_rand = np.append(good_kp1_rand, np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), axis = 0)
    good_kp1_rand_in = (inst_1@good_kp1_rand).tolist()
    good_kp1_rand_in = matlab.double(good_kp1_rand_in)
    good_kp2_rand = good_kp2[random_pt].T
    good_kp2_rand = np.append(good_kp2_rand, np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]), axis = 0)
    good_kp2_rand_in = (inst_1@good_kp2_rand).tolist()
    good_kp2_rand_in = matlab.double(good_kp2_rand_in)
    E = eng.calibrated_fivepoint(good_kp1_rand_in, good_kp2_rand_in)
    E = np.array(E)
    for i in range(len(E[0])):
        Ei = E[:, i].reshape(3, 3)
        det = np.linalg.det(Ei)
        #Ei = Ei
        F = inst_1.T@Ei@inst_1
        #F = F.T
        det = np.linalg.det(F)
        #const = 2 * F @ F.T @ F - np.trace(F@F.T)*F
        loss = np.diag(good_kp2_rand.T@F@good_kp1_rand)
        loss_all = np.diag(q2.T@F@q1)
        cnt_all = sum(np.where(((loss_all < threshold) & (loss_all > -threshold)), True, False))
        if cnt_max < cnt_all:
            E_est = Ei
            cnt_max = cnt_all
            inlinear = np.where(((loss_all < threshold) & (loss_all > -threshold)))
E_true, mask = cv2.findEssentialMat(good_kp1, good_kp2, method=cv2.RANSAC, focal=3451.5, pp=(2312, 1734), maxIters = 3000, threshold=0.01)
print(E_true)
print(E_est)
print(cnt_max)
inlinear = np.array(inlinear).reshape(-1)
print(len(inlinear))
"""E_min = None
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
    for pt in range(max_idx):"""
        
        
        
""" if s_est > s_min:
        E_est = E_min
        s_est = s_min
print(E_est)
print(s_est) """

print("step C")
U, s, VT = np.linalg.svd(E_est, full_matrices = True)
print(s)
u3 = U[:, 2].reshape(3, 1)
print(U)
print(u3)
P = [
np.append(U@W@VT, u3, axis=1),
np.append(U@W@VT, -u3, axis=1),
np.append(U@W.T@VT, u3, axis=1),
np.append(U@W.T@VT, -u3, axis=1)]
#print(P1)
#W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
"""res2 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,res2,flags=2)        
res2 = cv2.resize(res2, dsize=(0, 0), fx=0.5, fy=0.5)
cv2.imshow("BF with SIFT",res2)
cv2.waitKey(0)
cv2.d """
for i in range(1):
    for j in range(4):
        print(P[j])
        print(P[j][:, :3])
        print(np.linalg.det(P[j][:, :3]))
        #print(dt)
