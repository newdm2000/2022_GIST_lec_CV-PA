from email.errors import NonPrintableDefect
from time import sleep
import cv2
import matlab.engine
import matplotlib. pyplot as plt
#%matplotlib inline
import numpy as np
import time
from tqdm import tqdm

#---------setting----------
image_num = 3
threshold = 5.0e-5
max_iter = 5000
max_iter_pose = 3000
inst = [[3451.5, 0.0, 2312.0],
       [0.0,3451.5, 1734.0],
       [0.0, 0.0, 1.0]]
inst = np.array(inst)
inst_1 = np.linalg.inv(inst)
W = [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]
W = np.array(W)
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0)
P_all = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
#Step 1. Feature Extraction & Matching in General
print("----------------Step1----------------")
print("Image Count:" + str(image_num))
img_list = []
for i in range(image_num):
    img_list.append('./Data/sfm' + format(i, '02') + '.jpg')

kp = []
des = []
img = []
gray = []
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

inlinear_x_all = np.array([])
print_X = np.array([])
print_Y = np.array([])
print_Z = np.array([])

for i, img_name in enumerate(tqdm(img_list, desc="image load and detect keypoint")):
    _img = cv2.imread(img_name)
    _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _kp, _des = sift.detectAndCompute(_img, None)
    img.append(_img)
    gray.append(_gray)
    kp.append(_kp)
    des.append(_des)
    
kp = np.array(kp)
des = np.array(des)
img = np.array(img)
gray = np.array(gray)

"""matches = np.array([[None]*image_num]*image_num)
matches_cnt = np.array([[0]*image_num]*image_num)
for i in tqdm(range(image_num), desc="match keypoint all images"):
    for j in range(i):
        _matches = bf.knnMatch(des[i], des[j], k=2)
        good = []
        #print_good = None
        for m,n in _matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
                #print_good.append([m])
        matches[i, j] = good
        matches_cnt[i, j] = len(good)
print(matches_cnt)


#list images most matches with Floyd-Warshall Algorithm
def floyd_warshall(n, data):
    dist = data
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = max(dist[i][j], dist[i][k] + dist[k][j])
    return dist

print(floyd_warshall(image_num, matches_cnt))
print("A")
res2 = None

inst = np.array(inst)
inst_1 = np.linalg.inv(inst)"""
matches = []
matches_cnt = np.zeros(image_num-1)
for i in tqdm(range(image_num-1), desc="match keypoint all images"):
    _matches = bf.knnMatch(des[i], des[i+1], k=2)
    good = []
        #print_good = None
    for m,n in _matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            #print_good.append([m])
    good = sorted(good, key=lambda x: x.distance)
    matches.append(good)
    matches_cnt[i] = len(good)
matches = np.array(matches)
ink = []
for i in range(image_num-1):
    image1_idx = i
    image2_idx = i+1
    print("image " + str(image1_idx) + " and " + str(image2_idx))
    rel_idx = i
    good_kp1 = []
    good_kp2 = []
    query_idx = [match.queryIdx for match in matches[rel_idx]]
    train_idx = [match.trainIdx for match in matches[rel_idx]]
    good_kp1 = np.float32([kp[image1_idx][ind].pt for ind in query_idx])
    good_kp2 = np.float32([kp[image2_idx][ind].pt for ind in train_idx])
    max_idx = len(good_kp1)
    print(max_idx)
    
    one = np.ones((1, max_idx))
    good_kp1 = np.array(good_kp1).reshape(-1, 2)
    q1 = good_kp1.T
    q1 = np.append(q1, one, axis=0)
    
    good_kp2 = np.array(good_kp2).reshape(-1, 2)
    q2 = good_kp2.T
    q2 = np.append(q2, one, axis=0)


#5-pts algorithms with RANSAC

    print('step B')
    E_est = None
    cnt_max = 0
    inlinear = None
    for iter in tqdm(range(max_iter)):
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
        for j in range(len(E[0])):
            Ei = E[:, j].reshape(3, 3)
            det = np.linalg.det(Ei)
            #Ei = Ei
            F = inst_1.T@Ei@inst_1
            #F = F.T
            det = np.linalg.det(F)
            const = 2 * Ei @ Ei.T @ Ei - np.trace(Ei@Ei.T)*Ei
            loss = np.diag(good_kp2_rand.T@F@good_kp1_rand)
            loss_all = np.diag(q2.T@F@q1)
            cnt_all = sum(np.where(((loss_all < threshold) & (loss_all > -threshold)), True, False))
            if cnt_max < cnt_all:
                E_est = Ei
                cnt_max = cnt_all
                inlinear = np.where(((loss_all < threshold) & (loss_all > -threshold)))
    print(cnt_max)
    inlinear = np.array(inlinear).reshape(-1)
    ink.append(len(inlinear))


    print("step C")
    U, s, VT = np.linalg.svd(E_est, full_matrices = True)
    u3 = U[:, 2].reshape(3, 1)
    P = [
    np.append(U@W@VT, u3, axis=1),
    np.append(U@W@VT, -u3, axis=1),
    np.append(U@W.T@VT, u3, axis=1),
    np.append(U@W.T@VT, -u3, axis=1)]
#print(P1)
#W = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]

    q1 = inst_1@q1
    q2 = inst_1@q2
    EM0 = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    E_idx = None
    for j in range(4):
        EM1 = P[j]
        A = np.array([q1[0, inlinear[0]]*EM0[2] - EM0[0],
                  q1[1, inlinear[0]]*EM0[2] - EM0[1],
                  q2[0, inlinear[0]]*EM1[2] - EM1[0],
                  q2[1, inlinear[0]]*EM1[2] - EM1[1]])
        U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
        X = V_A[3]/V_A[3, 3]
        print(X)
        if X[2]>0 and (EM1@X)[2]>0:
            print("E index is " + str(j))
            E_idx = j
    EM1 = P[E_idx]
    P_all =np.concatenate((P_all, EM1.reshape(1, 3, 4)), axis=0)
    inlinear_X = []

    for k in range(len(inlinear)):
        A = np.array([q1[0, inlinear[k]]*EM0[2] - EM0[0],
                  q1[1, inlinear[k]]*EM0[2] - EM0[1],
                  q2[0, inlinear[k]]*EM1[2] - EM1[0],
                  q2[1, inlinear[k]]*EM1[2] - EM1[1]])
        U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
        X = V_A[3]/V_A[3, 3]
        inlinear_X.append(X[:3])
    inlinear_X = np.array(np.linalg.inv(P_all[i, :, :3])@(np.array(inlinear_X)-P_all[i, :, 3]).T).T
    #inlinear_X_all = np.append(inlinear_x_all, inlinear_X, axis=0)
    print_X = np.append(print_X, inlinear_X[:, 0])
    print_Y = np.append(print_Y, inlinear_X[:, 1])
    print_Z = np.append(print_Z, inlinear_X[:, 2])
    print(len(print_X))
print("----------------End----------------")
print(P_all.shape)
print(len(print_X))
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.scatter3D(print_X[:ink[0]], print_Y[:ink[0]], print_Z[:ink[0]], c='b', marker='o')
ax.scatter3D(print_X[ink[0]:], print_Y[ink[0]:], print_Z[ink[0]:], c='b', marker='x')
plt.show()

"""    print("camera pose")
    for iter_pose in range(max_iter_pose):
        random_pt = np.random.randint(0, max_idx, size=3)"""
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

"""res2 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,res2,flags=2)        
res2 = cv2.resize(res2, dsize=(0, 0), fx=0.5, fy=0.5)
cv2.imshow("BF with SIFT",res2)
cv2.waitKey(0)
cv2.d """