
from time import sleep
import cv2
import matlab.engine
import matplotlib. pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
#import open3d as o3d
#result write
#cherry picking 2022 threshhold 2.0e-4 iter 5000 match threshhold 0.85
np.random.seed(2022)
threshold = 1.0e-5
threshold_knn = 0.85
max_iter = 3000

initial_img_num_1= 0
initial_img_num_2= 1

inst = [[2.68439246e+03, 0.00000000e+00, 1.98415047e+03],
 [0.00000000e+00, 2.69934836e+03, 1.48623339e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
inst = np.array(inst)
inst_1 = np.linalg.inv(inst)
W = [[0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0]]

W = np.array(W)
kp = []
des = []
img = []
gray = []

inlinear_x_all = np.array([])
print_X = np.array([])
print_Y = np.array([])
print_Z = np.array([])
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
P_all = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])

#matlab engine start
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0)
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step5/', nargout=0)

######################################################################################################
#Step 1. Feature Extraction & Matching in General
######################################################################################################
#for three continuous images
print("--------------------------------Step1 Feature Extraction & Matching in General--------------------------------")
print("Initial Image Number : " + str(initial_img_num_1) + ", " + str(initial_img_num_2))
#image_list_up
img_list = glob.glob('./Custom_Data/Data/*.jpg')
img_list = np.array(sorted(img_list))
img_list = img_list[np.array([initial_img_num_1, initial_img_num_2])]

#image_load_and_detect_keypoint
for i, img_name in enumerate(tqdm(img_list, desc="image load and detect keypoint")):
    _img = cv2.imread(img_name)
    _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _kp, _des = sift.detectAndCompute(_img, None)
    img.append(_img)
    gray.append(_gray)
    kp.append(_kp)
    des.append(_des)

#image_matches_keypoint_and_sorting
_matches = bf.knnMatch(des[0], des[1], k=2)
good = []
good_print = []
for m,n in _matches:
    if m.distance < threshold_knn*n.distance:
        good.append(m)
        good_print.append([m])
good = sorted(good, key=lambda x: x.distance)
matches = good
matches_cnt = len(good)
print("match pair : " + str(matches_cnt))

#matching image out
res = cv2.drawMatchesKnn(img[0], kp[0], img[1], kp[1], good_print, None, flags=2)
res = cv2.resize(res, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imwrite('./result/2view_Custom/matching.jpg', res)


######################################################################################################
#Step2. Essential Matrix Estimation
######################################################################################################
print("---------------------Step2 Essential Matrix Estimation---------------------")

#convert keypoint to np
good_kp1 = []
good_kp2 = []
query_idx = [match.queryIdx for match in matches]
train_idx = [match.trainIdx for match in matches]
good_kp1 = np.float32([kp[0][ind].pt for ind in query_idx])
good_kp2 = np.float32([kp[1][ind].pt for ind in train_idx])
max_idx = len(good_kp1)

one = np.ones((1, max_idx))
good_kp1 = np.array(good_kp1).reshape(-1, 2)
q1 = good_kp1.T
q1 = np.append(q1, one, axis=0)
norm_q1 = inst_1@q1

good_kp2 = np.array(good_kp2).reshape(-1, 2)
q2 = good_kp2.T
q2 = np.append(q2, one, axis=0)
norm_q2 = inst_1@q2

_print_rgb1 = img[0]
_print_rgb1 = _print_rgb1[q1[1].astype(np.int32), q1[0].astype(np.int32)]
_print_rgb2 = img[1]
_print_rgb2 = _print_rgb2[q2[1].astype(np.int32), q2[0].astype(np.int32)]
_print_rgb =  (_print_rgb1 + _print_rgb2) / 2

#5 Point Algorithm with RANSAC
E_est = None
cnt_max = 0
inlinear = None
for iter in tqdm(range(max_iter), desc="5 Point Algorithm with RANSAC"):
    random_pt = np.random.randint(0, max_idx, size=5)
    norm_q1_rand = norm_q1[:, random_pt]
    norm_q1_rand_in = matlab.double(norm_q1_rand.tolist())
    norm_q2_rand = norm_q2[:, random_pt]
    norm_q2_rand_in = matlab.double(norm_q2_rand.tolist())
    E = eng.calibrated_fivepoint(norm_q1_rand_in, norm_q2_rand_in)
    E = np.array(E)
    for j in range(len(E[0])):
        Ei = E[:, j].reshape(3, 3)
        loss = np.diag(norm_q2.T@Ei@norm_q1)
        cnt_all = sum(np.where(((loss < threshold) & (loss > 0)), True, False))
        if cnt_max < cnt_all:
            E_est = Ei
            cnt_max = cnt_all
            inlinear_TF = np.where(((loss < threshold) & (loss > 0)), True, False)
            inlinear = np.where(((loss < threshold) & (loss > 0)))
inlinear = np.array(inlinear).reshape(-1)

_print_kp = np.concatenate((norm_q1[0:2], norm_q2[0:2]), axis=0)
_print_kp = np.concatenate((_print_kp, inlinear_TF.reshape(1, -1)), axis=0).T
_print_kp = np.concatenate((_print_kp, _print_rgb1), axis=1)
_print_kp = np.concatenate((_print_kp, _print_rgb2), axis=1)

print("Essential Matrix : ")
print(E_est)
df = pd.DataFrame(_print_kp, columns=['q1_x','q1_y', 'q2_x', 'q2_y', 'inlinear', 'r1', 'g1', 'b1', 'r2', 'g2', 'b2'])
df.to_csv('./result/2view_Custom/Init_kp.csv', mode='w')

df = pd.DataFrame(E_est)
df.to_csv('./result/2view_Custom/Ematrix.csv', mode='w')

######################################################################################################
#Step3, 4. Essential Matrix Decomposition & Triangulation
######################################################################################################
print("-----------------Step3, 4 Essential Matrix Decomposition & Triangulation-----------------")
U, s, VT = np.linalg.svd(E_est, full_matrices = True)
u3 = U[:, 2].reshape(3, 1)
P = [
    np.append(U@W@VT, u3, axis=1),
    np.append(U@W@VT, -u3, axis=1),
    np.append(U@W.T@VT, u3, axis=1),
    np.append(U@W.T@VT, -u3, axis=1)]
EM_cnt = [0, 0, 0, 0]
EM0 = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
E_idx = None
for j in range(4):
    EM1 = P[j]
    for k in range(len(inlinear)):
        A = np.array([norm_q1[0, inlinear[k]]*EM0[2] - EM0[0],
                  norm_q1[1, inlinear[k]]*EM0[2] - EM0[1],
                  norm_q2[0, inlinear[k]]*EM1[2] - EM1[0],
                  norm_q2[1, inlinear[k]]*EM1[2] - EM1[1]])
        U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
        X = V_A[3]/V_A[3, 3]
        if X[2]>0 and (EM1@X.T)[2]>0:
            EM_cnt[j] += 1

print(EM_cnt)
E_idx = np.argmax(EM_cnt)
EM1 = P[E_idx]

print("Camera 1 Pose : ")
print(EM0)
print("Camera 2 Pose : ")
print(EM1)

inlinear_X = []
_inlinear = []
for k in range(len(inlinear)):
    A = np.array([norm_q1[0, inlinear[k]]*EM0[2] - EM0[0],
                  norm_q1[1, inlinear[k]]*EM0[2] - EM0[1],
                  norm_q2[0, inlinear[k]]*EM1[2] - EM1[0],
                  norm_q2[1, inlinear[k]]*EM1[2] - EM1[1]])
    U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
    X = V_A[3]/V_A[3, 3]
    if X[2]>0 and (EM1@X.T)[2] > 0:
        if X[2] < 5 and X[2] > 1 and (EM1@X.T)[2] < 5 and (EM1@X.T)[2] > 1:
            inlinear_X.append(X[:3])
            _inlinear.append(inlinear[k])


_inlinear = np.array(_inlinear)
inlinear_X = np.array(inlinear_X)
point_cloud = np.concatenate((inlinear_X, _print_rgb[_inlinear]), axis=1)
point_cloud = np.concatenate((point_cloud, _inlinear.reshape(-1, 1)), axis=1)

df = pd.DataFrame(np.concatenate((EM0, EM1), axis = 0))
df.to_csv('./result/2view_Custom/Camera-Pose.csv', mode='w')

df = pd.DataFrame(point_cloud, columns=['x','y','z','r', 'g', 'b','inlinear_idx'])
df.to_csv('./result/2view_Custom/3D_Point_Clouds_Two_Views.csv', mode='w')

print("----------------End----------------")
print_X = np.append(print_X, inlinear_X[:, 0])
print_Y = np.append(print_Y, inlinear_X[:, 1])
print_Z = np.append(print_Z, inlinear_X[:, 2])
print(len(print_X))
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.scatter3D(print_X, print_Y, print_Z, c='b', marker='o')
plt.show()