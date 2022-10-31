from email.errors import NonPrintableDefect
from time import sleep
import cv2
import matlab.engine
import matplotlib. pyplot as plt
import numpy as np
import time
from tqdm import tqdm
image_num = 3
threshold = 1.0e-3
threshold_pose = 500
max_iter = 3000
max_iter_pose = 3000

initial_img_num_1= 0
initial_img_num_2= 1

inst = [[3451.5, 0.0, 2312.0],
       [0.0,3451.5, 1734.0],
       [0.0, 0.0, 1.0]]
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

#matlab engine start
eng = matlab.engine.start_matlab()
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step2/', nargout=0)
eng.addpath(r'/home/dongmin/2022_GIST_lec_CV-PA/PA2/Step5/', nargout=0)

P_all = np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])

######################################################################################################
#Step 1. Feature Extraction & Matching in General
######################################################################################################
#for three continuous images
print("----------------Step1 Feature Extraction & Matching in General----------------")
print("Image Count:" + str(image_num))

#image_list_up
img_list = []
for i in range(image_num):
    img_list.append('./Data/sfm' + format(i, '02') + '.jpg')

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
matches = np.array([[None]*image_num]*image_num)
matches_cnt = np.array([[0]*image_num]*image_num)
for i in tqdm(range(image_num), desc="match keypoint all images"):
    for j in range(image_num):
        if i != j:
            _matches = bf.knnMatch(des[i], des[j], k=2)
            good = []
            for m,n in _matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            good = sorted(good, key=lambda x: x.distance)
            matches[i, j] = good
            matches_cnt[i, j] = len(good)
print(matches_cnt)

######################################################################################################
#Step2. Essential Matrix Estimation
######################################################################################################
print("---------------------Step2 Essential Matrix Estimation---------------------")

#convert keypoint to np
good_kp1 = []
good_kp2 = []
query_idx = [match.queryIdx for match in matches[initial_img_num_1, initial_img_num_2]]
train_idx = [match.trainIdx for match in matches[initial_img_num_1, initial_img_num_2]]
good_kp1 = np.float32([kp[initial_img_num_1][ind].pt for ind in query_idx])
good_kp2 = np.float32([kp[initial_img_num_2][ind].pt for ind in train_idx])
max_idx = len(good_kp1)
print(max_idx)

one = np.ones((1, max_idx))
good_kp1 = np.array(good_kp1).reshape(-1, 2)
q1 = good_kp1.T
q1 = np.append(q1, one, axis=0)
    
good_kp2 = np.array(good_kp2).reshape(-1, 2)
q2 = good_kp2.T
q2 = np.append(q2, one, axis=0)

#5 Point Algorithm with RANSAC
E_est = None
cnt_max = 0
inlinear = None
for iter in tqdm(range(max_iter), desc="5 Point Algorithm with RANSAC"):
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
        F = inst_1.T@Ei@inst_1
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

q1 = inst_1@q1
q2 = inst_1@q2
EM_cnt = [0, 0, 0, 0]
EM0 = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
E_idx = None
for j in range(4):
    EM1 = P[j]
    for k in range(len(inlinear)):
        A = np.array([q1[0, inlinear[k]]*EM0[2] - EM0[0],
                  q1[1, inlinear[k]]*EM0[2] - EM0[1],
                  q2[0, inlinear[k]]*EM1[2] - EM1[0],
                  q2[1, inlinear[k]]*EM1[2] - EM1[1]])
        U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
        X = V_A[3]/V_A[3, 3]
        if X[2]>0 and (EM1@X)[2]>0:
            EM_cnt[j] += 1
E_idx = np.argmax(EM_cnt)
EM1 = P[E_idx]
print(E_idx)
print(EM1)
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

inlinear_X = np.array(inlinear_X)
print_X = np.append(print_X, inlinear_X[:, 0])
print_Y = np.append(print_Y, inlinear_X[:, 1])
print_Z = np.append(print_Z, inlinear_X[:, 2])


######################################################################################################
#Step5. Growing Step
######################################################################################################
print("---------------------Step5. Growing Step---------------------")
merge_img_idx = 2
#merge_image_matching_point
train2_idx = [match.queryIdx for match in matches[initial_img_num_2, merge_img_idx]]
merge_idx = [match.trainIdx for match in matches[initial_img_num_2, merge_img_idx]]
merge_kp1 = np.float32([kp[initial_img_num_2][ind].pt for ind in train2_idx])
merge_kp2 = np.float32([kp[merge_img_idx][ind].pt for ind in merge_idx])

merge_q1 = merge_kp1.T
q1 = np.append(q1, one, axis=0)
    
good_kp2 = np.array(good_kp2).reshape(-1, 2)
q2 = good_kp2.T
q2 = np.append(q2, one, axis=0)

idx_train_inter = []
idx_3d = []
for i in range(len(train2_idx)):
    if train2_idx[i] in np.array(train_idx)[inlinear].tolist():
        idx_3d.append([(np.array(train_idx)[inlinear].tolist()).index(train2_idx[i]), i])
        idx_train_inter.append([train_idx.index(train2_idx[i]), i])
idx_train_inter = np.array(idx_train_inter)
idx_3d = np.array(idx_3d)
inlinear_in = np.concatenate((inlinear_X[idx_3d[:, 0]].T, np.ones(len(inlinear_X[idx_3d[:, 0]])).reshape(1, -1)), axis=0) 
merge_max_idx = len(idx_train_inter)
print(merge_max_idx)
distortion_coeffs = np.zeros((4,1))
pixel_cnt_max = 0
P2_max = None
#3-point algorthm with RANSAC
for iter in tqdm(range(max_iter_pose), desc="3 Point Algorithm with RANSAC"):
    random_pt = np.random.randint(0, merge_max_idx, size=3)
    point_3d = inlinear_X[idx_3d[random_pt, 0]]
    pkp1 = good_kp2[idx_train_inter[random_pt, 0]]
    merge_kp1_rand = merge_kp1[idx_train_inter[random_pt, 1]]
    merge_kp1_rand = (inst_1@np.concatenate((merge_kp1_rand.T, np.ones((1, 3))))).T
    merge_kp2_rand = merge_kp2[idx_train_inter[random_pt, 1]]
    merge_kp2_rand = (inst_1@np.concatenate((merge_kp2_rand.T, np.ones((1, 3))))).T
    pnp_in = np.concatenate((merge_kp2_rand, point_3d), axis=1)
    pnp_in = matlab.double(pnp_in)
    output = eng.PerspectiveThreePoint(pnp_in)
    #retval, rvec, tvec = cv2.solvePnP(point_3d, merge_kp2_rand, inst, distortion_coeffs, flags=cv2.SOLVEPNP_P3P)
    output = np.array(output)
    if output.shape != ():
        cnt_p3p = int(len(output) / 4)
        for cnt_p in range(cnt_p3p):
            P2 = np.array(output)[(4*cnt_p):(4*cnt_p)+3, :]
            points = inst@P2@inlinear_in
            points = points[:2, :]/points[2, :]
            loss = np.sum((points.T - merge_kp2[idx_train_inter[:, 1]])**2, axis=1)
            pixel_cnt = sum(np.where((loss < threshold_pose), True, False))
            if pixel_cnt > pixel_cnt_max:
                P2_max = P2
                pixel_cnt_max = pixel_cnt
                inlinear_merge = np.where((loss < threshold_pose))



        
EM0 = EM1                
EM1 = P2_max
for k in range(len(inlinear_merge)):
    A = np.array([q1[0, inlinear[k]]*EM0[2] - EM0[0],
                  q1[1, inlinear[k]]*EM0[2] - EM0[1],
                  q2[0, inlinear[k]]*EM1[2] - EM1[0],
                  q2[1, inlinear[k]]*EM1[2] - EM1[1]])
    U_A, s_A, V_A = np.linalg.svd(A, full_matrices=True)
    X = V_A[3]/V_A[3, 3]
    inlinear_X.append(X[:3])        
print(pixel_cnt_max)
print(P2_max)