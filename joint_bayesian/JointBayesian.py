import os
import sys
import numpy as np
import joblib
from joint_bayesian.utils import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from tqdm import tqdm

#Before training,the mean must be substract
def JointBayesian_Train(trainingset, label, fold = "./"):
    print(trainingset.shape)
    # the total num of image
    n_image = len(label)
    # the dim of features
    n_dim   = trainingset.shape[1]
    # filter the complicate label,for count the total people num
    classes, labels = np.unique(label, return_inverse=True)
    # the total people num
    n_class = len(classes)
    # save each people items
    cur = {}
    withinCount = 0
    # record the count of each people
    numberBuff = np.zeros(n_image)
    maxNumberInOneClass = 0
    print('preparing data...')
    for i in tqdm(range(n_class)):
        # get the item of i
        cur[i] = trainingset[labels==i]
        # get the number of the same label persons
        n_same_label = cur[i].shape[0]
        
        if n_same_label > 1:
            withinCount += n_same_label
        if numberBuff[n_same_label] == 0:
            numberBuff[n_same_label] = 1
            maxNumberInOneClass = max(maxNumberInOneClass, n_same_label)
    print("prepare done, maxNumberInOneClass=", maxNumberInOneClass)

    u  = np.zeros([n_dim, n_class])
    ep = np.zeros([n_dim, withinCount])
    nowp=0
    for i in range(n_class):
        # the mean of cur[i]
        u[:,i] = np.mean(cur[i], 0)
        b = u[:,i].reshape(n_dim, 1)
        n_same_label = cur[i].shape[0]
        if n_same_label > 1:
            ep[:, nowp:nowp+n_same_label] = cur[i].T-b
            nowp += n_same_label

    Su = np.cov(u.T,  rowvar=0)
    Sw = np.cov(ep.T, rowvar=0)
    oldSw = Sw
    SuFG  = {}
    SwG   = {}
    convergence = 1
    min_convergence = 1
    print('training...')
    for l in tqdm(range(500)):
        F  = np.linalg.pinv(Sw)
        u  = np.zeros([n_dim, n_class])
        ep = np.zeros([n_dim, n_image])
        nowp = 0
        for mi in range(maxNumberInOneClass + 1):
            if numberBuff[mi] == 1:
		#G = −(mS μ + S ε )−1*Su*Sw−1
                G = -np.dot(np.dot(np.linalg.pinv(mi*Su+Sw), Su), F)
		#Su*(F+mi*G) for u
                SuFG[mi] = np.dot(Su, (F+mi*G))
		#Sw*G for e
                SwG[mi]  = np.dot(Sw, G)
        for i in range(n_class):
            ##print l, i
            nn_class = cur[i].shape[0]
	    #formula 7 in suppl_760
            u[:,i] = np.sum(np.dot(SuFG[nn_class],cur[i].T), 1)
	    #formula 8 in suppl_760
            ep[:,nowp:nowp+nn_class] = cur[i].T+np.sum(np.dot(SwG[nn_class],cur[i].T),1).reshape(n_dim,1)
            nowp = nowp+nn_class

        Su = np.cov(u.T,  rowvar=0)
        Sw = np.cov(ep.T, rowvar=0)
        convergence = np.linalg.norm(Sw-oldSw)/np.linalg.norm(Sw)
        print_info("Iterations-" + str(l) + ": "+ str(convergence))
        
        oldSw=Sw

        if convergence < min_convergence:
       	    min_convergence = convergence
            F = np.linalg.pinv(Sw)
            G = -np.dot(np.dot(np.linalg.pinv(2*Su+Sw),Su), F)
            A = np.linalg.pinv(Su+Sw)-(F+G)
            data_to_pkl(G, os.path.join(fold, 'G.npy'))
            data_to_pkl(A, os.path.join(fold, 'A.npy'))
        
        if convergence < 1e-10:
            print("Convergence: ", l, convergence)
            break

#ratio of similar,the threshold we always choose in (-1,-2)            
def verify(A, G, embeddings_1, embeddings_2):
    length_1 = embeddings_1.shape[0]
    length_2 = embeddings_2.shape[0]

    r_1 = (embeddings_1@A@(embeddings_1.T)).diagonal().reshape(1, length_1)
    r_2 = (embeddings_2@A@(embeddings_2.T)).diagonal().reshape(1, length_2)
    result = -2*embeddings_1@G@(embeddings_2.T)

    r_1 = r_1.repeat(length_2, axis=0).T
    r_2 = r_2.repeat(length_1, axis=0)

    result += (r_1+r_2)
    return result


def PCA_Train(data, result_fold, n_components=2000):
    print_info("PCA training (n_components=%d)..."%n_components)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    joblib.dump(pca, result_fold+"pca_model.m")
     
    print_info("PCA done.")

    return pca

def data_pre(data):
    data = np.sqrt(data)
    data = np.divide(data, np.repeat(np.sum(data, 1), data.shape[1]).reshape(data.shape[0], data.shape[1]))
    
    return data

def get_ratios(A, G, pair_list, data):
    distance = []
    for pair in pair_list:
        ratio = Verify(A, G, data[pair[0]], data[pair[1]])
        distance.append(ratio)

    return distance