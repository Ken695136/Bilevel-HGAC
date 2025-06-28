
import time
import scipy.sparse as sp
import numpy as np
import torch

def embedding_adj(emb, topk):

    adj = np.matmul(emb, emb.T)
    adj[4019:11186, 11186:11246] = 0
    adj[11186:11246, 4019:11186] = 0
    adj[0:4019, 0:4019] = 0
    adj[4019:11186, 4019:11186] = 0
    adj[11186:11246, 11186:11246] = 0
    index = np.argsort(adj, axis=1)
    topk = index[:, -topk:]
    zero_matrix = np.zeros([11246, 11246])
    for i in range(adj.shape[0]):
        zero_matrix[i][topk[i]] = adj[i][topk[i]]
    high = zero_matrix
    high[high.nonzero()[0], high.nonzero()[1]] = 1
    return high

def embedding_Yelp(emb, topk):

    adj = np.matmul(emb, emb.T)
    adj[0:2614,0:2614] = 0
    adj[2614:3900,2614:3900] = 0
    adj[3900:3904,3900:3904] = 0
    adj[3904:3913, 3904:3913] = 0
    adj[0:2614, 0:2614] = 0
    adj[2614:3900, 3900:3904] = 0
    adj[3900:3904, 2614:3900] = 0
    adj[2614:3900, 3904:3913] = 0
    adj[3904:3913, 2614:3900] = 0
    adj[3900:3904, 3904:3913] = 0
    adj[3904:3913, 3900:3904] = 0

    index = np.argsort(adj, axis=1)
    topk = index[:, -topk:]
    zero_matrix = np.zeros([3913,3913])
    for i in range(adj.shape[0]):
        zero_matrix[i][topk[i]] = adj[i][topk[i]]
    high = zero_matrix
    high[high.nonzero()[0], high.nonzero()[1]] = 1
    return high

def embedding_adj_dblp(emb, topk):

    adj = np.matmul(emb, emb.T)
    adj[0:4057, 18385:26108] = 0
    adj[18385:26108, 0:4057] = 0
    adj[0:4057, 26108:26128] = 0
    adj[26108:26128, 0:4057] = 0
    adj[18385:26108, 26108:26128] = 0
    adj[26108:26128, 18385:26108] = 0
    adj[0:4057, 0:4057] = 0
    adj[18385:26108, 18385:26108] = 0
    adj[4057:18385, 4057:18385] = 0
    adj[26108:26128, 26108:26128] = 0

    index = np.argsort(adj, axis=1)
    topk = index[:, -topk:]
    zero_matrix = np.zeros([26128, 26128])
    for i in range(adj.shape[0]):
        zero_matrix[i][topk[i]] = adj[i][topk[i]]
    high = zero_matrix
    high[high.nonzero()[0], high.nonzero()[1]] = 1
    return high

def embedding_adj_imdb(emb, topk):

    adj = np.matmul(emb, emb.T)
    adj[0:4278, 0:4278] = 0
    adj[4278:6359, 4278:6359] = 0
    adj[4278:6359, 6359:11616] = 0
    adj[6359:11616, 4278:6359] = 0
    adj[4278:6359, 6359:11616] = 0
    index = np.argsort(adj, axis=1)
    topk = index[:, -topk:]
    zero_matrix = np.zeros([11616, 11616])
    for i in range(adj.shape[0]):
        zero_matrix[i][topk[i]] = adj[i][topk[i]]
    high = zero_matrix
    high[high.nonzero()[0], high.nonzero()[1]] = 1
    return high