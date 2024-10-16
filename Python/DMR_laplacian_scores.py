import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph 
from scipy.sparse.linalg import expm
from sklearn.manifold import SpectralEmbedding


def compute_laplacian_score(X, neighbour_size = 5):
    S = kneighbors_graph(X, neighbour_size+1, mode='distance',metric='euclidean')
    D = scipy.sparse.diags(np.transpose(np.array((S.sum(axis=1)))), [0])
    #construct graph Laplacian L
    L = D - S.toarray()
    #construct 1= [1,···,1]' 
    n_samples, n_features = np.shape(X)
    I=np.ones((n_samples,n_features))
    #construct fr' => fr= [fr1,...,frn]'
    Xt = np.transpose(X)
    #construct fr^=fr-(frt D I/It D I)I
    t=np.matmul(np.matmul(Xt,D.toarray()),I)/np.matmul(np.matmul(np.transpose(I),D.toarray()),I)
    t=t[:,0]
    t=np.tile(t,(n_samples,1))
    fr=X-t
    #Compute Laplacian Score
    fr_t=np.transpose(fr)
    Lr=np.matmul(np.matmul(fr_t,L),fr)/np.matmul(np.dot(fr_t,D.toarray()),fr)
    return np.diag(Lr)

