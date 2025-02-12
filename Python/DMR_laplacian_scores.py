
import numpy as np
from sklearn.neighbors import kneighbors_graph
import scipy.sparse

def compute_laplacian_score(X, neighbor_size=5):
    """
    Computes the Laplacian Score for each feature in X.
    
    Parameters:
      X : array-like of shape (n_samples, n_features)
          The input data.
      neighbor_size : int, default=5
          The number of nearest neighbors (plus one) to use when constructing the neighborhood graph.
    
    Returns:
      laplacian_scores : array of shape (n_features,)
          The Laplacian Score for each feature.
    """
    n_samples, n_features = X.shape

    # Construct the k-nearest neighbor graph.
    # Note: kneighbors_graph returns a sparse matrix (n_samples x n_samples).
    S = kneighbors_graph(X, neighbor_size + 1, mode='distance', metric='euclidean')

    # Compute the degree matrix D (as a sparse diagonal matrix)
    # S.sum(axis=1) returns a (n_samples x 1) matrix.
    D = scipy.sparse.diags(np.array(S.sum(axis=1)).flatten())

    # Compute the unnormalized graph Laplacian: L = D - S.
    # (We later convert to dense for the arithmetic on features.)
    L = D - S

    # Convert D and L to dense arrays.
    D_dense = D.toarray()
    L_dense = L.toarray()

    # Compute the weighted mean for each feature.
    # Use a ones vector (n_samples x 1) rather than an (n_samples x n_features) ones matrix.
    ones = np.ones((n_samples, 1))
    # weighted_mean: shape (1, n_features)
    weighted_mean = (ones.T @ D_dense @ X) / (ones.T @ D_dense @ ones)

    # Center each feature by subtracting its weighted mean.
    # Broadcasting: weighted_mean (1 x n_features) is subtracted from X (n_samples x n_features)
    X_centered = X - weighted_mean

    # Compute the numerator and denominator for the Laplacian Score for each feature.
    # Numerator: f^T L f  and Denom: f^T D f, where f is each centered feature vector.
    # We compute these for all features in a vectorized fashion.
    numerator = np.sum(X_centered * (L_dense @ X_centered), axis=0)
    denominator = np.sum(X_centered * (D_dense @ X_centered), axis=0)

    # Avoid division by zero.
    denominator[denominator == 0] = 1e-12

    laplacian_scores = numerator / denominator
    return laplacian_scores