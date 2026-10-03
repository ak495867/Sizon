import numpy as np
from scipy.cluster.hierarchy import linkage

def get_quasi_diag(link: np.ndarray) -> list[int]:
    """Return sorted list of original items based on linkage matrix."""
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    n_items = len(link) + 1

    while True:
        has_cluster = False
        new_sort_ix = []
        for i in sort_ix:
            if i < n_items:
                new_sort_ix.append(i)
            else:
                has_cluster = True
                new_sort_ix.append(link[i - n_items, 0])
                new_sort_ix.append(link[i - n_items, 1])
        sort_ix = new_sort_ix
        if not has_cluster:
            break
    return sort_ix

def get_cluster_var(cov: np.ndarray, c_items: list[int]) -> float:
    """Compute cluster variance using inverse variance portfolio."""
    cov_slice = cov[np.ix_(c_items, c_items)]
    # Protect against divide-by-zero for zero variance
    ivp = 1.0 / np.maximum(np.diag(cov_slice), 1e-12)
    ivp /= ivp.sum()
    return np.dot(np.dot(ivp, cov_slice), ivp)

def get_rec_bipart(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Compute HRP weights using recursive bisection."""
    w = np.ones(len(cov))
    c_items = [sort_ix]
    
    while len(c_items) > 0:
        c_items_new = []
        for i in c_items:
            if len(i) > 1:
                split = len(i) // 2
                c_items_new.append(i[:split])
                c_items_new.append(i[split:])
        if not c_items_new:
            break

        for i in range(0, len(c_items_new), 2):
            c0 = c_items_new[i]
            c1 = c_items_new[i+1]
            
            v0 = get_cluster_var(cov, c0)
            v1 = get_cluster_var(cov, c1)
            
            # Inverse variance weights
            alpha = v1 / max(v0 + v1, 1e-12)
            
            w[c0] *= alpha
            w[c1] *= (1 - alpha)
            
        c_items = c_items_new
        
    return w

def hrp_weights(cov_matrix: list[list[float]]) -> list[float]:
    """
    Computes Hierarchical Risk Parity weights for a given covariance matrix.
    Args:
        cov_matrix: A 2D list representing the covariance matrix.
    Returns:
        A list of optimal weights.
    """
    if not cov_matrix or not cov_matrix[0]:
        return []

    cov = np.array(cov_matrix)
    n = len(cov)
    
    if n == 1:
        return [1.0]

    # Convert covariance to correlation matrix
    vols = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    outer_vols = np.outer(vols, vols)
    corr = cov / np.maximum(outer_vols, 1e-12)
    
    # Clip correlations to [-1, 1] for numerical stability
    corr = np.clip(corr, -1.0, 1.0)

    # Distance matrix: D = sqrt(0.5 * (1 - rho))
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    
    # Extract condensed distance matrix manually to avoid scipy warnings/errors
    # if dist matrix doesn't perfectly satisfy distance matrix constraints.
    condensed_dist = dist[np.triu_indices(n, k=1)]
    
    # Perform linkage
    link = linkage(condensed_dist, method='single')
    
    # Quasi-diagonalization
    sort_ix = get_quasi_diag(link)
    
    # Recursive bisection
    weights = get_rec_bipart(cov, sort_ix)
    
    return weights.tolist()
