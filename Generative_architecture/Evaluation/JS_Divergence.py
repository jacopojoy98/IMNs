import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

def energy_distance(sample1, sample2):
    """Computes the energy distance between two samples."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    n, m = len(sample1), len(sample2)
    d_xx = cdist(sample1, sample1, 'euclidean')
    d_yy = cdist(sample2, sample2, 'euclidean')
    d_xy = cdist(sample1, sample2, 'euclidean')
    
    return 2 * d_xy.mean() - d_xx.mean() - d_yy.mean()

def compute_mmd(sample1, sample2, kernel='rbf', bandwidth=1.0):
    """Computes the Maximum Mean Discrepancy (MMD) between two samples."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    def rbf_kernel(x, y, gamma):
        dists = cdist(x, y, 'sqeuclidean')
        return np.exp(-gamma * dists)

    gamma = 1.0 / (2 * bandwidth**2)
    
    K_xx = rbf_kernel(sample1, sample1, gamma)
    K_yy = rbf_kernel(sample2, sample2, gamma)
    K_xy = rbf_kernel(sample1, sample2, gamma)
    
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def distribution_distance(sample1, sample2, metric='wasserstein', **kwargs):
    """Estimates the distance between two empirical distributions."""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    if metric == 'wasserstein':
        if sample1.ndim == 1:
            return wasserstein_distance(sample1, sample2)
        elif sample1.shape[1] == 1:
            return wasserstein_distance(sample1.ravel(), sample2.ravel())
        else:
            raise ValueError("Wasserstein distance (1D) only supports univariate samples.")
    elif metric == 'energy':
        return energy_distance(sample1, sample2)
    elif metric == 'mmd':
        bandwidth = kwargs.get('bandwidth', 1.0)
        return compute_mmd(sample1, sample2, bandwidth=bandwidth)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compute_probs(data, n=100): 
    data = np.clip(data, a_min = 0, a_max = 100)
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q): 
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples): 
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q): 
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=100): 
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)
    
    return kl_divergence(p, q)

def compute_js_divergence(train, test, n_bins=100): 
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    train_sample = np.array(train)
    test_sample = np.array(test)
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)
    
    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)
    
    return js_divergence(p, q)


def DdD (D1,D2,D3):
    if D3:
        TTTF = distribution_distance(D1,D2)
        TTFF = distribution_distance(D1,D3)
        TFFF = distribution_distance(D3,D2)
        return (TTTF + TTFF + TFFF)/3
    else:
        print("error: D3=False, cause")
        return 0.5
