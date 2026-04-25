import numpy as np
import copy
from create_data import create_patterns_simple
from compute_MI_singlew import M1M2_w, muvar_w


"""
This is the file that takes in the parameters of all data patterns
and output the MI for one specified connection

There are N data patterns, each follows a multivariate log-normal distribution:
x1 ~ log_normal mu1(d-dim vector)  Sigma1(dxd matrix)
x2 ~ mu2  Sigma2
...
xN ~ mun SigmaN

Now for a weight wij
wij also follows log-normal distribution

wij ~ log_normal mu_wij  Var_wij
For example,
w12 ~ log_normal mu_w12 Var_w12

For the joint activity for n weights 
w (n-d vector) ~ log_normal mu_w (n-d vector) Sigma_w (n-d vector)  
"""



def sigma_ww(data, ia, ja, ib, jb):
    """
    compute the covariance between ln(w_ij(a)) and ln(w_ij(b))
    data: {'x1': {'mu1': , 'Sigma1': }, 'x2': {'mu2': , 'Sigma2': }, 'x3': {'mu3': , 'Sigma3': } ...}
    Note that i j index starts from 1
    """
    numer = 0 # numerator
    denom = 0 # denominator
    for key, value in data.items():
        # Extract the keys for mu and Sigma
        num = ''.join(filter(str.isdigit, key))

        mu_key = f'mu{num}'  # Generate key 'mu1', 'mu2', 'mu3'
        Sigma_key = f'Sigma{num}'  # Generate key 'Sigma1', 'Sigma2', 'Sigma3'

        # Extract mu and Sigma values
        mu = value[mu_key]
        Sigma = value[Sigma_key]

        # add one term to numerator
        numer += (
                np.exp(mu[ia-1] + mu[ja-1] + mu[ib-1] + mu[jb-1] +
                        0.5 * (Sigma[ia-1, ia-1]  + Sigma[ja-1, ja-1] + Sigma[ib-1, ib-1] + Sigma[jb-1, jb-1]) +
                        Sigma[ia-1, ja-1] + Sigma[ib-1, jb-1]) *
                (np.exp(Sigma[ia-1, ib-1] + Sigma[ia-1, jb-1] + Sigma[ja-1, ib-1] + Sigma[ja-1, jb-1]) - 1)
                  )
        for _key, _value in data.items():
            # Extract the keys for mu and Sigma
            _num = ''.join(filter(str.isdigit, _key))

            _mu_key = f'mu{_num}'  # Generate key 'mu1', 'mu2', 'mu3'
            _Sigma_key = f'Sigma{_num}'  # Generate key 'Sigma1', 'Sigma2', 'Sigma3'

            # Extract mu and Sigma values
            _mu = _value[_mu_key]
            _Sigma = _value[_Sigma_key]

            # add one term to denominator
            denom += np.exp(mu[ia-1] + mu[ja-1] + _mu[ib-1] + _mu[jb-1] +
                            0.5 * (Sigma[ia-1, ia-1]  + Sigma[ja-1, ja-1] + _Sigma[ib-1, ib-1] + _Sigma[jb-1, jb-1]) +
                            Sigma[ia-1, ja-1] + _Sigma[ib-1, jb-1])
    # result
    sigma_ww = np.log( 1 + numer / denom)
    return sigma_ww

def is_positive_semi_definite(matrix):
    # Calculate the eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Check if all eigenvalues are non-negative
    return np.all(eigenvalues >= 0.)


def MI_nw_xl(data, w_list, l):
    """
    Compute the MI between joint activity of a list of connections and x^l
    w_list: [(1, 2), (3, 5), ....]
    Note that indices for neurons and connections start from 1
    """
    seen_tuples = set()

    for t in w_list:
        if t in seen_tuples or tuple(reversed(t)) in seen_tuples:
            raise ValueError(f"Duplicate or reverse tuple found: {t}")
        seen_tuples.add(t)

    # number of weights
    n = len(w_list)
    # mean and covariance matrix for distribution of ln(w_list)
    mu_nw = np.zeros(n)
    Sigma_nw = np.zeros((n, n))

    for a, ij_a in enumerate(w_list):
        ia, ja = ij_a[0], ij_a[1]
        mu_nw[a], _ = muvar_w(data, ia, ja) # specifiy elements in mu_nw
        for b, ij_b in enumerate(w_list):
            ib, jb = ij_b[0], ij_b[1]
            Sigma_nw[a, b] = sigma_ww(data, ia, ja, ib, jb) # specifiy elements in Sigma_nw


    # same thing, but now with pattern xl deleted
    data_copy = copy.deepcopy(data)
    del data_copy['x' + str(l)]

    mu_nw_minus_l = np.zeros(n)
    Sigma_nw_minus_l = np.zeros((n, n))

    for a, ij_a in enumerate(w_list):
        ia, ja = ij_a[0], ij_a[1]
        mu_nw_minus_l[a], _ = muvar_w(data_copy, ia, ja)
        for b, ij_b in enumerate(w_list):
            ib, jb = ij_b[0], ij_b[1]
            Sigma_nw_minus_l[a, b] = sigma_ww(data_copy, ia, ja, ib, jb)


    # compute the mutual information, in nats
    # sigma_nw might not be semi-positive definite, i.e. having negative eigenvalues
    # I believe this is due to the approximation error in theory setups.
    ratio = np.linalg.det(Sigma_nw) / np.linalg.det(Sigma_nw_minus_l)
    MI = np.sum(mu_nw - mu_nw_minus_l) + 0.5 * (np.log(ratio))
    return MI




if __name__ == "__main__":
    """
    data = create_patterns(20, 5, 1, 0.9)
    #data = create_patterns_simple(3, 3, 1, 5, -0.2)
    #print(data)
    M1, M2 = M1M2_w(data, 1, 2)
    print(M1, M2)
    muw, sigma2w = muvar_w(data, 3, 2)
    print(muw, sigma2w)
    s_ww = sigma_ww(data, 2, 3, 1, 3)
    print(s_ww)
    w_list = [(1, 2), (1, 3), (2, 3), (1, 4)]
    MI = MI_nw_xl(data, w_list, 3)
    print(MI)
    """
    # Set the precision to 3 digits
    np.set_printoptions(precision=3)
    ws = [(2, 3), (4, 5), (2, 4), (2, 5), (6, 7), (8, 9), (3, 4), (3, 5)]
    data = create_patterns_simple(10, 20, 1, 1, 0.8)
    MI = MI_nw_xl(data, ws, 1)
    print(MI)


