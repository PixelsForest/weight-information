import numpy as np
import copy
from create_data import *


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


def M1M2_w(data, i, j):
    """
    data: a dictionary for distributions
    compute the value of M1 and M2 for the connection w_ij
    i j index starts from 1
    """
    # Check the i != j
    if not i != j:
        raise ValueError(f"Invalid value for i, j: {i, j}. They cannot be the same.")

    M1 = 0
    M2 = 0
    # Loop over the dictionary to read out the mu and Sigma
    for key, value in data.items():
        # Extract the keys for mu and Sigma
        num = ''.join(filter(str.isdigit, key))

        mu_key = f'mu{num}'  # Generate key 'mu1', 'mu2', 'mu3'
        Sigma_key = f'Sigma{num}'  # Generate key 'Sigma1', 'Sigma2', 'Sigma3'

        # Extract mu and Sigma values
        mu = value[mu_key]
        Sigma = value[Sigma_key]

        M1 += np.exp(mu[i-1] + mu[j-1] +
                     0.5 * (Sigma[i-1, i-1] + Sigma[j-1, j-1]) +
                     Sigma[i-1, j-1])
        M2 += ((np.exp(Sigma[i-1,i-1] + Sigma[j-1,j-1] + 2* Sigma[i-1,j-1]) - 1) *
               np.exp(2*(mu[i-1] + mu[j-1]) + Sigma[i-1,i-1] + Sigma[j-1,j-1] + 2* Sigma[i-1,j-1]))
    return M1, M2


def muvar_w(data, i, j):
    """
    commpute the mu_w and sigma^2_w for the connection w_ij
    """
    M1, M2 = M1M2_w(data, i, j)
    muw = np.log(M1) - 0.5 * np.log(1 + M2 / M1 ** 2)
    sigma2w = np.log(1 + M2 / M1 ** 2)
    return muw, sigma2w


def MI_wijxl(data, i, j, l):
    """
    compute mutual information one connection has about one pattern, i.e. MI(w_ij;xl)
    MI in nats
    """
    muw, sigma2w = muvar_w(data, i, j)
    data_copy = copy.deepcopy(data)
    del data_copy['x'+str(l)]
    muw_minus_l, sigma2w_minus_l = muvar_w(data_copy, i, j)
    MI = muw - muw_minus_l + 0.5 * np.log(sigma2w) - 0.5 * np.log(sigma2w_minus_l)
    return MI




if __name__ == "__main__":
    data = create_patterns(20, 3, 1, 0.9)
    #data = create_patterns_simple(3, 3, 1, 5, -0.2)
    # Set the precision to 3 digits
    np.set_printoptions(precision=3)
    #print(data)
    M1, M2 = M1M2_w(data, 1, 2)
    print(M1, M2)
    muw, sigma2w = muvar_w(data, 1, 2)
    print(muw, sigma2w)
    MI = MI_wijxl(data, 1, 2, 2)
    print(MI)
