import numpy as np


"""
This is the file that defines functions that generate data distributions

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

def cov_eigenvalue_method(dim, range_s=1.0):
    """
    Generate a random covariance matrix using the random eigenvalue method
    """
    # Step 1: Generate random eigenvalues
    eigenvalues = np.abs(np.random.normal(0.01, range_s, dim))

    # Step 2: Generate a random orthogonal matrix using QR decomposition
    random_matrix = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(random_matrix)

    # Step 3: Construct the covariance matrix
    covariance_matrix = Q @ np.diag(eigenvalues) @ Q.T
    return covariance_matrix


def create_patterns_general_eigen(N, d, range_mu, range_s):
    """
    Create a dictionary with N random variables, each with dimension d
    mu are sampled from uniformly
    covariance matrix are sampled using: eigenvalue method + random factor * ones

    Different patterns have different mu but the still the same Sigma
    e.g.
    {'x1': {'mu1': , 'Sigma1': }, 'x2': {'mu2': , 'Sigma2': }, 'x3': {'mu3': , 'Sigma3': }}
    """
    # Initialize the dictionary
    data = {}

    # Populate the dictionary
    for i in range(1, N + 1):
        key = f'x{i}'  # Create the key, e.g., 'x1', 'x2', etc.

        cov_matrix = cov_eigenvalue_method(d, range_s)
        # Create an all-ones matrix
        ones_matrix = np.ones((d, d))
        # Add the scaled all-ones matrix to the original covariance matrix
        cov_matrix = cov_matrix + np.random.uniform(0, range_s ** 2) * ones_matrix

        mean = np.random.uniform(-range_mu, range_mu, d)

        data[key] = {
            f'mu{i}': mean,
            f'Sigma{i}': cov_matrix
        }
    return data



def create_patterns_general_ATA(N, d, range_mu, range_s):
    """
    Create a dictionary with N random variables, each with dimension d
    mu are sampled from uniformly
    covariance matrix are sampled using: A.T A

    Different patterns have different mu but the still the same Sigma
    e.g.
    {'x1': {'mu1': , 'Sigma1': }, 'x2': {'mu2': , 'Sigma2': }, 'x3': {'mu3': , 'Sigma3': }}
    """
    # Initialize the dictionary
    data = {}

    # Populate the dictionary
    for i in range(1, N + 1):
        key = f'x{i}'  # Create the key, e.g., 'x1', 'x2', etc.

        A = np.random.normal(0, range_s, (d, d))
        cov_matrix = A.T @ A

        mean = np.random.uniform(-range_mu, range_mu, d)

        data[key] = {
            f'mu{i}': mean,
            f'Sigma{i}': cov_matrix
        }
    return data


def create_patterns_samecov(N, d, range_mu, range_s):
    """
    Create a dictionary with N random variables, each with dimension d
    mu are sampled from uniformly
    covariance matrix are sampled randomly as exchangeable covariance matrix + ATA/ d
    But now the covariance matrices are the same across patterns

    Different patterns have different mu but the still the same Sigma
    e.g.
    {'x1': {'mu1': , 'Sigma1': }, 'x2': {'mu2': , 'Sigma2': }, 'x3': {'mu3': , 'Sigma3': }}
    """
    # Initialize the dictionary
    data = {}

    # randomly generate rho and sigma
    rho = np.random.uniform(-1 / (d-1), 1.0)
    sigma = np.abs(np.random.normal(0, range_s))

    # Generate the dxd covariance matrix
    cov_matrix = np.full((d, d), rho * sigma ** 2)  # Fill with rho*sigma^2
    np.fill_diagonal(cov_matrix, sigma ** 2)  # Set the diagonal elements to sigma^2
    A = np.random.normal(0, range_s, (d, d))
    cov_matrix = cov_matrix + A.T @ A / d

    # Populate the dictionary
    for i in range(1, N + 1):
        key = f'x{i}'  # Create the key, e.g., 'x1', 'x2', etc.

        mean = np.random.uniform(-range_mu, range_mu, d)

        data[key] = {
            f'mu{i}': mean,
            f'Sigma{i}': cov_matrix
        }
    return data


def create_patterns_simple(N, d, range_mu, sigma, rho):
    """
    Create a dictionary with N random variables, each with dimension d
    mu are sampled from uniformly
    covariance matrix are the same for all variables.
    covariance matrix is exchangeable covariance matrix, with the form like:
    range_mu^2 * [sigma^2         rho sigma^2]
                 [rho * sigma^2       sigma^2]
    -1 / (d-1)   =<   rho   <= 1
    """
    # initialize the parameters
    data = {}

    # Check the valid range of rho
    if not (-1 / (d - 1) <= rho <= 1):
        raise ValueError(f"Invalid value for rho: {rho}. It must be between {-1 / (d - 1)} and 1 for d = {d}.")

    # Generate the dxd covariance matrix
    cov_matrix = np.full((d, d), rho * sigma ** 2)  # Fill with rho*sigma^2
    np.fill_diagonal(cov_matrix, sigma ** 2)  # Set the diagonal elements to sigma^2
    cov_matrix = range_mu ** 2 * cov_matrix

    # Populate the dictionary
    # make sure each time the function generates the same set of different mean values
    np.random.seed(34)  # 34 goodlooking; random seed 42 & 47 & 52 will lead to different curve patterns in plots than others
    for i in range(1, N + 1):
        key = f'x{i}'  # Create the key, e.g., 'x1', 'x2', etc.

        mean = np.random.uniform(-range_mu, range_mu, d)

        data[key] = {
            f'mu{i}': mean,
            f'Sigma{i}': cov_matrix
        }
    np.random.seed()
    return data


if __name__ == "__main__":
    data = create_patterns(10, 20, 10, 0.1)
    # data = create_patterns_simple(3, 3, 1, 5, -0.2)
    # Set the precision to 3 digits
    np.set_printoptions(precision=3)
    print(data)










