import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------ Parameters (from paper) ------------------
alpha_X = 6      # Gamma shape parameter
beta_X = 0.5     # Gamma scale parameter
T = 15           # Number of time periods
Y_max = 6        # Maximum inventory level

# ------------------ Functions for Example 2 ------------------
def mu_b(b):
    return 0.4 * b  # Paper: Example 2

def lambda_b(b):
    return (1 + 0.4 * np.exp(4 - 4 * b)) / (1 + np.exp(4 - 4 * b))  # Paper: Example 2

def sigma_b(b):
    return 1 + 0.1 * b  # Shared across both examples

def F_p_given_b(p, b):
    z = (p - mu_b(b)) / sigma_b(b)
    z = np.maximum(z, 0)  # avoid invalid values
    return stats.gamma.cdf(z, a=alpha_X, scale=beta_X)



# ------------------ Initialization ------------------
pi = np.zeros((Y_max + 1, T + 1))      # π(y, t): expected profit
b_star = np.zeros((Y_max + 1, T + 1))  # Store optimal bids

# ------------------ Terminal Condition (from paper) ------------------
pi[:, 0] = 0  # Inventory has zero value at time 0

# ------------------ Dynamic Programming ------------------
for t in range(1, T + 1):
    for y in range(1, Y_max + 1):
        def objective(x):
            b, p = x[0], x[1]
            f = F_p_given_b(p, b)
            term1 = (1 - lambda_b(b)) * pi[y, t - 1]
            term2 = lambda_b(b) * f * (pi[y, t - 1] - b)
            term3 = lambda_b(b) * (1 - f) * (p + pi[y - 1, t - 1] - b)
            return -(term1 + term2 + term3)

        result = minimize(
            objective,
            x0=[1, 2],  # Stable initial guess
            bounds=[(0.01, None), (0.01, None)],
            method='L-BFGS-B'
        )

        pi[y, t] = -result.fun
        b_star[y, t] = result.x[0]  # Store optimal bid

# ------------------ Plotting ------------------
plt.figure(figsize=(10, 6))
for y in [2, 4, 6]:
    plt.plot(range(T, 0, -1), b_star[y, T:0:-1], label=f'Inventory Level = {y}')


plt.title('Example 2 \nOptimal Bidding Strategy')
plt.xlabel('Number of remaining periods')
plt.ylabel('Optimal Bid')
plt.legend()
plt.grid(True)
plt.xticks(range(15, 0, -1))
plt.gca().invert_xaxis()
plt.show()

