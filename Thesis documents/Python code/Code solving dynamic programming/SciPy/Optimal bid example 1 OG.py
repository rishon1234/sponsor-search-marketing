import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------ Parameters ------------------
alpha_X = 6
beta_X = 0.5
T = 15
Y_max = 6

# ------------------ Model Functions (Example 1) ------------------
def mu_b(b):
    return 0.6 * b

def lambda_b(b):
    return 0.8 - 0.7 * np.exp(-b)

def F_p_given_b(p, b):
    z = (p - mu_b(b)) / (1 + 0.1 * b)
    return stats.gamma.cdf(z, a=6, scale=0.5)



# ------------------ Initialization ------------------
pi = np.zeros((Y_max + 1, T + 1))      # π(y, t): expected profit
b_star = np.zeros((Y_max + 1, T + 1))  # Store optimal bids

# Enforce terminal condition explicitly: π(y, 0) = 0
pi[:, 0] = 0

# ------------------ Dynamic Programming ------------------
for t in range(1, T + 1):
    for y in range(1, Y_max + 1):
        def objective(x):
            b, p = x
            f = F_p_given_b(p, b)
            term1 = (1 - lambda_b(b)) * pi[y, t - 1]
            term2 = lambda_b(b) * f * (pi[y, t - 1] - b)
            term3 = lambda_b(b) * (1 - f) * (p + pi[y - 1, t - 1] - b)
            return -(term1 + term2 + term3)

        result = minimize(
    objective,
    x0=[1, 1],  # tuned initial guess
    bounds=[(0.01, None), (0.01, None)],
    method='L-BFGS-B',
    options={'maxiter': 100, 'ftol': 1e-9}
)

        pi[y, t] = -result.fun
        b_star[y, t] = result.x[0]

# ------------------ Plotting ------------------
plt.figure(figsize=(10, 6))
for y in [2, 4, 6]:
    plt.plot(range(T, 0, -1), b_star[y, T:0:-1], label=f'Inventory Level = {y}')

plt.title('Example 1 \nOptimal Bidding Strategy')
plt.xlabel('Number of remaining periods')
plt.ylabel('Optimal Bid')
plt.legend()
plt.grid(True)
plt.xticks(range(15, 0, -1))
plt.gca().invert_xaxis()
plt.show()
