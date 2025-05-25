import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# ------------------ Parameters (Example 2) ------------------
alpha_X = 6
beta_X = 0.5
T = 15
Y_max = 6

# ------------------ Model Functions (Example 2) ------------------
def mu_b(b):
    return 0.4 * b

def sigma_b(b):
    return 1 + 0.1 * b

def lambda_b(b):
    return (1 + 0.4 * np.exp(4 - 4 * b)) / (1 + np.exp(4 - 4 * b))

def F_p_given_b(p, b):
    z = (p - mu_b(b)) / sigma_b(b)
    z = np.maximum(z, 0)  # Ensure input is non-negative
    return gamma.cdf(z, a=alpha_X, scale=beta_X)

# ------------------ Grid Search Implementation ------------------
bid_grid = np.linspace(0.01, 1.5, 150)    # more fine-grained bids
price_grid = np.linspace(0.5, 3.5, 200)   # tighter, realistic price band


pi = np.zeros((Y_max + 1, T + 1))
b_star = np.zeros((Y_max + 1, T + 1))

pi[:, 0] = 0  # Terminal condition

for t in range(1, T + 1):
    for y in range(1, Y_max + 1):
        best_profit = -np.inf
        best_b = 2
        for b in bid_grid:
            for p in price_grid:
                f = F_p_given_b(p, b)
                term1 = (1 - lambda_b(b)) * pi[y, t - 1]
                term2 = lambda_b(b) * f * (pi[y, t - 1] - b)
                term3 = lambda_b(b) * (1 - f) * (p + pi[y - 1, t - 1] - b)
                profit = term1 + term2 + term3
                if profit > best_profit:
                    best_profit = profit
                    best_b = b
        pi[y, t] = best_profit
        b_star[y, t] = best_b

# ------------------ Plotting ------------------
plt.figure(figsize=(10, 6))
for y in [2, 4, 6]:
    plt.plot(range(T, 0, -1), b_star[y, T:0:-1], label=f'Inventory Level = {y}')

plt.title('Example 2 \nOptimal Bidding Strategy(Grid search)')
plt.xlabel('Number of remaining periods')
plt.ylabel('Optimal Bid')
plt.legend()
plt.grid(True)
plt.xticks(range(15, 0, -1))
plt.gca().invert_xaxis()
plt.show()

