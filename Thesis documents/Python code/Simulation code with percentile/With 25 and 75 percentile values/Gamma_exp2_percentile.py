import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters
alpha_X = 6
beta_X = 0.5
T = 15  # Number of periods
Y_max = 6  # Max inventory level to consider
n_sim = 1000

# Initialize storage
pi_ans = np.zeros((n_sim + 1, T + 1))
price_p_ans = np.zeros((n_sim + 1, T + 1))
bid_b_ans = np.zeros((n_sim + 1, T + 1))
Y_max_ans = np.zeros((n_sim + 1, T + 1))

# Functions (from paper, Example 2)
def mu_b(b):
    return 0.4 * b

def lambda_b(b):
    return (1 + 0.4 * np.exp(4 - 4 * b)) / (1 + np.exp(4 - 4 * b))

def sigma_b(b):
    return 1 + 0.1 * b

def F_p_given_b(p, b):
    z = (p - mu_b(b)) / sigma_b(b)
    z = np.maximum(z, 0)
    return stats.gamma.cdf(z, a=alpha_X, scale=beta_X)

# Dynamic programming matrices
pi = np.zeros((Y_max + 1, T + 1))
bid_b = np.zeros((Y_max + 1, T + 1))
price_p = np.zeros((Y_max + 1, T + 1))
p_of_purchase = np.zeros((Y_max + 1, T + 1))
p_of_click = np.zeros((Y_max + 1, T + 1))

# Optimization loop
for t in range(1, T + 1):
    for y in range(1, Y_max + 1):
        def objective(x):
            b, p = x[0], x[1]
            term1 = (1 - lambda_b(b)) * pi[y, t - 1]
            term2 = lambda_b(b) * F_p_given_b(p, b) * (pi[y, t - 1] - b)
            term3 = lambda_b(b) * (1 - F_p_given_b(p, b)) * (p + pi[y - 1, t - 1] - b)
            return -(term1 + term2 + term3)

        result = minimize(objective, x0=[1.0, 2.0], bounds=[(0, None), (0, None)], method='L-BFGS-B')
        pi[y, t] = round(-result.fun, 2)
        bid_b[y, t] = round(result.x[0], 2)
        price_p[y, t] = round(result.x[1], 2)

# Probabilities
for y in range(1, Y_max + 1):
    for t in range(1, T + 1):
        p_of_purchase[y, t] = 1 - F_p_given_b(price_p[y, t], bid_b[y, t])
        p_of_click[y, t] = lambda_b(bid_b[y, t])

# Simulation
for iter in range(n_sim + 1):
    rows, cols = p_of_purchase.shape
    price_p_ans[iter, -1] = price_p[rows - 1, cols - 1]
    bid_b_ans[iter, -1] = bid_b[rows - 1, cols - 1]
    Y_max_ans[iter, -1] = Y_max
    loop_exited = False

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            r_click = np.random.rand()
            r_purchase = np.random.rand()

            if r_click < p_of_click[i, j]:
                if r_purchase < p_of_purchase[i, j]:
                    pi_ans[iter, j] = price_p[i, j] - bid_b[i, j]
                    j -= 1
                    i -= 1
                    if i >= 0 and j >= 0:
                        price_p_ans[iter, j] = price_p[i, j]
                        bid_b_ans[iter, j] = bid_b[i, j]
                        Y_max_ans[iter, j] = Y_max_ans[iter, j + 1] - 1
                    continue
                else:
                    if j < 0:
                        loop_exited = True
                        break
                    pi_ans[iter, j] = -bid_b[i, j]
                    if j <= 0:
                        loop_exited = True
                        break
                    j -= 1
                    if j >= 0:
                        price_p_ans[iter, j] = price_p[i, j]
                        bid_b_ans[iter, j] = bid_b[i, j]
                        Y_max_ans[iter, j] = Y_max_ans[iter, j + 1]
            else:
                if j < 0:
                    loop_exited = True
                    break
                pi_ans[iter, j] = 0
                if j <= 0:
                    loop_exited = True
                    break
                j -= 1
                if j >= 0:
                    price_p_ans[iter, j] = price_p[i, j]
                    bid_b_ans[iter, j] = bid_b[i, j]
                    Y_max_ans[iter, j] = Y_max_ans[iter, j + 1]
        if i < 0 or loop_exited:
            break

# Plotting function with spacing and colored percentiles
def plot_with_percentiles(data_arrays, titles, y_labels):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    lines, labels = [], []

    for data_array, title, y_label, ax in zip(data_arrays, titles, y_labels, axs.flat):
        data_array = np.array(data_array)
        data_array = np.delete(data_array, 0, axis=1)
        p25 = np.percentile(data_array, 25, axis=0)
        p75 = np.percentile(data_array, 75, axis=0)
        mean_vals = np.mean(data_array, axis=0)
        x_vals = list(range(1, data_array.shape[1] + 1))

        # Plot with distinct colors
        l1, = ax.plot(x_vals, mean_vals, marker='o', color='blue', label="Mean")
        l2, = ax.plot(x_vals, p25, linestyle='--', color='orange', label="25th Percentile")
        l3, = ax.plot(x_vals, p75, linestyle='--', color='green', label="75th Percentile")

        ax.set_title(title)
        ax.set_xlabel("Time Period")
        ax.set_ylabel(y_label)
        ax.set_xticks(x_vals)
        ax.grid()
        ax.invert_xaxis()

        if not lines:
            lines = [l1, l2, l3]
            labels = [line.get_label() for line in lines]

    # Shared legend with padding
    fig.legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

# Plot results
plot_with_percentiles(
    [Y_max_ans, pi_ans, price_p_ans, bid_b_ans],
    ["Inventory Over Time", "Profit Over Time", "Price Over Time", "Bid Over Time"],
    ["Inventory", "Profit", "Price", "Bid"]
)
