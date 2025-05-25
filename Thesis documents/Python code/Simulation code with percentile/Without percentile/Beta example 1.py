import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters
alpha_X = 3
beta_X = 7
T = 15  # Number of periods
Y_max = 6  # Max inventory level to consider
n_sim = 1000
pi_ans = np.zeros((n_sim + 1, T + 1))
price_p_ans = np.zeros((n_sim + 1, T + 1))
bid_b_ans = np.zeros((n_sim + 1, T + 1))
Y_max_ans = np.zeros((n_sim + 1, T + 1))

# Functions
def mu_b(b):
    return 0.6 * b

def lambda_b(b):
    return 0.8 - 0.7 * np.exp(-b)

def sigma_b(b):
    return 1 + 0.1 * b

def F_p_given_b(p, b):
    z = (p - mu_b(b)) / (10 * sigma_b(b))
    z = np.clip(z, 0, 1)
    return stats.beta.cdf(z, a=alpha_X, b=beta_X)

def average_of_columns(array):
    np_array = np.array(array)
    return np.mean(np_array, axis=0).tolist()

def plot_array(array, title, x_label, y_label, ax):
    n = len(array)
    x_values = list(range(1, n))
    array.pop(0)
    ax.plot(x_values, array, marker='o')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    ax.grid()
    ax.invert_xaxis()

# Initialize matrices
pi = np.zeros((Y_max + 1, T + 1))
bid_b = np.zeros((Y_max + 1, T + 1))
price_p = np.zeros((Y_max + 1, T + 1))
p_of_purchase = np.zeros((Y_max + 1, T + 1))
p_of_click = np.zeros((Y_max + 1, T + 1))

# Fix #1: Define objective inside loop
for t in range(1, T + 1):
    for y in range(1, Y_max + 1):
        def objective(x):
            b, p = x[0], x[1]
            term1 = (1 - lambda_b(b)) * pi[y, t - 1]
            term2 = lambda_b(b) * F_p_given_b(p, b) * (pi[y, t - 1] - b)
            term3 = lambda_b(b) * (1 - F_p_given_b(p, b)) * (p + pi[y - 1, t - 1] - b)
            return -(term1 + term2 + term3)

        result = minimize(objective, x0=[1, 1], bounds=[(0, None), (0, None)])
        pi[y, t] = round(-result.fun, 2)
        bid_b[y, t] = round(result.x[0], 2)
        price_p[y, t] = round(result.x[1], 2)

# Compute probabilities
for row in range(1, Y_max + 1):
    for col in range(1, T + 1):
        p_of_purchase[row, col] = 1 - F_p_given_b(price_p[row, col], bid_b[row, col])
        p_of_click[row, col] = lambda_b(bid_b[row, col])

# Simulate
for iter in range(n_sim + 1):
    rows, cols = p_of_purchase.shape
    price_p_ans[iter, -1] = price_p[rows - 1, cols - 1]
    bid_b_ans[iter, -1] = bid_b[rows - 1, cols - 1]
    Y_max_ans[iter, -1] = Y_max
    loop_exited = False

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            random_num = np.random.rand()
            random_num1 = np.random.rand()

            if random_num < p_of_click[i, j]:
                if random_num1 < p_of_purchase[i, j]:
                    pi_ans[iter, j] = price_p[i, j] - bid_b[i, j]
                    j -= 1
                    i -= 1
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
                price_p_ans[iter, j] = price_p[i, j]
                bid_b_ans[iter, j] = bid_b[i, j]
                Y_max_ans[iter, j] = Y_max_ans[iter, j + 1]

        if i < 0 or loop_exited:
            break

# Averages
pi_ans_avg = average_of_columns(pi_ans)
Y_max_ans_avg = average_of_columns(Y_max_ans)
price_p_ans_avg = average_of_columns(price_p_ans)
bid_b_ans_avg = average_of_columns(bid_b_ans)

# Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
plot_array(Y_max_ans_avg, "Average Inventory Over Time", "Time Period", "Average Inventory", axs[0, 0])
plot_array(pi_ans_avg, "Average Profit Over Time", "Time Period", "Profit", axs[0, 1])
plot_array(price_p_ans_avg, "Average Price Over Time", "Time Period", "Price", axs[1, 0])
plot_array(bid_b_ans_avg, "Average Bid Over Time", "Time Period", "Bid", axs[1, 1])
plt.tight_layout()
plt.show()
