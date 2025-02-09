import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters for the Gamma distribution and problem setup
alpha_X = 6  # Shape parameter for the Gamma distribution
beta_X = 0.5  # Rate parameter for the Gamma distribution
T = 15  # Number of time periods considered
Y_max = 6  # Maximum inventory level

# Function defining the mean of the price distribution as a function of bid (b)
def mu_b(b):
    return 0.6 * b

# Function defining the probability of winning a bid as a function of bid (b)
def lambda_b(b):
    return 0.8 - 0.7 * np.exp(-b)

# CDF of price (p) given bid (b), modeled using a Gamma distribution
def F_p_given_b(p, b):
    return stats.gamma.cdf((p - mu_b(b)) / sigma_b(b), alpha_X, scale=1 / beta_X)

# sigma_b represents the variability in price as a function of the bid (b)
# It is modeled as (1 + 0.1 * b), increasing with higher bid values
def sigma_b(b):
    return ( 1+ 0.1 * b )

# Initialize the profit matrix for dynamic programming
# Rows represent inventory levels (0 to Y_max), columns represent time periods (0 to T)
pi = np.zeros((Y_max + 1, T + 1))

# Dynamic Programming algorithm to compute the profit matrix
for t in range(1, T + 1):  # Iterate over time periods
    for y in range(1, Y_max + 1):  # Iterate over inventory levels
        # Define the objective function for optimization
        def objective(x):
            b, p = x[0], x[1]  # Decision variables: bid (b) and price (p)
            # Calculate terms for expected profit
            term1 = (1 - lambda_b(b)) * pi[y, t - 1]  # No bid is won
            term2 = lambda_b(b) * F_p_given_b(p, b) * (pi[y, t - 1] - b)  # Bid won but no inventory sold
            term3 = lambda_b(b) * (1 - F_p_given_b(p, b)) * (p + pi[y - 1, t - 1] - b)  # Bid won and inventory sold
            # Return the negative profit (minimization problem)
            return -(term1 + term2 + term3)

        # Optimize over bid (b) and price (p) within bounds
        result = minimize(objective, x0=[1, 1], bounds=[(0, 10), (0, 10)])
        # Store the maximum profit for inventory level y at time t
        pi[y, t] = -result.fun

# Plotting the optimal bid as a function of time for selected inventory levels
plt.figure(figsize=(10, 6))

# Analyze optimal bids for inventory levels 2, 4, and 6
for y in [2, 4, 6]:
    optimal_bids = []  # Store optimal bids for the given inventory level
    for t in range(T, 0, -1):  # Iterate over time periods in reverse
        # Re-solve the optimization problem for each time period
        result = minimize(objective, x0=[1, 1], bounds=[(0, 10), (0, 10)])
        # Append the optimal bid (b) for the current time period
        optimal_bids.append(result.x[0])
    # Plot the results
    plt.plot(range(T, 0, -1), optimal_bids, label=f'Inventory Level = {y}')

# Add titles, labels, legend, and grid to the plot
plt.title('Optimal Bid vs. Time Interval (Inventory Levels 2, 4, 6)')
plt.xlabel('Time Interval (15 to 1)')
plt.ylabel('Optimal Bid')
plt.legend()
plt.grid(True)

# Set x-axis labels from 15 down to 1 and invert the axis direction
plt.xticks(range(15, 0, -1))
plt.gca().invert_xaxis()

# Display the plot
plt.show()