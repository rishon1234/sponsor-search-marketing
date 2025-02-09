import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters for the model
alpha_X = 6  # Shape parameter for the Gamma distribution
beta_X = 0.5  # Scale parameter for the Gamma distribution
T = 15  # Number of periods (time steps) for the optimization
Y_max = 6  # Maximum inventory level to consider

# Functions defining the model dynamics

# mu_b(b): Mean reservation price as a function of the bid (b)
def mu_b(b):
    return 0.4 * b  # Linear relationship between bid and reservation price

# lambda_b(b): Click rate as a function of the bid (b)
def lambda_b(b):
    # Sigmoid function that models the click rate based on the bid
    return (1 + 0.4 * np.exp(4 - 4 * b)) / (1 + np.exp(4 - 4 * b))

# sigma_b(b): Standard deviation of reservation price as a function of the bid (b)
def sigma_b(b):
    return (1 + 0.1 * b)  # Linear relationship for variability

# F_p_given_b(p, b): CDF of the reservation price, representing the probability of purchase
def F_p_given_b(p, b):
    # Gamma CDF normalized by sigma and shifted by mu(b)
    return stats.gamma.cdf((p - mu_b(b)) / sigma_b(b), alpha_X, scale=1 / beta_X)

# Initialize a matrix to store the optimal profits for each inventory level (y) and period (t)
pi = np.zeros((Y_max + 1, T + 1))  # Profit matrix: pi[y, t]

# Dynamic Programming to calculate the optimal profit for each (y, t) pair
for t in range(1, T + 1):  # Iterate through all periods
    for y in range(1, Y_max + 1):  # Iterate through all inventory levels
        # Define the objective function that needs to be maximized (negative profit for minimization)
        def objective(x):
            b, p = x[0], x[1]  # Bid and price as decision variables
            # Term 1: Profit when no customer clicks the link
            term1 = (1 - lambda_b(b)) * pi[y, t-1]
            # Term 2: Profit when a customer clicks and does not purchase (conversion rate < 1)
            term2 = lambda_b(b) * F_p_given_b(p, b) * (pi[y, t-1] - b)
            # Term 3: Profit when a customer clicks and purchases (conversion rate = 1)
            term3 = lambda_b(b) * (1 - F_p_given_b(p, b)) * (p + pi[y-1, t-1] - b)
            # Return negative of total profit (because we are minimizing in `minimize` function)
            return -(term1 + term2 + term3)

        # Use scipy's minimize function to find the optimal bid (b) and price (p) for each (y, t)
        result = minimize(objective, x0=[1, 1], bounds=[(0, None), (0, None)])
        # Store the optimal profit (negate to get positive profit)
        pi[y, t] = -result.fun

# Plotting the optimal bid vs. time interval for specific inventory levels (2, 4, 6)
plt.figure(figsize=(10, 6))  # Set the plot size

# Loop over selected inventory levels to plot the optimal bid
for y in [2, 4, 6]:
    optimal_bids = []  # List to store the optimal bids for the inventory level y
    for t in range(T, 0, -1):  # Iterate through periods in reverse (T to 1)
        # Re-calculate the optimal bid for each period and inventory level
        result = minimize(objective, x0=[1, 1], bounds=[(0, None), (0, None)])
        optimal_bids.append(result.x[0])  # Store the optimal bid for this period
    plt.plot(range(T, 0, -1), optimal_bids, label=f'Inventory Level = {y}')  # Plot the optimal bids

# Title and labels for the plot
plt.title('Optimal Bid vs. Time Interval (Inventory Levels 2, 4, 6)')
plt.xlabel('Time Interval (15 to 1)')
plt.ylabel('Optimal Bid')
plt.legend()  # Add a legend
plt.grid(True)  # Add gridlines for better readability

# Customize x-axis ticks to show time intervals from 15 to 1
plt.xticks(range(15, 0, -1))  # Label the x-axis from 15 down to 1
plt.gca().invert_xaxis()  # Invert the x-axis so that time goes from 15 to 1

plt.show()  # Display the plot