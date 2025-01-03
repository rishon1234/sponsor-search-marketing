
# Dynamic Pricing and Inventory Optimization

This project implements a dynamic programming approach to optimize pricing and inventory decisions over a series of time intervals. The objective is to maximize the total profit while considering inventory levels and dynamic pricing factors. This work aims to recreate the graph presented in the research paper *"Sponsored Search Marketing: Dynamic Pricing and Advertising for an Online Retailer"* by Shengqi Ye, Goker Aydin, and Shanshan Hu, as part of the first step in my thesis and simulation efforts.

## Features

- **Dynamic Programming**: Computes the optimal profit matrix using recursive relations.
- **Inventory Levels**: Considers inventory levels ranging from 0 to a specified maximum.
- **Pricing Dynamics**: Incorporates dynamic pricing functions based on bid (`b`) and price (`p`).
- **Optimization**: Utilizes numerical optimization techniques to determine the optimal bid and price at each time step.
- **Visualization**: Plots the optimal bid against time intervals for selected inventory levels.

## Code Structure

- **Parameter Initialization**: Defines the initial parameters for the problem, including the gamma distribution parameters and inventory constraints.
- **Dynamic Programming Logic**: Calculates the profit matrix (`pi`) for all inventory levels and time intervals.
- **Optimization Function**: Uses `scipy.optimize.minimize` to find optimal `b` (bid) and `p` (price) values at each step.
- **Plotting**: Visualizes the results for selected inventory levels.

## Constraints

- Bid (`b`) and price (`p`) are limited to a maximum of 10. This constraint is enforced through the bounds in the optimization process.
- Inventory levels are capped at the defined `Y_max`.

## Usage

1. **Run the Script**: Execute the Python script to calculate the profit matrix and generate the plots.

   ```bash
   python dynamic_pricing_optimization.py
   ```

2. **Results**: View the plots showing the optimal bid over time for different inventory levels.

## Dependencies

- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`

## Example Output

The script generates a plot of optimal bids versus time intervals for inventory levels 2, 4, and 6. The x-axis represents the time intervals (in reverse order), and the y-axis shows the optimal bid values.

## License

This project is open-source and available under the MIT License.

---

Feel free to modify and adapt the code for your specific needs. Contributions are welcome!
