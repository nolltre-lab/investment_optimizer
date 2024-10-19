import numpy as np
import matplotlib.pyplot as plt
import argparse

# 1. Parse Command-Line Arguments

parser = argparse.ArgumentParser(description='Investment Opportunities Optimizer')

parser.add_argument('--fixed_dimension', type=str, required=True,
                    choices=['purchase_amount', 'down_payment', 'term_length', 'interest_rate'],
                    help='Dimension to fix for plotting.')

parser.add_argument('--fixed_value', type=float, required=True,
                    help='Value at which to fix the selected dimension.')

# Optional parameter ranges
parser.add_argument('--purchase_min', type=float, default=1000,
                    help='Minimum purchase amount.')
parser.add_argument('--purchase_max', type=float, default=300000,
                    help='Maximum purchase amount.')
parser.add_argument('--purchase_steps', type=int, default=20,
                    help='Number of steps in purchase amount range.')

parser.add_argument('--down_payment_min', type=float, default=0.0,
                    help='Minimum down payment percentage (0.0 to 1.0).')
parser.add_argument('--down_payment_max', type=float, default=0.5,
                    help='Maximum down payment percentage (0.0 to 1.0).')
parser.add_argument('--down_payment_steps', type=int, default=20,
                    help='Number of steps in down payment percentage range.')

parser.add_argument('--term_min', type=float, default=6,
                    help='Minimum term length in months.')
parser.add_argument('--term_max', type=float, default=36,
                    help='Maximum term length in months.')
parser.add_argument('--term_steps', type=int, default=12,
                    help='Number of steps in term length range.')

parser.add_argument('--interest_min', type=float, default=0.00,
                    help='Minimum interest rate (as a decimal, e.g., 0.05 for 5%).')
parser.add_argument('--interest_max', type=float, default=0.20,
                    help='Maximum interest rate (as a decimal).')
parser.add_argument('--interest_steps', type=int, default=20,
                    help='Number of steps in interest rate range.')

# Optional parameters for customer's desired point
parser.add_argument('--customer_purchase_amount', type=float,
                    help="Customer's desired purchase amount.")
parser.add_argument('--customer_down_payment', type=float,
                    help="Customer's desired down payment amount ($).")
parser.add_argument('--customer_term_length', type=float,
                    help="Customer's desired term length in months.")
parser.add_argument('--customer_interest_rate', type=float,
                    help="Customer's desired interest rate (as a decimal, e.g., 0.05 for 5%).")

# Flag to find the closest acceptable point
parser.add_argument('--find_closest', action='store_true',
                    help="Find the closest acceptable point to the customer's desired combination if it is not acceptable.")

args = parser.parse_args()

# 2. Define Parameters

# Base probability of default
P_base = 0.01  # 1% base default probability

# Sensitivity coefficients
alpha = 0.0001  # Sensitivity to effective loan amount
beta = 0.01     # Sensitivity to term length

# Base loan amount and term length
L_base = 5000   # Base loan amount in dollars
T_base = 12     # Base term length in months

# Base customer interest rate
r_c_base = 0.10  # 10%

# Effective Loan Amount at base values
E_L_base = L_base * (1 + r_c_base * T_base / 12)

# Cost of funds rate
r_f = 0.05  # 5% annual cost of funds

# Processing revenue rate (percentage of purchase amount)
R_t_rate = 0.02  # 2% of the purchase amount

# Risk appetite
R_max = 300  # Maximum acceptable risk in dollars

# Minimum acceptable profit margin (percentage)
P_margin_min = 0  # Minimum acceptable profit margin

# 3. Generate Parameter Space

# Generate parameter ranges
P_values = np.linspace(args.purchase_min, args.purchase_max, args.purchase_steps)
D_percentage_values = np.linspace(args.down_payment_min, args.down_payment_max, args.down_payment_steps)
T_values = np.linspace(args.term_min, args.term_max, args.term_steps)
r_c_values = np.linspace(args.interest_min, args.interest_max, args.interest_steps)

# Include customer's desired point in parameter ranges if provided
if args.customer_purchase_amount is not None:
    if args.customer_purchase_amount < args.purchase_min or args.customer_purchase_amount > args.purchase_max:
        print(f"Customer's purchase amount {args.customer_purchase_amount} is outside the purchase amount range ({args.purchase_min} to {args.purchase_max}).")
        exit()
    if not np.isclose(P_values, args.customer_purchase_amount).any():
        P_values = np.sort(np.append(P_values, args.customer_purchase_amount))

if args.customer_down_payment is not None:
    D_percentage_customer = args.customer_down_payment / args.customer_purchase_amount
    if D_percentage_customer < args.down_payment_min or D_percentage_customer > args.down_payment_max:
        print(f"Customer's down payment percentage {D_percentage_customer} is outside the down payment percentage range ({args.down_payment_min} to {args.down_payment_max}).")
        exit()
    if not np.isclose(D_percentage_values, D_percentage_customer).any():
        D_percentage_values = np.sort(np.append(D_percentage_values, D_percentage_customer))
else:
    D_percentage_customer = None

if args.customer_term_length is not None:
    if args.customer_term_length < args.term_min or args.customer_term_length > args.term_max:
        print(f"Customer's term length {args.customer_term_length} is outside the term length range ({args.term_min} to {args.term_max}).")
        exit()
    if not np.isclose(T_values, args.customer_term_length).any():
        T_values = np.sort(np.append(T_values, args.customer_term_length))

if args.customer_interest_rate is not None:
    if args.customer_interest_rate < args.interest_min or args.customer_interest_rate > args.interest_max:
        print(f"Customer's interest rate {args.customer_interest_rate} is outside the interest rate range ({args.interest_min} to {args.interest_max}).")
        exit()
    if not np.isclose(r_c_values, args.customer_interest_rate).any():
        r_c_values = np.sort(np.append(r_c_values, args.customer_interest_rate))

# Ensure fixed value is included in parameter ranges
def ensure_value_in_array(values, fixed_value, name):
    if fixed_value < np.min(values) or fixed_value > np.max(values):
        print(f"Fixed value {fixed_value} is outside the {name} range ({np.min(values)} to {np.max(values)}).")
        exit()
    if not np.isclose(values, fixed_value).any():
        values = np.sort(np.append(values, fixed_value))
    return values

if args.fixed_dimension == 'purchase_amount':
    P_values = ensure_value_in_array(P_values, args.fixed_value, 'purchase amount')
elif args.fixed_dimension == 'down_payment':
    D_percentage_values = ensure_value_in_array(D_percentage_values, args.fixed_value / args.purchase_max, 'down payment percentage')
elif args.fixed_dimension == 'term_length':
    T_values = ensure_value_in_array(T_values, args.fixed_value, 'term length')
elif args.fixed_dimension == 'interest_rate':
    r_c_values = ensure_value_in_array(r_c_values, args.fixed_value, 'interest rate')

# Create a 4D meshgrid
P_grid, D_percentage_grid, T_grid, r_c_grid = np.meshgrid(
    P_values, D_percentage_values, T_values, r_c_values, indexing='ij'
)

# Compute Down Payment Grid
D_grid = P_grid * D_percentage_grid

# Compute Loan Amount Grid
L_grid = P_grid - D_grid

# Exclude combinations where loan amount is zero or negative
valid_L_mask = L_grid > 0

# 4. Compute Profit Margin, Profit Amount, and Risk

def default_probability(L, T, r_c):
    E_L = L * (1 + r_c * T / 12)
    exponent = alpha * (E_L - E_L_base) + beta * (T - T_base)
    P_d = P_base * np.exp(exponent)
    P_d = np.clip(P_d, 0, 1)  # Ensure P_d is between 0 and 1
    return P_d

def cost_of_funds(T):
    return r_f * (T / 12)

def profit_amount(P, L, T, r_c):
    Pd = default_probability(L, T, r_c)
    Cf = cost_of_funds(T)
    interest_income = L * r_c * (T / 12)
    expected_repayment = (L + interest_income) * (1 - Pd)
    # Processing revenue is a percentage of the purchase amount
    R_t = P * R_t_rate
    profit_amt = expected_repayment - L - (L * Cf) + R_t
    return profit_amt

def profit_margin(L, profit_amt):
    profit_margin_percentage = (profit_amt / L) * 100
    return profit_margin_percentage

def risk(L, T, r_c):
    Pd = default_probability(L, T, r_c)
    total_amount = L + L * r_c * (T / 12)
    expected_loss = total_amount * Pd
    return expected_loss

# Calculate grids
Pd_grid = default_probability(L_grid, T_grid, r_c_grid)
profit_amount_grid = profit_amount(P_grid, L_grid, T_grid, r_c_grid)
profit_margin_grid = profit_margin(L_grid, profit_amount_grid)
R_grid = risk(L_grid, T_grid, r_c_grid)

# Exclude invalid combinations where L_grid <= 0
valid_combination_mask = valid_L_mask

# 5. Apply Risk Appetite Constraints

profit_margin_mask = (profit_margin_grid >= P_margin_min)
risk_mask = R_grid <= R_max
acceptable_mask = profit_margin_mask & risk_mask & valid_combination_mask

# Flatten the grids
P_flat = P_grid.flatten()
D_flat = D_grid.flatten()
D_percentage_flat = D_percentage_grid.flatten()
L_flat = L_grid.flatten()
T_flat = T_grid.flatten()
r_c_flat = r_c_grid.flatten()
profit_margin_flat = profit_margin_grid.flatten()
profit_amount_flat = profit_amount_grid.flatten()
R_flat = R_grid.flatten()
acceptable_flat = acceptable_mask.flatten()

# Filter acceptable points
P_acc = P_flat[acceptable_flat]
D_acc = D_flat[acceptable_flat]
D_percentage_acc = D_percentage_flat[acceptable_flat]
L_acc = L_flat[acceptable_flat]
T_acc = T_flat[acceptable_flat]
r_c_acc = r_c_flat[acceptable_flat]
profit_margin_acc = profit_margin_flat[acceptable_flat]
profit_amount_acc = profit_amount_flat[acceptable_flat]
R_acc = R_flat[acceptable_flat]

# Convert interest rates to percentages for output and plotting
r_c_acc_percent = r_c_acc * 100

# 6. Filter based on the fixed dimension

fixed_dimension = args.fixed_dimension
fixed_value = args.fixed_value

if fixed_dimension == 'purchase_amount':
    fixed_mask = np.isclose(P_acc, fixed_value)
elif fixed_dimension == 'down_payment':
    fixed_mask = np.isclose(D_acc, fixed_value)
elif fixed_dimension == 'term_length':
    fixed_mask = np.isclose(T_acc, fixed_value)
elif fixed_dimension == 'interest_rate':
    fixed_mask = np.isclose(r_c_acc, fixed_value)
else:
    print("Invalid fixed dimension")
    exit()

# Apply the fixed mask
P_acc_plot = P_acc[fixed_mask]
D_acc_plot = D_acc[fixed_mask]
D_percentage_acc_plot = D_percentage_acc[fixed_mask]
L_acc_plot = L_acc[fixed_mask]
T_acc_plot = T_acc[fixed_mask]
r_c_acc_plot = r_c_acc[fixed_mask]
r_c_acc_percent_plot = r_c_acc_percent[fixed_mask]
profit_margin_acc_plot = profit_margin_acc[fixed_mask]
profit_amount_acc_plot = profit_amount_acc[fixed_mask]
R_acc_plot = R_acc[fixed_mask]

# Check if there are any acceptable points
if len(P_acc_plot) == 0:
    print(f"No acceptable investment opportunities found for fixed {fixed_dimension} = {args.fixed_value}.")
    exit()

# 7. Identify Top 5 Most Profitable Combinations

# Sort the acceptable combinations by profit amount in descending order
sorted_indices = np.argsort(-profit_amount_acc_plot)
top_n = min(5, len(sorted_indices))  # Adjust if less than 5 combinations are available
top_indices = sorted_indices[:top_n]

# Extract top combinations
top_P = P_acc_plot[top_indices]
top_D = D_acc_plot[top_indices]
top_L = L_acc_plot[top_indices]
top_T = T_acc_plot[top_indices]
top_r_c = r_c_acc_percent_plot[top_indices]
top_profit_margin = profit_margin_acc_plot[top_indices]
top_profit_amount = profit_amount_acc_plot[top_indices]
top_R = R_acc_plot[top_indices]

# Display the top combinations
print("Top Most Profitable Combinations:")
print(f"{'Index':<6} {'Purchase Amount ($)':<20} {'Down Payment ($)':<17} {'Loan Amount ($)':<15} {'Term (Months)':<15} {'Interest Rate (%)':<17} {'Profit Margin (%)':<18} {'Profit Amount ($)':<17} {'Risk ($)':<10}")
for i in range(len(top_indices)):
    print(f"{i+1:<6} {top_P[i]:<20.2f} {top_D[i]:<17.2f} {top_L[i]:<15.2f} {top_T[i]:<15.2f} {top_r_c[i]:<17.2f} {top_profit_margin[i]:<18.2f} {top_profit_amount[i]:<17.2f} {top_R[i]:<10.2f}")

# 8. Customer's Desired Point and Finding Closest Acceptable Point

plot_closest_point = False  # Flag to indicate whether to plot the closest acceptable point

if args.customer_purchase_amount is not None and args.customer_down_payment is not None and args.customer_term_length is not None and args.customer_interest_rate is not None:
    customer_P = args.customer_purchase_amount
    customer_D = args.customer_down_payment
    customer_D_percentage = customer_D / customer_P
    customer_L = customer_P - customer_D
    customer_T = args.customer_term_length
    customer_r_c = args.customer_interest_rate

    # Calculate profit amount, margin, and risk for customer's point
    profit_amt_customer = profit_amount(customer_P, customer_L, customer_T, customer_r_c)
    profit_margin_customer = profit_margin(customer_L, profit_amt_customer)
    risk_customer = risk(customer_L, customer_T, customer_r_c)

    # Check if customer's point is acceptable
    acceptable_customer = (profit_margin_customer >= P_margin_min) and (risk_customer <= R_max) and (customer_L > 0)

    if acceptable_customer:
        print("\nCustomer's desired combination is acceptable:")
        print(f"{'Purchase Amount ($)':<20} {'Down Payment ($)':<17} {'Loan Amount ($)':<15} "
              f"{'Term (Months)':<15} {'Interest Rate (%)':<17} {'Profit Margin (%)':<18} {'Profit Amount ($)':<17} {'Risk ($)':<10}")
        print(f"{customer_P:<20.2f} {customer_D:<17.2f} {customer_L:<15.2f} "
              f"{customer_T:<15.2f} {customer_r_c*100:<17.2f} {profit_margin_customer:<18.2f} {profit_amt_customer:<17.2f} {risk_customer:<10.2f}")
    elif args.find_closest:
        # Compute Euclidean distances to acceptable points
        distances = np.sqrt(
            ((P_acc - customer_P) / (args.purchase_max - args.purchase_min)) ** 2 +
            ((D_acc - customer_D) / (args.purchase_max - args.purchase_min)) ** 2 +
            ((T_acc - customer_T) / (args.term_max - args.term_min)) ** 2 +
            ((r_c_acc - customer_r_c) / (args.interest_max - args.interest_min)) ** 2
        )

        # Find the index of the closest acceptable point
        closest_idx = np.argmin(distances)

        # Extract the closest acceptable point
        closest_P = P_acc[closest_idx]
        closest_D = D_acc[closest_idx]
        closest_L = L_acc[closest_idx]
        closest_T = T_acc[closest_idx]
        closest_r_c = r_c_acc[closest_idx]
        closest_profit_margin = profit_margin_acc[closest_idx]
        closest_profit_amount = profit_amount_acc[closest_idx]
        closest_R = R_acc[closest_idx]

        print("\nCustomer's desired combination is NOT acceptable.")
        print("Closest acceptable combination:")
        print(f"{'Purchase Amount ($)':<20} {'Down Payment ($)':<17} {'Loan Amount ($)':<15} "
              f"{'Term (Months)':<15} {'Interest Rate (%)':<17} {'Profit Margin (%)':<18} {'Profit Amount ($)':<17} {'Risk ($)':<10}")
        print(f"{closest_P:<20.2f} {closest_D:<17.2f} {closest_L:<15.2f} "
              f"{closest_T:<15.2f} {closest_r_c*100:<17.2f} {closest_profit_margin:<18.2f} {closest_profit_amount:<17.2f} {closest_R:<10.2f}")

        plot_closest_point = True  # Set flag to plot the closest acceptable point
    else:
        print("\nCustomer's desired combination is NOT acceptable.")
        print("No acceptable alternatives found.")

# 9. Plot the Profitable Space (Profit Margin)

# For plotting, we use the three variable dimensions
if fixed_dimension == 'purchase_amount':
    X = D_acc_plot
    Y = T_acc_plot
    Z = r_c_acc_percent_plot
    title_fixed = f'Fixed Purchase Amount: ${fixed_value:.2f}'
    x_label = 'Down Payment ($)'
    y_label = 'Term Length (Months)'
    z_label = 'Interest Rate (%)'
elif fixed_dimension == 'down_payment':
    X = P_acc_plot
    Y = T_acc_plot
    Z = r_c_acc_percent_plot
    title_fixed = f'Fixed Down Payment: ${fixed_value:.2f}'
    x_label = 'Purchase Amount ($)'
    y_label = 'Term Length (Months)'
    z_label = 'Interest Rate (%)'
elif fixed_dimension == 'term_length':
    X = P_acc_plot
    Y = D_acc_plot
    Z = r_c_acc_percent_plot
    title_fixed = f'Fixed Term Length: {fixed_value:.2f} months'
    x_label = 'Purchase Amount ($)'
    y_label = 'Down Payment ($)'
    z_label = 'Interest Rate (%)'
elif fixed_dimension == 'interest_rate':
    X = P_acc_plot
    Y = T_acc_plot
    Z = D_acc_plot
    title_fixed = f'Fixed Interest Rate: {fixed_value*100:.2f}%'
    x_label = 'Purchase Amount ($)'
    y_label = 'Term Length (Months)'
    z_label = 'Down Payment ($)'
else:
    print("Invalid fixed dimension")
    exit()

# Exclude NaN or infinite profit margins from plotting
profit_margin_plot = np.array(profit_margin_acc_plot)
profit_margin_plot = np.where(np.isfinite(profit_margin_plot), profit_margin_plot, np.nan)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot of acceptable points
scatter = ax.scatter(X, Y, Z, c=profit_margin_plot, cmap='viridis', marker='o')

# Add colorbar for profit margin
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Profit Margin (%)', rotation=270, labelpad=15)

# Labels and title
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
ax.set_title(f'Acceptable Investment Opportunities (Profit Margin)\n{title_fixed}')

# Highlight the customer's point if applicable
if args.customer_purchase_amount is not None and args.customer_down_payment is not None and args.customer_term_length is not None and args.customer_interest_rate is not None:
    customer_P = args.customer_purchase_amount
    customer_D = args.customer_down_payment
    customer_T = args.customer_term_length
    customer_r_c = args.customer_interest_rate
    customer_r_c_percent = customer_r_c * 100

    if fixed_dimension == 'purchase_amount':
        if np.isclose(customer_P, fixed_value):
            X_cust = customer_D
            Y_cust = customer_T
            Z_cust = customer_r_c_percent
        else:
            X_cust, Y_cust, Z_cust = None, None, None
    elif fixed_dimension == 'down_payment':
        if np.isclose(customer_D, fixed_value):
            X_cust = customer_P
            Y_cust = customer_T
            Z_cust = customer_r_c_percent
        else:
            X_cust, Y_cust, Z_cust = None, None, None
    elif fixed_dimension == 'term_length':
        if np.isclose(customer_T, fixed_value):
            X_cust = customer_P
            Y_cust = customer_D
            Z_cust = customer_r_c_percent
        else:
            X_cust, Y_cust, Z_cust = None, None, None
    elif fixed_dimension == 'interest_rate':
        if np.isclose(customer_r_c, fixed_value):
            X_cust = customer_P
            Y_cust = customer_T
            Z_cust = customer_D
        else:
            X_cust, Y_cust, Z_cust = None, None, None
    else:
        X_cust, Y_cust, Z_cust = None, None, None

    if X_cust is not None:
        ax.scatter(X_cust, Y_cust, Z_cust, color='red', marker='*', s=200, label='Customer\'s Desired Point')

# Highlight the closest acceptable point if applicable
if plot_closest_point:
    if fixed_dimension == 'purchase_amount':
        if np.isclose(closest_P, fixed_value):
            X_close = closest_D
            Y_close = closest_T
            Z_close = closest_r_c * 100
        else:
            X_close, Y_close, Z_close = None, None, None
    elif fixed_dimension == 'down_payment':
        if np.isclose(closest_D, fixed_value):
            X_close = closest_P
            Y_close = closest_T
            Z_close = closest_r_c * 100
        else:
            X_close, Y_close, Z_close = None, None, None
    elif fixed_dimension == 'term_length':
        if np.isclose(closest_T, fixed_value):
            X_close = closest_P
            Y_close = closest_D
            Z_close = closest_r_c * 100
        else:
            X_close, Y_close, Z_close = None, None, None
    elif fixed_dimension == 'interest_rate':
        if np.isclose(closest_r_c, fixed_value):
            X_close = closest_P
            Y_close = closest_T
            Z_close = closest_D
        else:
            X_close, Y_close, Z_close = None, None, None
    else:
        X_close, Y_close, Z_close = None, None, None

    if X_close is not None:
        ax.scatter(X_close, Y_close, Z_close, color='green', marker='x', s=100, label='Closest Acceptable Point')

# Highlight the top combinations in the plot
ax.scatter(X[top_indices], Y[top_indices], Z[top_indices], color='red', marker='^', s=100, label='Top Combinations')

# Add legend
ax.legend()

plt.show()