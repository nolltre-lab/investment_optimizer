# Investment Opportunities Optimizer

## Introduction

The Investment Opportunities Optimizer is a Python script designed to help financial institutions evaluate potential loan offerings based on profitability and risk constraints. It computes acceptable investment opportunities by analyzing various combinations of loan parameters and identifies options that meet predefined risk appetite and profit margin requirements.

## Key Concepts

### Loan Parameters

The script considers the following key loan parameters:

* Purchase Amount (P): The total cost of the item or service being financed.
* Down Payment (D): The initial payment made by the customer, reducing the loan amount.
* Loan Amount (L): The amount financed by the lender, calculated as L = P - D.
* Term Length (T): The duration of the loan in months.
* Customer Interest Rate (r_c): The annual interest rate charged to the customer (as a decimal, e.g., 0.05 for 5%).

### Profitability and Risk Assessment

The script evaluates each potential loan offering based on:

* Profit Amount: The expected monetary gain from the loan after accounting for costs and risks.
* Profit Margin: The profit amount expressed as a percentage of the loan amount.
* Risk (R): The expected loss due to the probability of default.

### Acceptable Investment Opportunities

An investment opportunity is considered acceptable if:

1. Profit Margin meets or exceeds the minimum acceptable profit margin (P_margin_min).
2. Risk (R) is less than or equal to the maximum acceptable risk (R_max).
3. Loan Amount (L) is positive.

### Assumptions and Parameters

The script is built upon several financial assumptions and parameters:

1. Base Probability of Default (P_base): Set at 0.01 (1%), representing the default probability under base conditions. *Typically output from a customer scoring model*
2. Sensitivity Coefficients:
 * Alpha (alpha): Sensitivity to the effective loan amount, set at 0.0001. *Typically another output of the customer scoring model. Used to indicate that the higher the debt a customer is having, the higher the likelihood that customer will default*
 * Beta (beta): Sensitivity to the term length uncertainty, set at 0.02. *Typically another output of the customer scoring model. Used to indicate that certain customers are more sensitity to for example changing macroeconomic factors*
4. Uncertainty Factor Coefficient:
 * Gamma (gamma): Rate at which uncertainty increases with term length, set at 0.05. *Either defined as an uncertainty by analyst or as an output from scoring model*
3. Base Loan Conditions:
 * Base Loan Amount (L_base): $5,000. *This could be used to define current open debt*
 * Base Term Length (T_base): 12 months. *This is used to base the term sensitivity measures*
 * Base Customer Interest Rate (r_c_base): 0.10 (10%).
4. Cost of Funds Rate (r_f): The lender’s annual cost of capital, set at 0.05 (5%). *Should use standard lending company values*
5. Processing Revenue Rate (R_t_rate): Additional revenue from processing fees, set at 0.02 (2%) of the purchase amount. *Likely different for different purchase partners*
6. Risk Appetite Constraints:
 * Maximum Acceptable Risk (R_max): $300. *Key input from risk analysts to control the amount of risk acceptable for each purchase*
 * Minimum Acceptable Profit Margin (P_margin_min): 0%. *Key input from risk analyst to control the amount of risk to accept. High required profit margin would indicate low risk appetite for scenario*

### Formulas Used

#### 1. Default Probability (P_d)

The probability that a borrower will default on the loan is calculated using an exponential function that accounts for the loan amount and incorporates increasing uncertainty over the term length:

* Effective Loan Amount (E_L):

E_L = L * (1 + (r_c * T) / 12)

* Exponent:

exponent = alpha * (E_L - E_L_base) + beta * sqrt(max(T - T_base, 0))

 * Note: The sqrt(max(T - T_base, 0)) term models the increasing uncertainty over time, increasing at a decreasing rate as the term lengthens.

* Probability of Default (P_d):

P_d = P_base * exp(exponent)

* Ensure that P_d is between 0 and 1:

P_d = min(max(P_d, 0), 1)

#### 2. Cost of Funds (C_f)

The lender’s cost of borrowing capital for the loan term:

C_f = r_f * (T / 12)

#### 3. Profit Amount

The expected profit from the loan, considering interest income, expected repayment, processing revenue, and costs:

* Interest Income:

interest_income = L * r_c * (T / 12)

* Expected Repayment:

expected_repayment = (L + interest_income) * (1 - P_d)

* Processing Revenue (R_t):

R_t = P * R_t_rate

* Profit Amount:

profit_amount = expected_repayment - L - (L * C_f) + R_t

#### 4. Profit Margin

The profit amount expressed as a percentage of the loan amount:

profit_margin = (profit_amount / L) * 100

#### 5. Risk (R)

The expected loss due to default, adjusted for uncertainty over the term length:

* Total Amount:

total_amount = L + L * r_c * (T / 12)

*	Uncertainty Factor:

uncertainty_factor = 1 + gamma * sqrt(max(T - T_base, 0))

 * Note: This factor increases the expected loss to account for the growing uncertainty in default probability assessments over longer terms.


* Risk:

R = total_amount * P_d * uncertainty_factor

## Examples of Script Usage

### Example 1: Finding Profitable Options with No Down Payment

To find acceptable investment opportunities for a fixed purchase amount of $5,000 with no down payment, use the following command:

```
python3 investment_optimizer.py --fixed_dimension purchase_amount --fixed_value 5000 --down_payment_max 0
```

Explanation:

```
--fixed_dimension purchase_amount: Fixes the purchase amount parameter.
--fixed_value 5000: Sets the fixed purchase amount to $5,000.
--down_payment_max 0: Sets the maximum down payment to 0% (no down payment).
```

Output:

```
Top Most Profitable Combinations:
Index  Purchase Amount ($)  Down Payment ($)  Loan Amount ($)  Term (Months)  Interest Rate (%)  Profit Margin (%)  Profit Amount ($)  Risk ($)
1      5000.00              0.00              5000.00          36.00          20.00              44.39              2219.41            130.59
2      5000.00              0.00              5000.00          36.00          19.00              42.07              2099.33            108.69
...
```

### Example 2: Evaluating a Customer’s Desired Loan

If a customer requests specific loan terms, you can input their desired combination and find the closest acceptable alternative if it’s not acceptable.

Command:

```
python3 investment_optimizer.py \
  --fixed_dimension purchase_amount \
  --fixed_value 10000 \
  --customer_purchase_amount 10000 \
  --customer_down_payment 1000 \
  --customer_term_length 12 \
  --customer_interest_rate 0.03 \
  --find_closest
```

Explanation:
```
--customer_purchase_amount 10000: Customer’s desired purchase amount of $10,000.
--customer_down_payment 1000: Customer’s down payment of $1,000.
--customer_term_length 12: Customer’s desired term length of 12 months.
--customer_interest_rate 0.03: Customer’s desired interest rate of 3% (entered as 0.03).
--find_closest: Finds the closest acceptable combination if the desired one isn’t acceptable.
```

Possible Output:
```
Customer's desired combination is NOT acceptable.
Closest acceptable combination:
Purchase Amount ($)  Down Payment ($)  Loan Amount ($)  Term (Months)  Interest Rate (%)  Profit Margin (%)  Profit Amount ($)  Risk ($)
10000.00             1000.00           9000.00          12.00          5.00               5.50               495.00             200.00
```

How the Script Can Be Used

* Financial Analysts: To assess the profitability and risk of various loan offerings and identify optimal loan parameters.
* Customer Service Representatives: To evaluate customer requests and offer acceptable loan alternatives when the desired terms are not viable.
* Decision Makers: To understand the impact of different loan parameters on profitability and risk, aiding in policy formulation.

Important Notes

* Interest Rates as Decimals: When inputting interest rates, use decimal form (e.g., 0.05 for 5%).
* Parameter Ranges: Ensure that the customer’s desired values fall within the specified parameter ranges.
* Simplified Model: The script uses a simplified risk model and assumes fixed sensitivity coefficients (alpha and beta). Adjust these values as needed to fit real-world data.
* Assumptions Validity: Validate the assumptions against your organization’s data and adjust parameters accordingly.

## Conclusion

The Investment Opportunities Optimizer is a powerful tool that helps in:

* Identifying acceptable investment opportunities based on profitability and risk criteria.
* Offering customers viable loan alternatives when their desired terms are not acceptable.
* Making informed decisions by analyzing the interplay between different loan parameters.

By customizing the script’s parameters and ranges, you can tailor the analysis to your organization’s specific needs and risk appetite.

## Disclaimer

Important: This script is provided for educational and informational purposes only. It is a simplified model intended to demonstrate concepts related to loan risk assessment and profitability analysis.

* Not for Production Use: The script is not intended for production environments and should not be used as-is to make real-world financial decisions or to offer actual loan products.
* No Warranty: The author(s) make no warranties or representations regarding the accuracy, completeness, or suitability of the script for any particular purpose.
* Use at Your Own Risk: Any use of this script or its components is at your own risk. The author(s) are not liable for any losses, damages, or adverse outcomes resulting from the use of this script.
* Professional Consultation Recommended: Before implementing any financial models or tools in a production setting, it is essential to consult with qualified financial professionals and risk management experts.
* Subject to Change: Financial models and regulatory requirements can vary widely and are subject to change. This script may not reflect the most current industry practices or legal requirements.

## Appendix: Script Parameters Overview

Command-Line Arguments

```
--fixed_dimension: Dimension to fix for plotting (purchase_amount, down_payment, term_length, or interest_rate).
--fixed_value: Value at which to fix the selected dimension.
```

Optional Parameters for Customer’s Desired Point

```
--customer_purchase_amount: Customer’s desired purchase amount.
--customer_down_payment: Customer’s desired down payment amount.
--customer_term_length: Customer’s desired term length in months.
--customer_interest_rate: Customer’s desired interest rate (as a decimal).
```

Flags

```
--find_closest: Find the closest acceptable point to the customer’s desired combination if it is not acceptable.
```

Remember to run the script with the -h or --help flag to see all available options and their descriptions:

```
python3 investment_optimizer.py --help
```

Note: This document is intended to provide an overview of the Investment Opportunities Optimizer script, its functionality, and how to use it effectively. Adjust the script parameters and formulas as needed to align with your specific business requirements and regulatory environment.
