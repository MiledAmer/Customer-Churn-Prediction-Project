import numpy as np

def calculate_profit_curve(y_probs, y_test, clv, cost, success_rate):
    """
    Calculates the estimated profit for 100 different probability thresholds.
    """
    thresholds = np.linspace(0, 1, 101)
    profits = []

    for t in thresholds:
        # Who do we target? (Risk Score > Threshold)
        targeted_customers = (y_probs >= t)

        # True Positives: Churners we correctly targeted
        tp = np.sum((targeted_customers == 1) & (y_test == 1))

        # False Positives: Loyal people we annoyed (Wasted money)
        fp = np.sum((targeted_customers == 1) & (y_test == 0))

        # Money Logic
        gained = tp * success_rate * clv
        spent  = (tp + fp) * cost
        
        profit = gained - spent
        profits.append(profit)
        
    return thresholds, profits
