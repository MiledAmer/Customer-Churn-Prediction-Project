import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from utils.evaluation import calculate_profit_curve
# Set style globally
sns.set_style("whitegrid")

def plot_target_distribution(df, target_col='Churn'):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Target Distribution: {target_col} vs Non-{target_col}')
    plt.xlabel(f'{target_col} Status')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_correlation_heatmap(df_numeric):
    """
    Plots the correlation heatmap.
    Expects a dataframe that is ALREADY encoded/numeric.
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation
    corr_matrix = df_numeric.corr()
    
    # Plot
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix: Features vs Churn')
    plt.show()

def plot_categorical_distributions(df, cols_to_plot, target_col='Churn'):
    for col in cols_to_plot:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, hue=target_col, palette='viridis')
        
        plt.title(f'Churn Distribution by {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Number of Customers', fontsize=12)
        
        if df[col].nunique() > 3 or df[col].str.len().max() > 10:
            plt.xticks(rotation=45)
            
        plt.legend(title='Churn Status', loc='upper right')
        plt.tight_layout()
        plt.show()

def plot_numeric_distribution(df, col, target_col='Churn'):
    """
    Plots a Kernel Density Estimate (KDE) to compare the distribution 
    of a numeric feature for Churn vs Non-Churn customers.
    """
    plt.figure(figsize=(10, 6))
    
    # KDE plot shows the shape of the distribution (smooth histogram)
    sns.kdeplot(data=df, x=col, hue=target_col, fill=True, palette='viridis')
    
    plt.title(f'Distribution of {col} by {target_col} Status')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()

def plot_boxplot(df, x_col, y_col='Churn'):
    """
    Plots a boxplot to visualize statistical outliers.
    - x_col: The categorical column (usually the target 'Churn')
    - y_col: The numerical column to check for outliers (e.g., 'TotalCharges')
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x_col, y=y_col, palette='viridis')
    plt.title(f'Boxplot of {y_col} by {x_col}')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """
    Plots a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Loyal', 'Predicted Churn'],
                yticklabels=['Actual Loyal', 'Actual Churn'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_sensitivity_analysis(scenarios, y_probs, y_test):
    """
    Plots multiple profit curves on the same graph to compare different 
    business assumptions (Sensitivity Analysis).
    """
    plt.figure(figsize=(12, 7))

    # Loop through each scenario dictionary
    for sc in scenarios:
        # Calculate curve using the helper function we defined earlier
        thresholds, profits = calculate_profit_curve(y_probs, y_test, sc['clv'], sc['cost'], sc['rate'])
        
        # Find Max for this specific scenario
        max_profit = max(profits)
        best_thresh = thresholds[np.argmax(profits)]
        
        # Plot the Line
        plt.plot(thresholds, profits, label=f"{sc['name']} (Max: ${max_profit:,.0f})", color=sc['color'], linewidth=2.5)
        
        # Mark the Peak
        plt.scatter(best_thresh, max_profit, color=sc['color'], s=50, zorder=5)

    # Final Formatting
    plt.title("Sensitivity Analysis: Profit at Different Scenarios", fontsize=16)
    plt.xlabel("Decision Threshold (Probability of Churn)", fontsize=12)
    plt.ylabel("Estimated Profit ($)", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--') # Add a zero line to show losses
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()