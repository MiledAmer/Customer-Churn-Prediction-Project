import matplotlib.pyplot as plt
import seaborn as sns

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