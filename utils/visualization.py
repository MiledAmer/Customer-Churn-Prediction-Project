import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    """Plots a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_distribution(df, column):
    """Plots distribution of a single column."""
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_confusion_matrix(cm, classes):
    """Plots a confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_scatter(df, x_col, y_col, hue=None):
    """Plots a scatter plot between two variables."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.6)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.show()

def plot_histogram(df, col, bins=30):
    """Plots a histogram with bins."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=bins, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()