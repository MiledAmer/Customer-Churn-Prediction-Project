import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Loads data from CSV."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Basic cleaning: Drops duplicates and fills missing numeric values with mean."""
    df = df.drop_duplicates()
    
    # Assuming 'df' is your dataframe name
    # 1. Convert TotalCharges to numeric. 'coerce' turns invalid parsing into NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 2. Check how many missing values were created
    missing_values = df['TotalCharges'].isnull().sum()
    print(f"Missing values in TotalCharges: {missing_values}")

    # 3. Handling the missing values
    # Since TotalCharges is likely 0 for new customers (tenure=0), let's fill them with 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0)


    # 4. Drop customerID (High cardinality, no predictive power)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print("customerID dropped.")
    
    return df




















# def split_data(X, y, test_size=0.2, random_state=42):
#     """Splits data into train and test sets."""
#     return train_test_split(X, y, test_size=test_size, random_state=random_state)

# def scale_features(X_train, X_test):
#     """Scales features using StandardScaler."""
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     return X_train_scaled, X_test_scaled, scaler