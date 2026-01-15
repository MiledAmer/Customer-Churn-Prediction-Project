import pandas as pd

def load_data(filepath):
    """Loads data from CSV."""
    return pd.read_csv(filepath)

def clean_data(df):

    # 1. Convert TotalCharges to numeric. 'coerce' turns invalid parsing into NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"Duplicate rows found: {df.duplicated().sum()}")

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

def encode_for_eda(df):
    """
    Prepares a copy of the dataframe specifically for Correlation Analysis.
    - Maps Binary columns (Yes/No) to 1/0.
    - Label Encodes nominal columns (Text -> Numbers) so they appear in heatmaps.
    
    Note: This creates a 'Label Encoded' version, not One-Hot. 
    Use this for EDA, not for training linear models.
    """
    df_encoded = df.copy()
    
    # 1. Map typical Binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
            
    if 'gender' in df_encoded.columns:
        df_encoded['gender'] = df_encoded['gender'].map({'Female': 1, 'Male': 0})
        
    # 2. Factorize (Label Encode) remaining categorical columns
    # We select all columns that are still 'object' type
    object_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in object_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
    return df_encoded