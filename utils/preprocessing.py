import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
import pickle

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

def select_features(df):
    """
    Removes features based on EDA findings to prepare for modeling.
    - Drops 'TotalCharges' due to high correlation (0.83) with 'tenure'.
    """
    df_clean = df.copy()
    if 'TotalCharges' in df_clean.columns:
        df_clean.drop('TotalCharges', axis=1, inplace=True)
        print("Feature Selection: Dropped 'TotalCharges' (Multicollinearity).")
    return df_clean

def split_stratified_data(df, target='Churn', test_size=0.2, random_state=42):
    """
    Separates X and y, and performs a Stratified Train-Test Split.
    Returns: X_train, X_test, y_train, y_test
    """
    # 1. Define Features (X) and Target (y)
    X = df.drop(target, axis=1)
    y = df[target]

    # 2. Perform Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data Split Successfully (Test Size: {test_size})")
    print(f" - Train Shape: {X_train.shape}")
    print(f" - Test Shape:  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def encode_binary_target(y, pos_label='Yes', neg_label='No'):
    """
    Encodes a binary target column into 1 and 0.
    
    Args:
        y (pd.Series): The target variable.
        pos_label (str): The label to map to 1 (Positive Class).
        neg_label (str): The label to map to 0 (Negative Class).
        
    Returns:
        pd.Series: The encoded target.
    """
    mapping = {pos_label: 1, neg_label: 0}
    print(f"Target Encoding: Mapping '{pos_label}' -> 1 and '{neg_label}' -> 0")
    
    return y.map(mapping)

def create_feature_transformer(X):
    """
    Builds a ColumnTransformer that scales numerical features 
    and one-hot encodes categorical features.
    """
    # 1. Detect Column Types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Transformer Configuration:")
    print(f" - Scaling {len(num_cols)} numerical cols: {num_cols}")
    print(f" - Encoding {len(cat_cols)} categorical cols: {cat_cols}")

    # 2. Build the Transformer
    transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), num_cols),
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    return transformer

def save_processed_data(X_train, X_test, y_train, y_test, transformer, save_dir='../data/processed'):
    """
    Saves the processed DataFrames and the Feature Transformer object.
    
    Args:
        X_train, X_test (pd.DataFrame): Processed feature sets.
        y_train, y_test (pd.Series): Encoded targets.
        transformer (ColumnTransformer): The fitted feature transformer.
        save_dir (str): The directory to save files to.
    """
    # 1. Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Save DataFrames to CSV
    # index=False prevents pandas from adding a distinct index column (0, 1, 2...) to the file
    X_train.to_csv(f'{save_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{save_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{save_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{save_dir}/y_test.csv', index=False)
    
    print(f"Data saved to {save_dir}/")
    
    # 3. Save the Transformer (Pickle)
    # We use 'wb' (write binary) because it's an object, not text
    pkl_path = f'{save_dir}/feature_transformer.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(transformer, f)
        
    print(f"Transformer saved to {pkl_path}")