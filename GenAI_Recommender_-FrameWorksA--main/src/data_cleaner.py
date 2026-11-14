
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Project folder paths
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Function to clean product data
def clean_products():

    file_path = RAW_DIR / "products.csv"

    df = pd.read_csv(file_path)

    print("Original Product Dataset:")
    print(df.head())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Remove duplicate records
    df = df.drop_duplicates()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    # Standardise text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Identify price and discount columns
    price_cols = [c for c in df.columns if "price" in c]
    discount_cols = [c for c in df.columns if "discount" in c]

    # Handle missing prices
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Handle missing discounts
    for col in discount_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    # Standardise product names and categories
    for col in df.columns:
        if "name" in col or "category" in col:
            df[col] = df[col].astype("string").str.strip().str.title()

    # Encode categorical variables
    for col in df.select_dtypes(include=["object", "string"]).columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].fillna("Unknown").astype(str))

    # Save clean dataset
    df.to_csv(
        PROCESSED_DIR / "products_clean.csv",
        index=False
    )

    print("\nClean product dataset saved successfully!")
    return df


# Function to clean user behaviour data
def clean_user_behavior():

    file_path = RAW_DIR / "user_behavior.csv"

    df = pd.read_csv(file_path)

    print("\nOriginal User Behaviour Dataset:")
    print(df.head())

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower()

    # Standardise text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Convert timestamp
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(
            df["event_timestamp"], errors="coerce"
        )

    # Handle missing categorical values
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Handle missing numerical values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    # Encode categorical variables
    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    # Save cleaned user behaviour data
    df.to_csv(
        PROCESSED_DIR / "user_behavior_clean.csv",
        index=False
    )

    print("\nClean user behaviour dataset saved successfully!")
    return df


# Main execution
if __name__ == "__main__":

    products = clean_products()
    behaviour = clean_user_behavior()

    print("\nDATA CLEANING COMPLETED!")
    print("Products:", products.shape)
    print("User Behaviour:", behaviour.shape)

