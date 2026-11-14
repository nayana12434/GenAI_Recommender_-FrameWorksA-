
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned datasets
products = pd.read_csv(DATA_DIR / "products_clean.csv")
users = pd.read_csv(DATA_DIR / "user_behavior_clean.csv")

print("PRODUCT DATA:", products.shape)
print("USER DATA:", users.shape)

# --------------------------------
# 1. PRODUCT FEATURE ENGINEERING
# --------------------------------

numeric_features = ["price", "discount", "rating"]

categorical_features = [
    "product_name",
    "category",
    "material",
    "colour",
    "style",
    "room_type"
]

# Handle missing values
for col in numeric_features:
    products[col] = products[col].fillna(products[col].median())

for col in categorical_features:
    products[col] = products[col].fillna(-1).astype(str)

# Scale numerical features
scaler = StandardScaler()

scaled_values = scaler.fit_transform(products[numeric_features])

scaled_df = pd.DataFrame(
    scaled_values,
    columns=numeric_features
)

# Convert categorical features into one-hot encoded columns
encoded_df = pd.get_dummies(
    products[categorical_features],
    columns=categorical_features,
    dtype=int
)

# Combine all features
product_features = pd.concat(
    [
        products[["product_id"]].reset_index(drop=True),
        scaled_df,
        encoded_df.reset_index(drop=True)
    ],
    axis=1
)

product_features.to_csv(
    OUTPUT_DIR / "product_features.csv",
    index=False
)

print("\nPRODUCT FEATURES CREATED")
print(product_features.head())
print("Feature shape:", product_features.shape)

# --------------------------------
# 2. USER BEHAVIOUR ENGINEERING
# --------------------------------

users["event_timestamp"] = pd.to_datetime(
    users["event_timestamp"],
    errors="coerce"
)

users["quantity"] = users["quantity"].fillna(0)

# Aggregate user-product interactions
user_interactions = users.groupby(
    ["user_id", "product_id"],
    as_index=False
).agg(
    event_count=("event_type", "count"),
    quantity_sum=("quantity", "sum"),
    last_interaction=("event_timestamp", "max")
)

user_interactions.to_csv(
    OUTPUT_DIR / "user_interactions.csv",
    index=False
)

print("\nUSER INTERACTIONS CREATED")
print(user_interactions.head())

# --------------------------------
# 3. SAVE RESULTS
# --------------------------------

print("\nFEATURE ENGINEERING COMPLETED!")
print("Saved files:")
print("outputs/product_features.csv")
print("outputs/user_interactions.csv")

