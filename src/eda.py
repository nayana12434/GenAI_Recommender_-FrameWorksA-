
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load cleaned datasets
products = pd.read_csv(PROCESSED_DIR / "products_clean.csv")
users = pd.read_csv(PROCESSED_DIR / "user_behavior_clean.csv")

# Display basic information
print("\nPRODUCT DATASET")
print(products.head())

print("\nPRODUCT INFORMATION")
print(products.info())

print("\nPRODUCT STATISTICS")
print(products.describe(include="all"))

print("\nUSER BEHAVIOUR DATASET")
print(users.head())

print("\nUSER BEHAVIOUR INFORMATION")
print(users.info())

# Product category analysis
print("\nPRODUCTS BY CATEGORY")
print(products["category"].value_counts())

print("\nAVERAGE PRICE BY CATEGORY")
print(products.groupby("category")["price"].mean())

# Product category chart
plt.figure(figsize=(9, 5))
sns.countplot(data=products, x="category")
plt.title("Furniture Products by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price distribution
plt.figure(figsize=(9, 5))
sns.histplot(products["price"], bins=10, kde=True)
plt.title("Furniture Price Distribution")
plt.xlabel("Price")
plt.tight_layout()
plt.show()

# Rating distribution
plt.figure(figsize=(9, 5))
sns.histplot(products["rating"], bins=10, kde=True)
plt.title("Furniture Rating Distribution")
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")



# Average rating by category
print("\nAVERAGE RATING BY CATEGORY")
print(products.groupby("category")["rating"].mean())

# Discount analysis
print("\nAVERAGE DISCOUNT")
print(products["discount"].mean())

# User behaviour analysis
print("\nUSER EVENT DISTRIBUTION")
print(users["event_type"].value_counts())

# Products with highest ratings
print("\nTOP RATED PRODUCTS")
print(products.nlargest(5, "rating")[
    ["product_id", "product_name", "price", "rating"]
])

# Price by category chart
plt.figure(figsize=(10, 5))
sns.barplot(data=products, x="category", y="price")
plt.title("Average Furniture Price by Category")
plt.xlabel("Category")
plt.ylabel("Average Price")
plt.tight_layout()
plt.show()

# User behaviour chart
plt.figure(figsize=(8, 5))
sns.countplot(data=users, x="event_type")
plt.title("User Behaviour Event Distribution")
plt.xlabel("Event Type")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.show()

print("\nCOMPLETE EDA FINISHED!")

