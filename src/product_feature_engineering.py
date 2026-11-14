"""
Person 5 - Product Feature Engineering
Run from the project root:
    python person5_product_feature_engineering.py

Input priority:
1. data/raw/products.csv (keeps original text categories)
2. data/processed/products_clean.csv (fallback; may contain LabelEncoder IDs)
Outputs:
    outputs/product_features.csv
    outputs/product_features.parquet
"""

from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]  # project root (script is inside src/)
# If this script is copied into the original project root, these paths work.
RAW_FILE = BASE_DIR / "data" / "raw" / "products.csv"
CLEAN_FILE = BASE_DIR / "data" / "processed" / "products_clean.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_ALIASES = {
    "product_id": ["product_id", "productid", "id"],
    "product_name": ["product_name", "name", "product"],
    "category": ["category", "product_category"],
    "material": ["material"],
    "colour": ["colour", "color"],
    "style": ["style"],
    "price": ["price", "selling_price"],
    "discount": ["discount", "discount_percentage"],
    "rating": ["rating", "average_rating"],
    "room_type": ["room_type", "room", "recommended_room"],
    "description": ["description", "product_description"],
    "length_cm": ["length_cm", "length"],
    "width_cm": ["width_cm", "width"],
    "height_cm": ["height_cm", "height"],
}

CATEGORY_HIERARCHY = {
    "sofa": ("Living Room Furniture", "Seating"),
    "chair": ("Seating Furniture", "Chair"),
    "bed": ("Bedroom Furniture", "Sleeping"),
    "wardrobe": ("Bedroom Furniture", "Storage"),
    "table": ("Table Furniture", "Surface"),
    "desk": ("Office Furniture", "Workstation"),
    "bookshelf": ("Storage Furniture", "Shelving"),
    "cabinet": ("Storage Furniture", "Cabinet"),
    "dining table": ("Dining Room Furniture", "Dining"),
    "dining furniture": ("Dining Room Furniture", "Dining"),
    "other furniture": ("Decor & Utility Furniture", "Utility"),
}

def normalise_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in df.columns
    ]
    # Resolve common spelling/column variants.
    rename = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
            if alias in df.columns:
                rename[alias] = standard
                break
    return df.rename(columns=rename)

def load_products():
    source = RAW_FILE if RAW_FILE.exists() else CLEAN_FILE
    if not source.exists():
        raise FileNotFoundError(
            "Could not find data/raw/products.csv or "
            "data/processed/products_clean.csv. Run the data generation "
            "and cleaning steps first."
        )
    print(f"Reading: {source}")
    return normalise_columns(pd.read_csv(source))

def make_description(row):
    existing = row.get("description")
    if pd.notna(existing) and str(existing).strip():
        return str(existing).strip()

    name = row.get("product_name", "Furniture item")
    category = row.get("category", "Furniture")
    material = row.get("material", "quality material")
    colour = row.get("colour", "neutral")
    style = row.get("style", "versatile")
    room = row.get("room_type", "multiple rooms")
    dims = []
    for col, label in [
        ("length_cm", "L"), ("width_cm", "W"), ("height_cm", "H")
    ]:
        value = row.get(col)
        if pd.notna(value):
            dims.append(f"{label} {value} cm")
    dimension_text = (
        " Dimensions: " + " × ".join(dims) + "."
        if dims else ""
    )
    return (
        f"{name} is a {style} {category.lower()} designed for the "
        f"{str(room).lower()}. Made from {str(material).lower()} with "
        f"a {str(colour).lower()} finish, it combines practical use "
        f"with an attractive appearance.{dimension_text}"
    )

def engineer_product_features(products):
    df = normalise_columns(products)

    required = ["product_id", "product_name", "category", "material",
                "colour", "style", "price", "rating", "room_type"]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    # Convert numerical fields safely.
    for col in ["price", "discount", "rating",
                "length_cm", "width_cm", "height_cm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean text fields without converting missing values into the word 'nan'.
    text_cols = ["product_name", "category", "material", "colour",
                 "style", "room_type"]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Fill only missing optional dimensions with NA; do not invent measurements.
    for col in ["length_cm", "width_cm", "height_cm"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Derived total dimension string, useful for display and filtering.
    def dimension_label(row):
        vals = [row[c] for c in ["length_cm", "width_cm", "height_cm"]]
        if all(pd.notna(v) for v in vals):
            return f"{vals[0]} x {vals[1]} x {vals[2]} cm"
        return "Not available"
    df["dimensions_cm"] = df.apply(dimension_label, axis=1)

    # Price bands: explicit INR thresholds, adjustable to project requirements.
    def price_band(price):
        if pd.isna(price):
            return "Unknown"
        if price < 10000:
            return "Budget"
        if price <= 40000:
            return "Mid-Range"
        return "Premium"
    df["price_category"] = df["price"].apply(price_band)

    # Category hierarchy.
    def hierarchy(category):
        key = str(category).strip().lower() if pd.notna(category) else ""
        if key in CATEGORY_HIERARCHY:
            return CATEGORY_HIERARCHY[key]
        return ("Other Furniture", "Other")
    hierarchy_values = df["category"].apply(hierarchy)
    df["category_group"] = hierarchy_values.apply(lambda x: x[0])
    df["category_subgroup"] = hierarchy_values.apply(lambda x: x[1])

    # Room classification: retain source room, add a broad room group.
    def room_group(room):
        value = str(room).lower() if pd.notna(room) else ""
        if "living" in value:
            return "Living Area"
        if "bedroom" in value or "bed room" in value:
            return "Bedroom"
        if "dining" in value or "kitchen" in value:
            return "Dining Area"
        if "office" in value or "study" in value:
            return "Work Area"
        if "entry" in value:
            return "Entryway"
        return "Multi-purpose / Unknown"
    df["room_group"] = df["room_type"].apply(room_group)

    # A readable text field to use later for text embeddings / semantic search.
    df["product_description"] = df.apply(make_description, axis=1)
    df["embedding_text"] = df.apply(
        lambda r: (
            f"Product: {r['product_name']}. Category: {r['category']}. "
            f"Category group: {r['category_group']}. "
            f"Material: {r['material']}. Colour: {r['colour']}. "
            f"Style: {r['style']}. Room: {r['room_type']}. "
            f"Price category: {r['price_category']}. "
            f"Description: {r['product_description']}"
        ), axis=1
    )

    # Keep useful original columns and engineered columns.
    output_cols = [
        "product_id", "product_name", "category", "category_group",
        "category_subgroup", "material", "colour", "style", "room_type",
        "room_group", "price", "price_category", "discount", "rating",
        "length_cm", "width_cm", "height_cm", "dimensions_cm",
        "product_description", "embedding_text"
    ]
    for col in output_cols:
        if col not in df.columns:
            df[col] = pd.NA
    result = df[output_cols].copy()

    # Remove duplicate IDs only when IDs are present; keep first catalogue row.
    if result["product_id"].notna().any():
        result = result.drop_duplicates(subset=["product_id"], keep="first")
    return result

def main():
    source_df = load_products()
    features = engineer_product_features(source_df)

    csv_path = OUTPUT_DIR / "product_features.csv"
    parquet_path = OUTPUT_DIR / "product_features.parquet"
    features.to_csv(csv_path, index=False)

    try:
        features.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet: {parquet_path}")
    except ImportError:
        print("Parquet engine missing. Install one with: pip install pyarrow")
        print("CSV output was saved successfully.")

    print("\nPRODUCT FEATURE ENGINEERING COMPLETE")
    print("Rows:", len(features))
    print("Columns:", len(features.columns))
    print(features.head().to_string(index=False))

if __name__ == "__main__":
    main()
