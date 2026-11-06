
import pandas as pd
import random
from pathlib import Path

# ==========================================
# URBANNEST FURNITURE PRODUCT GENERATOR
# ==========================================

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

SOURCE_FILE = RAW_DIR / "Furniture.csv"
OUTPUT_FILE = RAW_DIR / "products.csv"

NUM_PRODUCTS = 600

# ==========================================
# 1. FURNITURE CATALOGUE CONFIGURATION
# ==========================================

FURNITURE = {
    "Sofa": {
        "types": ["2-Seater Sofa", "3-Seater Sofa", "L-Shaped Sofa",
                  "Sectional Sofa", "Loveseat", "Recliner Sofa"],
        "materials": ["Fabric", "Linen", "Velvet", "Leather", "Wood"],
        "colours": ["Beige", "Grey", "Brown", "Navy Blue", "White", "Green"],
        "styles": ["Modern", "Minimalist", "Contemporary", "Scandinavian"],
        "rooms": ["Living Room"],
        "price": (18000, 150000),
        "dimensions": ((140, 300), (75, 120), (70, 110))
    },

    "Bed": {
        "types": ["Single Bed", "Queen Bed", "King Bed",
                  "Storage Bed", "Platform Bed", "Bunk Bed"],
        "materials": ["Solid Wood", "Engineered Wood", "Metal", "Upholstered Fabric"],
        "colours": ["Brown", "White", "Black", "Walnut", "Grey"],
        "styles": ["Modern", "Traditional", "Minimalist", "Contemporary"],
        "rooms": ["Bedroom"],
        "price": (12000, 120000),
        "dimensions": ((90, 220), (90, 210), (60, 120))
    },

    "Chair": {
        "types": ["Dining Chair", "Accent Chair", "Office Chair",
                  "Lounge Chair", "Rocking Chair", "Armchair"],
        "materials": ["Wood", "Metal", "Plastic", "Fabric", "Leather"],
        "colours": ["Black", "White", "Brown", "Grey", "Green", "Blue"],
        "styles": ["Modern", "Industrial", "Minimalist", "Traditional"],
        "rooms": ["Living Room", "Dining Room", "Home Office"],
        "price": (1500, 35000),
        "dimensions": ((40, 90), (40, 90), (70, 120))
    },

    "Table": {
        "types": ["Coffee Table", "Side Table", "Console Table",
                  "Nesting Table", "Center Table", "End Table"],
        "materials": ["Solid Wood", "Glass", "Metal", "Engineered Wood"],
        "colours": ["Brown", "Black", "White", "Oak", "Walnut"],
        "styles": ["Modern", "Minimalist", "Rustic", "Industrial"],
        "rooms": ["Living Room", "Bedroom", "Entryway"],
        "price": (2500, 50000),
        "dimensions": ((40, 180), (35, 100), (30, 90))
    },

    "Desk": {
        "types": ["Study Desk", "Computer Desk", "Executive Desk",
                  "Writing Desk", "Standing Desk", "Gaming Desk"],
        "materials": ["Solid Wood", "Engineered Wood", "Metal"],
        "colours": ["White", "Black", "Brown", "Oak", "Grey"],
        "styles": ["Modern", "Minimalist", "Industrial", "Scandinavian"],
        "rooms": ["Home Office", "Bedroom"],
        "price": (3500, 60000),
        "dimensions": ((80, 180), (45, 90), (70, 120))
    },

    "Wardrobe": {
        "types": ["Two Door Wardrobe", "Three Door Wardrobe",
                  "Sliding Door Wardrobe", "Mirror Wardrobe"],
        "materials": ["Engineered Wood", "Solid Wood", "Plywood"],
        "colours": ["White", "Brown", "Grey", "Oak", "Black"],
        "styles": ["Modern", "Minimalist", "Contemporary"],
        "rooms": ["Bedroom"],
        "price": (10000, 100000),
        "dimensions": ((80, 220), (45, 75), (160, 240))
    },

    "Cabinet": {
        "types": ["Storage Cabinet", "Display Cabinet", "TV Unit",
                  "Shoe Cabinet", "Sideboard"],
        "materials": ["Solid Wood", "Engineered Wood", "Metal"],
        "colours": ["Brown", "White", "Black", "Grey"],
        "styles": ["Modern", "Rustic", "Industrial", "Minimalist"],
        "rooms": ["Living Room", "Bedroom", "Entryway"],
        "price": (4000, 65000),
        "dimensions": ((50, 200), (30, 70), (50, 180))
    },

    "Bookshelf": {
        "types": ["Open Bookshelf", "Ladder Shelf", "Corner Shelf",
                  "Wall Bookshelf", "Display Shelf"],
        "materials": ["Solid Wood", "Engineered Wood", "Metal"],
        "colours": ["Brown", "White", "Black", "Oak"],
        "styles": ["Modern", "Rustic", "Minimalist", "Industrial"],
        "rooms": ["Living Room", "Home Office", "Bedroom"],
        "price": (2500, 45000),
        "dimensions": ((40, 150), (20, 50), (80, 220))
    },

    "Dining Furniture": {
        "types": ["Dining Table", "Dining Set", "Bar Stool",
                  "Kitchen Island", "Dining Bench"],
        "materials": ["Solid Wood", "Glass", "Metal", "Engineered Wood"],
        "colours": ["Brown", "Black", "White", "Walnut"],
        "styles": ["Modern", "Traditional", "Contemporary", "Rustic"],
        "rooms": ["Dining Room", "Kitchen"],
        "price": (5000, 150000),
        "dimensions": ((70, 250), (60, 120), (70, 110))
    },

    "Other Furniture": {
        "types": ["Ottoman", "Room Divider", "Footstool",
                  "Entryway Bench", "Vanity Table"],
        "materials": ["Wood", "Fabric", "Metal", "Engineered Wood"],
        "colours": ["Beige", "Brown", "White", "Grey", "Black"],
        "styles": ["Modern", "Minimalist", "Bohemian", "Contemporary"],
        "rooms": ["Living Room", "Bedroom", "Entryway"],
        "price": (1500, 40000),
        "dimensions": ((40, 160), (30, 90), (30, 150))
    }
}


# ==========================================
# 2. GENERATE PRODUCT DESCRIPTIONS
# ==========================================

def generate_description(name, category, material, colour,
                         style, room, dimensions):

    descriptions = [
        f"Enhance your {room.lower()} with this {style.lower()} "
        f"{name.lower()}. Crafted from {material.lower()}, "
        f"it features an elegant {colour.lower()} finish. "
        f"Designed to combine comfort, durability and functionality.",

        f"Bring a refined look to your home with this {colour.lower()} "
        f"{name.lower()}. Its {style.lower()} design and "
        f"{material.lower()} construction make it suitable for "
        f"your {room.lower()}.",

        f"This {style.lower()} {category.lower()} offers a practical "
        f"and stylish furniture solution. Made with {material.lower()}, "
        f"it complements a variety of interiors and adds character "
        f"to your {room.lower()}.",

        f"Designed for everyday use, this {name.lower()} combines "
        f"a {colour.lower()} appearance with a functional design. "
        f"Its {style.lower()} aesthetic makes it a versatile addition "
        f"to your home."
    ]

    description = random.choice(descriptions)

    return description + f" Dimensions: {dimensions}."


# ==========================================
# 3. GENERATE FURNITURE PRODUCTS
# ==========================================

def generate_products(count=NUM_PRODUCTS):

    products = []

    categories = list(FURNITURE.keys())

    for i in range(1, count + 1):

        category = categories[(i - 1) % len(categories)]

        details = FURNITURE[category]

        furniture_type = random.choice(details["types"])
        material = random.choice(details["materials"])
        colour = random.choice(details["colours"])
        style = random.choice(details["styles"])
        room = random.choice(details["rooms"])

        # Generate price and rating
        price = round(random.uniform(*details["price"]), 2)
        rating = round(random.uniform(3.5, 5.0), 1)

        # Generate dimensions
        length = random.randint(*details["dimensions"][0])
        width = random.randint(*details["dimensions"][1])
        height = random.randint(*details["dimensions"][2])

        dimensions = f"{length} x {width} x {height} cm"

        # Generate unique product name
        name = f"{style} {furniture_type} {i:04d}"

        # Generate realistic product description
        description = generate_description(
            name,
            category,
            material,
            colour,
            style,
            room,
            dimensions
        )

        products.append({
            "product_id": f"FUR{i:05d}",
            "name": name,
            "category": category,
            "material": material,
            "colour": colour,
            "style": style,
            "price": price,
            "rating": rating,
            "dimensions": dimensions,
            "room_type": room,
            "description": description,
            "data_source": "synthetic"
        })

    return pd.DataFrame(products)


# ==========================================
# 4. VALIDATE PRODUCT DATA
# ==========================================

def validate_products(df):

    assert len(df) >= 500, "Minimum 500 products required!"

    assert df["product_id"].is_unique, "Duplicate product IDs found!"

    required_columns = [
        "product_id", "name", "category", "material",
        "colour", "style", "price", "rating",
        "dimensions", "room_type", "description"
    ]

    assert all(col in df.columns for col in required_columns)

    assert df[required_columns].notnull().all().all()

    assert (df["price"] > 0).all()

    assert df["rating"].between(1, 5).all()

    assert (df["description"].str.len() > 30).all()

    print("All product validation checks passed!")


# ==========================================
# 5. SAVE PRODUCTS TO CSV
# ==========================================

def save_products(df):

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nProducts saved successfully!")
    print("File location:", OUTPUT_FILE)


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

def main():

    print("Starting UrbanNest Furniture Generation...")

    products = generate_products(NUM_PRODUCTS)

    validate_products(products)

    save_products(products)

    print("\n========== GENERATION SUMMARY ==========")

    print("Total products:", len(products))
    print("Unique product IDs:", products["product_id"].nunique())

    print("\nCategory distribution:")
    print(products["category"].value_counts())

    print("\nMaterial distribution:")
    print(products["material"].value_counts())

    print("\nAverage product price:", round(products["price"].mean(), 2))

    print("\nAverage rating:", round(products["rating"].mean(), 2))

    print("\nFirst five products:")
    print(products.head().to_string(index=False))


if __name__ == "__main__":
    main()
