
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

SEED = 42
NUMBER_OF_USERS = 2000

random.seed(SEED)

# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCT_FILE = BASE_DIR / "data" / "raw" / "products.csv"
OUTPUT_FILE = BASE_DIR / "data" / "raw" / "user_behavior.csv"


# -----------------------------------------
# LOAD PRODUCT CATALOGUE
# -----------------------------------------

def load_products():
    with open(PRODUCT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        products = list(reader)

    required_columns = {"product_id", "price", "category"}

    if not products:
        raise ValueError("Product catalogue is empty.")

    if not required_columns.issubset(products[0].keys()):
        raise ValueError("Required product columns are missing.")

    for product in products:
        product["price"] = float(product["price"])

    print("Products loaded:", len(products))
    return products


# -----------------------------------------
# GENERATE USER BEHAVIOUR
# -----------------------------------------

def generate_behaviour(products):
    events = []
    event_number = 1

    product_ids = [p["product_id"] for p in products]

    start_date = datetime(2025, 11, 7)
    end_date = datetime(2026, 11, 6)

    total_seconds = int((end_date - start_date).total_seconds())

    purchase_counts = defaultdict(int)

    for user_number in range(1, NUMBER_OF_USERS + 1):

        user_id = f"U{user_number:05d}"

        # Each user purchases between 1 and 10 different products
        number_of_purchases = random.randint(1, 10)

        purchased_products = random.sample(
            products,
            min(number_of_purchases, len(products))
        )

        purchase_counts[user_id] = len(purchased_products)

        # Generate browsing, cart and wishlist activity
        viewed_products = random.sample(
            products,
            min(random.randint(5, 15), len(products))
        )

        cart_products = random.sample(
            products,
            min(random.randint(1, 4), len(products))
        )

        wishlist_products = random.sample(
            products,
            min(random.randint(1, 3), len(products))
        )

        user_events = []

        def add_event(product, event_type):
            nonlocal event_number

            seconds = random.randint(0, total_seconds)
            event_time = start_date + timedelta(seconds=seconds)

            session_number = random.randint(1, 4)

            user_events.append({
                "event_id": f"E{event_number:07d}",
                "user_id": user_id,
                "product_id": product["product_id"],
                "event_type": event_type,
                "event_timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                "quantity": 1,
                "unit_price": round(product["price"], 2),
                "session_id": f"S{user_number:05d}_{session_number}"
            })

            event_number += 1

        for product in viewed_products:
            add_event(product, "view")

        for product in cart_products:
            add_event(product, "add_to_cart")

        for product in wishlist_products:
            add_event(product, "wishlist")

        for product in purchased_products:
            add_event(product, "purchase")

        events.extend(user_events)

    return events, purchase_counts


# -----------------------------------------
# SAVE CSV FILE
# -----------------------------------------

def save_events(events):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "event_id",
        "user_id",
        "product_id",
        "event_type",
        "event_timestamp",
        "quantity",
        "unit_price",
        "session_id"
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    print("\nDataset saved successfully!")
    print("Output:", OUTPUT_FILE)


# -----------------------------------------
# VALIDATE GENERATED DATA
# -----------------------------------------

def validate_data(events, products, purchase_counts):
    valid_product_ids = {p["product_id"] for p in products}

    event_types = Counter(e["event_type"] for e in events)

    user_ids = {e["user_id"] for e in events}

    used_product_ids = {e["product_id"] for e in events}

    assert len(user_ids) == NUMBER_OF_USERS
    assert used_product_ids.issubset(valid_product_ids)
    assert all(1 <= count <= 10 for count in purchase_counts.values())
    assert len({e["event_id"] for e in events}) == len(events)
    assert all(e["unit_price"] >= 0 for e in events)

    print("\nVALIDATION RESULTS")
    print("--------------------------")
    print("Unique users:", len(user_ids))
    print("Total activity records:", len(events))
    print("Event types:", dict(event_types))
    print("Minimum purchases per user:", min(purchase_counts.values()))
    print("Maximum purchases per user:", max(purchase_counts.values()))
    print("All product IDs valid: YES")
    print("All event IDs unique: YES")
    print("All validation checks passed!")


# -----------------------------------------
# MAIN PROGRAM
# -----------------------------------------

def main():
    print("Starting User Behaviour Generation...")

    products = load_products()

    if len(products) < 1:
        raise ValueError("No products available.")

    events, purchase_counts = generate_behaviour(products)

    save_events(events)

    validate_data(events, products, purchase_counts)


if __name__ == "__main__":
    main()
