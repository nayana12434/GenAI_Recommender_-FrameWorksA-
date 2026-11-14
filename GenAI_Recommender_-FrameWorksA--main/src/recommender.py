"""Content-based and customer-personalized furniture recommendations."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


PREFERENCE_WEIGHTS = {
    "category": 0.4,
    "colour": 0.3,
    "style": 0.3,
}


def _normalize_value(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _preference_score(customer: pd.Series, attribute: str, value: object) -> float:
    preference_column = (
        f"{attribute}_pref__{_normalize_value(value)}"
    )
    if preference_column in customer.index:
        score = pd.to_numeric(customer[preference_column], errors="coerce")
        return float(score) if pd.notna(score) else 0.0

    favorite_column = f"favorite_{attribute}"
    if favorite_column in customer.index and pd.notna(customer[favorite_column]):
        return float(
            _normalize_value(customer[favorite_column]) == _normalize_value(value)
        )
    return 0.0


def recommend_for_customer(
    customer_id: object,
    products: pd.DataFrame,
    customer_features: pd.DataFrame,
    interactions: pd.DataFrame | None = None,
    number_of_recommendations: int = 5,
) -> pd.DataFrame:
    """Rank products using a customer's learned category, colour, and style preferences."""
    if number_of_recommendations < 1:
        raise ValueError("number_of_recommendations must be at least 1")

    required_product_columns = {
        "product_id",
        "product_name",
        "category",
        "colour",
        "style",
        "price",
        "rating",
    }
    missing_product_columns = required_product_columns.difference(products.columns)
    if missing_product_columns:
        raise ValueError(
            "Product data is missing columns: "
            + ", ".join(sorted(missing_product_columns))
        )
    if "user_id" not in customer_features.columns:
        raise ValueError("Customer features are missing the user_id column.")

    customers = customer_features.loc[
        customer_features["user_id"].astype(str).str.strip()
        == str(customer_id).strip()
    ]
    if customers.empty:
        raise ValueError(f"No customer features found for {customer_id}.")
    if len(customers) > 1:
        raise ValueError(f"Customer features are not unique for {customer_id}.")
    customer = customers.iloc[0]

    candidates = products.copy()
    candidates["product_id"] = candidates["product_id"].astype(str).str.strip()
    candidates = candidates.drop_duplicates("product_id", keep="first")

    if interactions is not None and not interactions.empty:
        required_interaction_columns = {"user_id", "product_id", "event_type"}
        missing_interaction_columns = required_interaction_columns.difference(
            interactions.columns
        )
        if missing_interaction_columns:
            raise ValueError(
                "Interaction data is missing columns: "
                + ", ".join(sorted(missing_interaction_columns))
            )
        purchases = interactions.loc[
            interactions["user_id"].astype(str).str.strip()
            == str(customer_id).strip()
        ]
        purchased_ids = set(
            purchases.loc[
                purchases["event_type"].astype(str).str.strip().str.casefold()
                == "purchase",
                "product_id",
            ]
            .astype(str)
            .str.strip()
        )
        candidates = candidates.loc[~candidates["product_id"].isin(purchased_ids)]

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
                "colour",
                "style",
                "price",
                "rating",
                "recommendation_score",
            ]
        )

    def score_product(product: pd.Series) -> float:
        return sum(
            weight
            * _preference_score(customer, attribute, product[attribute])
            for attribute, weight in PREFERENCE_WEIGHTS.items()
        )

    candidates["recommendation_score"] = candidates.apply(
        score_product, axis=1
    )
    candidates["rating"] = pd.to_numeric(candidates["rating"], errors="coerce")
    candidates["price"] = pd.to_numeric(candidates["price"], errors="coerce")

    return (
        candidates.sort_values(
            ["recommendation_score", "rating", "product_name"],
            ascending=[False, False, True],
            na_position="last",
        )
        .loc[
            :,
            [
                "product_id",
                "product_name",
                "category",
                "colour",
                "style",
                "price",
                "rating",
                "recommendation_score",
            ],
        ]
        .head(number_of_recommendations)
        .reset_index(drop=True)
    )


def cosine_similarity(row_a, row_b):
    """Return cosine similarity for two equally sized feature vectors."""
    if len(row_a) != len(row_b):
        raise ValueError("Feature vectors must have the same length.")
    dot_product = sum(a * b for a, b in zip(row_a, row_b))
    magnitude_a = math.sqrt(sum(a * a for a in row_a))
    magnitude_b = math.sqrt(sum(b * b for b in row_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def recommend_products(
    product_id: object,
    number_of_recommendations: int = 5,
) -> None:
    """Print similar-product recommendations from the engineered feature file."""
    feature_file = Path(__file__).resolve().parents[1] / "outputs" / "product_features.csv"
    products = pd.read_csv(feature_file)
    if "product_id" not in products.columns:
        raise ValueError("Product features are missing the product_id column.")
    if number_of_recommendations < 1:
        raise ValueError("number_of_recommendations must be at least 1")

    feature_columns = [column for column in products.columns if column != "product_id"]
    matching = products.index[products["product_id"] == product_id]
    if len(matching) == 0:
        raise ValueError(f"Product ID {product_id} was not found.")

    selected_index = matching[0]
    selected_features = products.loc[selected_index, feature_columns].tolist()
    scored = []

    for index, row in products.iterrows():
        if index == selected_index:
            continue
        score = cosine_similarity(selected_features, row[feature_columns].tolist())
        scored.append((index, score))

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[
        :number_of_recommendations
    ]
    print(f"\nRecommendations for product ID {product_id}:")
    for index, score in ranked:
        print(f"Product ID: {products.loc[index, 'product_id']} | Similarity: {score:.3f}")


if __name__ == "__main__":
    features_file = Path(__file__).resolve().parents[1] / "outputs" / "product_features.csv"
    feature_products = pd.read_csv(features_file)
    recommend_products(feature_products["product_id"].iloc[0])
