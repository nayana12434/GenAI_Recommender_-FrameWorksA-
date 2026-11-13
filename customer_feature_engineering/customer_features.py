"""Build customer RFM, interaction, spending, and preference features."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

PREFERENCE_WEIGHTS = {
    "view": 1.0,
    "wishlist": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}
ENGAGEMENT_WEIGHTS = {**PREFERENCE_WEIGHTS, "click": 1.0}
PREFERENCE_ATTRIBUTES = ("category", "colour", "style", "material", "room_type")
REQUIRED_PRODUCT_ATTRIBUTES = ("category", "colour", "style")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in frame.columns
    ]
    return frame


def _require_columns(
    frame: pd.DataFrame, required: set[str], source_name: str
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _slug(value: object) -> str:
    return re.sub(
        r"[^a-z0-9]+", "_", str(value).strip().casefold()
    ).strip("_") or "unknown"


def _canonical_event(value: object) -> str:
    normalized = _slug(value)
    aliases = {
        "addtocart": "add_to_cart",
        "cart": "add_to_cart",
        "add_to_wishlist": "wishlist",
        "wish_list": "wishlist",
        "wishlisted": "wishlist",
        "product_view": "view",
        "page_view": "view",
        "product_click": "click",
    }
    return aliases.get(normalized, normalized)


def _score_rfm_component(
    values: pd.Series, eligible: pd.Series, higher_is_better: bool
) -> pd.Series:
    scores = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid_values = values.loc[eligible & values.notna()]
    if valid_values.empty:
        return scores
    if valid_values.nunique() == 1:
        scores.loc[valid_values.index] = 3
        return scores

    percentile_rank = valid_values.rank(
        method="average", pct=True, ascending=higher_is_better
    )
    scores.loc[valid_values.index] = (
        np.ceil(percentile_rank * 5).clip(1, 5).astype("int64")
    )
    return scores


def _customer_segments(features: pd.DataFrame) -> pd.Series:
    segments = pd.Series("Regular Customers", index=features.index, dtype="string")
    no_purchase = features["frequency"].eq(0)
    incomplete_spending = (
        features["frequency"].gt(0) & features["monetary_score"].isna()
    )
    segments.loc[no_purchase] = "No Purchases"
    segments.loc[incomplete_spending] = "Spending Data Incomplete"

    eligible = ~(no_purchase | incomplete_spending)
    r = features["recency_score"]
    f = features["frequency_score"]
    m = features["monetary_score"]

    segments.loc[eligible & r.ge(4) & f.ge(4) & m.ge(4)] = "Champions"
    segments.loc[eligible & r.ge(3) & f.ge(4)] = "Loyal Customers"
    segments.loc[eligible & r.ge(4) & f.ge(2)] = "Potential Loyalists"
    segments.loc[eligible & r.ge(4) & f.eq(1)] = "Recent Customers"
    segments.loc[eligible & r.le(2) & f.ge(2)] = "At Risk"
    segments.loc[eligible & r.le(2) & f.eq(1)] = "Needs Attention"
    return segments


def build_customer_features(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Create one validated feature row per customer found in interaction data."""
    project_root = Path(project_root)
    behavior_path = project_root / "data" / "raw" / "user_behavior.csv"
    products_path = project_root / "data" / "raw" / "products.csv"

    if not behavior_path.is_file():
        raise FileNotFoundError(f"Behavior data not found: {behavior_path}")
    if not products_path.is_file():
        raise FileNotFoundError(f"Product data not found: {products_path}")

    behavior = _normalize_columns(pd.read_csv(behavior_path))
    products = _normalize_columns(pd.read_csv(products_path))
    _require_columns(
        behavior,
        {"user_id", "product_id", "event_type", "event_timestamp"},
        "Behavior data",
    )
    _require_columns(
        products,
        {"product_id", "price", *REQUIRED_PRODUCT_ATTRIBUTES},
        "Product data",
    )

    if "quantity" not in behavior:
        behavior["quantity"] = 0
    if "session_id" not in behavior:
        behavior["session_id"] = pd.NA
    if "discount" not in products:
        products["discount"] = 0

    behavior["user_id"] = behavior["user_id"].astype("string").str.strip()
    behavior["product_id"] = behavior["product_id"].astype("string").str.strip()
    behavior["event_type"] = behavior["event_type"].map(_canonical_event)
    behavior["event_timestamp"] = pd.to_datetime(
        behavior["event_timestamp"], errors="coerce"
    )
    behavior["quantity"] = pd.to_numeric(behavior["quantity"], errors="coerce")
    behavior["session_id"] = behavior["session_id"].astype("string").str.strip()
    behavior["session_id"] = behavior["session_id"].replace("", pd.NA)

    if behavior[["user_id", "product_id", "event_type"]].isna().any().any():
        raise ValueError("Behavior data contains missing user, product, or event IDs.")
    if behavior["user_id"].eq("").any() or behavior["product_id"].eq("").any():
        raise ValueError("Behavior data contains blank user or product IDs.")
    if behavior["event_timestamp"].isna().any():
        raise ValueError("Behavior data contains invalid event timestamps.")

    products["product_id"] = products["product_id"].astype("string").str.strip()
    if products["product_id"].isna().any() or products["product_id"].eq("").any():
        raise ValueError("Product data contains missing or blank product IDs.")
    for attribute in PREFERENCE_ATTRIBUTES:
        if attribute in products:
            products[attribute] = (
                products[attribute]
                .astype("string")
                .str.strip()
                .str.casefold()
                .replace("", pd.NA)
            )
    products["price"] = pd.to_numeric(products["price"], errors="coerce")
    products["discount"] = (
        pd.to_numeric(products["discount"], errors="coerce").fillna(0).clip(0, 100)
    )
    if products["price"].lt(0).any():
        raise ValueError("Product catalogue contains negative prices.")

    products = products.drop_duplicates("product_id", keep="first")
    product_columns = [
        "product_id",
        "price",
        "discount",
        *[col for col in PREFERENCE_ATTRIBUTES if col in products.columns],
    ]
    events = behavior.merge(
        products[product_columns],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    events["is_purchase"] = events["event_type"].eq("purchase")
    invalid_purchase_quantity = events["is_purchase"] & (
        events["quantity"].isna() | events["quantity"].le(0)
    )
    if invalid_purchase_quantity.any():
        rows = (events.index[invalid_purchase_quantity] + 2).tolist()
        raise ValueError(
            "Purchase events must have a positive quantity; check CSV row(s): "
            + ", ".join(map(str, rows))
        )

    events["preference_weight"] = (
        events["event_type"].map(PREFERENCE_WEIGHTS).fillna(0.0)
    )
    events["engagement_weight"] = (
        events["event_type"].map(ENGAGEMENT_WEIGHTS).fillna(0.0)
    )
    events["purchase_amount"] = (
        events["price"] * events["quantity"]
    ).where(events["is_purchase"])

    customer_ids = pd.Index(
        events["user_id"].drop_duplicates().astype(str), name="user_id"
    )
    features = pd.DataFrame(index=customer_ids)
    features["total_interactions"] = events.groupby("user_id").size().reindex(
        customer_ids, fill_value=0
    )
    event_counts = pd.crosstab(events["user_id"], events["event_type"])
    count_columns = {
        "total_product_views": ("view",),
        "total_clicks": ("click",),
        "total_add_to_cart": ("add_to_cart",),
        "total_wishlist": ("wishlist",),
        "total_purchases": ("purchase",),
    }
    for output_column, event_names in count_columns.items():
        counts = pd.Series(0, index=customer_ids, dtype="int64")
        for event_name in event_names:
            event_count = event_counts.get(event_name)
            if event_count is not None:
                counts = counts.add(event_count.reindex(customer_ids, fill_value=0))
        features[output_column] = counts

    features["frequency"] = features["total_purchases"].astype("int64")
    features["purchase_frequency"] = features["frequency"]

    purchase_events = events.loc[events["is_purchase"]].copy()
    features["total_quantity"] = (
        purchase_events.groupby("user_id")["quantity"]
        .sum()
        .reindex(customer_ids, fill_value=0)
    )
    spending = purchase_events.groupby("user_id")["purchase_amount"].sum(
        min_count=1
    )
    unpriced = purchase_events.assign(
        has_missing_price=purchase_events["price"].isna()
    ).groupby("user_id")["has_missing_price"].sum()
    features["unpriced_purchase_events"] = (
        unpriced.reindex(customer_ids, fill_value=0).astype("int64")
    )
    features["total_spending"] = spending.reindex(customer_ids).fillna(0)
    incomplete_spending = features["unpriced_purchase_events"].gt(0)
    features.loc[incomplete_spending, "total_spending"] = np.nan
    features["monetary"] = features["total_spending"]
    features["average_order_value"] = (
        features["total_spending"]
        / features["total_purchases"].replace(0, np.nan)
    )
    features.loc[features["frequency"].eq(0), "average_order_value"] = 0

    last_purchase = purchase_events.groupby("user_id")["event_timestamp"].max()
    latest_event_date = events["event_timestamp"].max().normalize()
    reference_date = latest_event_date + pd.Timedelta(days=1)
    features["recency_days"] = (
        reference_date - last_purchase.dt.normalize()
    ).dt.days.reindex(customer_ids)
    features.loc[features["frequency"].eq(0), "recency_days"] = np.nan

    sessions_available = events["session_id"].notna().any()
    if sessions_available:
        session_events = events.loc[events["session_id"].notna()]
        total_sessions = session_events.groupby("user_id")["session_id"].nunique()
        purchasing_sessions = (
            session_events.loc[session_events["is_purchase"]]
            .groupby("user_id")["session_id"]
            .nunique()
        )
        features["total_sessions"] = total_sessions.reindex(
            customer_ids, fill_value=0
        )
        features["purchasing_sessions"] = purchasing_sessions.reindex(
            customer_ids, fill_value=0
        )
        features["conversion_rate_basis"] = "purchasing_sessions / total_sessions"
    else:
        features["total_sessions"] = features["total_interactions"]
        features["purchasing_sessions"] = features["total_purchases"]
        features["conversion_rate_basis"] = "purchase_events / total_interactions"
    features["purchase_conversion_rate"] = (
        features["purchasing_sessions"]
        .div(features["total_sessions"].replace(0, np.nan))
        .fillna(0)
    )
    features["customer_engagement_score"] = (
        events.groupby("user_id")["engagement_weight"]
        .sum()
        .reindex(customer_ids, fill_value=0)
    )
    features["engagement_score"] = features["customer_engagement_score"]
    features["unknown_event_count"] = (
        ~events["event_type"].isin(
            {*ENGAGEMENT_WEIGHTS, "unknown"}
        )
    ).groupby(events["user_id"]).sum().reindex(customer_ids, fill_value=0)

    for attribute in PREFERENCE_ATTRIBUTES:
        attribute_available = attribute in products.columns
        features[f"{attribute}_attribute_available"] = attribute_available
        favorite_column = f"favorite_{attribute}"
        features[favorite_column] = pd.Series(
            pd.NA, index=customer_ids, dtype="string"
        )
        if not attribute_available:
            continue

        attribute_events = events.loc[
            events[attribute].notna() & events["preference_weight"].gt(0),
            ["user_id", attribute, "preference_weight"],
        ].copy()
        attribute_events["preference_value"] = attribute_events[attribute].map(_slug)
        value_names = (
            events[attribute].dropna().drop_duplicates().astype(str).to_dict()
        )
        slug_labels: dict[str, str] = {}
        for value in value_names.values():
            slug_labels.setdefault(_slug(value), value)
        preference_columns = sorted(slug_labels)

        if attribute_events.empty:
            weighted = pd.DataFrame(index=customer_ids, columns=preference_columns)
            weighted = weighted.fillna(0.0)
        else:
            weighted = attribute_events.pivot_table(
                index="user_id",
                columns="preference_value",
                values="preference_weight",
                aggfunc="sum",
                fill_value=0,
            ).reindex(
                index=customer_ids,
                columns=preference_columns,
                fill_value=0,
            )
        totals = weighted.sum(axis=1)
        normalized = weighted.div(totals.where(totals.ne(0)), axis=0).fillna(0)

        has_preference = totals.gt(0)
        if has_preference.any():
            favorite_slug = weighted.loc[has_preference].idxmax(axis=1)
            features.loc[has_preference, favorite_column] = favorite_slug.map(
                slug_labels
            )

        for value_slug in preference_columns:
            features[f"{attribute}_weighted__{value_slug}"] = weighted[
                value_slug
            ].astype(float)
            features[f"{attribute}_pref__{value_slug}"] = normalized[
                value_slug
            ].astype(float)

    eligible_purchasers = features["frequency"].gt(0)
    features["recency_score"] = _score_rfm_component(
        features["recency_days"], eligible_purchasers, higher_is_better=False
    )
    features["frequency_score"] = _score_rfm_component(
        features["frequency"], eligible_purchasers, higher_is_better=True
    )
    features["monetary_score"] = _score_rfm_component(
        features["monetary"], eligible_purchasers, higher_is_better=True
    )
    no_purchase = features["frequency"].eq(0)
    for column in ("recency_score", "frequency_score", "monetary_score"):
        features.loc[no_purchase, column] = 1
    features["rfm_score"] = features.apply(
        lambda row: "-".join(
            "NA" if pd.isna(row[column]) else str(int(row[column]))
            for column in ("recency_score", "frequency_score", "monetary_score")
        ),
        axis=1,
    )
    features["customer_segment"] = _customer_segments(features)
    features["rfm_reference_date"] = reference_date.date().isoformat()
    features["spending_basis"] = "catalogue list price x purchase quantity; discounts not applied"
    features["purchase_frequency_basis"] = "purchase event count; no order ID in source data"

    features.index.name = "user_id"
    features = features.reset_index()
    if features["user_id"].isna().any() or features["user_id"].str.strip().eq("").any():
        raise ValueError("Generated customer features contain missing customer IDs.")
    if features["user_id"].duplicated().any():
        raise ValueError("Generated customer features contain duplicate customer IDs.")
    if features["total_spending"].dropna().lt(0).any():
        raise ValueError("Generated customer features contain negative spending.")
    return features


def _build_spending_output(features: pd.DataFrame) -> pd.DataFrame:
    return features[
        [
            "user_id",
            "total_purchases",
            "total_spending",
            "total_quantity",
            "average_order_value",
            "purchase_frequency",
            "unpriced_purchase_events",
        ]
    ].rename(
        columns={
            "total_purchases": "Total_Purchases",
            "total_spending": "Total_Spending",
            "total_quantity": "Total_Quantity",
            "average_order_value": "Average_Order_Value",
            "purchase_frequency": "Purchase_Frequency",
            "unpriced_purchase_events": "Unpriced_Purchase_Events",
        }
    )


def main() -> None:
    features = build_customer_features()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "user_features.parquet"
    csv_path = OUTPUT_DIR / "customer_features.csv"
    spending_path = OUTPUT_DIR / "customer_spending.csv"

    features.to_parquet(parquet_path, index=False)
    features.to_csv(csv_path, index=False)
    _build_spending_output(features).to_csv(spending_path, index=False)

    legacy_spending_path = PROJECT_ROOT / "output_s" / "customer_spending.csv"
    if legacy_spending_path.parent.is_dir():
        _build_spending_output(features).to_csv(legacy_spending_path, index=False)

    print(f"Saved customer features: {parquet_path}")
    print(f"Saved customer features: {csv_path}")
    print(f"Saved customer spending: {spending_path}")
    if legacy_spending_path.parent.is_dir():
        print(f"Updated existing spending file: {legacy_spending_path}")
    print(
        features[
            [
                "user_id",
                "recency_days",
                "frequency",
                "monetary",
                "rfm_score",
                "customer_segment",
                "favorite_category",
                "favorite_colour",
                "favorite_style",
                "favorite_material",
                "favorite_room_type",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
