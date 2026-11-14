import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.recommender import recommend_for_customer

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Furniture Recommendation System",
    page_icon="🛋️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #ffffff;
}

h1 {
    color: #30364a;
}

h2 {
    color: #30364a;
}

h3 {
    color: #30364a;
}

.metric-card {
    background-color: #f0f4fa;
    padding: 20px;
    border-radius: 10px;
}

.preference-card {
    background-color: #e8f1ff;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.product-card {
    background-color: #ffffff;
    padding: 15px;
    border: 1px solid #dddddd;
    border-radius: 10px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================

st.title("🛋️ Furniture Recommendation System")

st.subheader(
    "AI-Based Customer Segmentation and Personalized Product Recommendations"
)

st.write(
    "Discover furniture recommendations based on customer "
    "preferences, purchasing behaviour and product ratings."
)

st.divider()


# =====================================================
# LOAD DATASETS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_project_file(filename):
    """Find a file in the project root or common subfolders."""
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "customer_feature_engineering", filename),
        os.path.join(BASE_DIR, "outputs", filename),
        os.path.join(BASE_DIR, "output_s", filename),
        os.path.join(BASE_DIR, "customer_feature_engineering", "outputs", filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    for root, _, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)

    return None


@st.cache_data
def load_data():
    feature_file = find_project_file("user_features.parquet")
    if feature_file is None:
        raise FileNotFoundError(
            "user_features.parquet not found. Run "
            "python customer_feature_engineering/customer_features.py first."
        )

    products_file = os.path.join(BASE_DIR, "data", "raw", "products.csv")
    interactions_file = os.path.join(BASE_DIR, "data", "raw", "user_behavior.csv")
    if not os.path.isfile(products_file):
        raise FileNotFoundError(f"Product catalogue not found: {products_file}")
    if not os.path.isfile(interactions_file):
        raise FileNotFoundError(f"Customer interaction data not found: {interactions_file}")

    customer_features = pd.read_parquet(feature_file)
    products = pd.read_csv(products_file)
    interactions = pd.read_csv(interactions_file)

    customer_features.columns = customer_features.columns.str.strip().str.lower()
    products.columns = products.columns.str.strip().str.lower()
    interactions.columns = interactions.columns.str.strip().str.lower()

    customer_features["user_id"] = (
        customer_features["user_id"].astype(str).str.strip()
    )
    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()

    recommendations = []
    for customer_id in customer_features["user_id"].drop_duplicates():
        customer_recommendations = recommend_for_customer(
            customer_id,
            products,
            customer_features,
            interactions,
            number_of_recommendations=5,
        )
        customer_recommendations["user_id"] = customer_id
        recommendations.append(customer_recommendations)

    if recommendations:
        all_recommendations = pd.concat(recommendations, ignore_index=True)
    else:
        all_recommendations = pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
                "colour",
                "style",
                "price",
                "rating",
                "recommendation_score",
                "user_id",
            ]
        )

    return all_recommendations, customer_features


try:
    all_recommendations, customer_features = load_data()
except Exception as e:
    st.error(f"Unable to load data: {e}")
    st.info(
        "Check the raw data files and regenerate user_features.parquet if needed."
    )
    st.stop()
# =====================================================
# DASHBOARD OVERVIEW
# =====================================================

st.header("📊 Dashboard Overview")

total_customers = customer_features["user_id"].nunique()

total_recommendations = len(all_recommendations)

average_score = all_recommendations[
    "recommendation_score"
].mean()

average_rating = all_recommendations[
    "rating"
].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    total_customers
)

col2.metric(
    "Total Recommendations",
    total_recommendations
)

col3.metric(
    "Average Recommendation Score",
    f"{average_score:.2f}"
)

col4.metric(
    "Average Product Rating",
    f"{average_rating:.2f}"
)

st.divider()


# =====================================================
# CUSTOMER ANALYSIS
# =====================================================

st.header("👤 Customer Analysis")

customer_list = sorted(
    customer_features["user_id"].unique()
)

selected_customer = st.selectbox(
    "Select Customer ID",
    customer_list
)


# =====================================================
# CUSTOMER INFORMATION
# =====================================================

st.subheader("Customer Information")

customer_info = customer_features[
    customer_features["user_id"] == selected_customer
]

if not customer_info.empty:

    info = customer_info.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Purchases",
        int(info["total_purchases"])
    )

    col2.metric(
        "Total Spending",
        (
            f"₹{info['total_spending']:,.2f}"
            if pd.notna(info["total_spending"])
            else "Unavailable"
        )
    )

    col3.metric(
        "Average Order Value",
        (
            f"₹{info['average_order_value']:,.2f}"
            if pd.notna(info["average_order_value"])
            else "Unavailable"
        )
    )

    if info["unpriced_purchase_events"] > 0:
        st.caption(
            "Spending is unavailable because at least one purchased product has no price in the source catalogue."
        )

else:

    st.warning("Customer information not available.")


st.divider()


# =====================================================
# CUSTOMER PREFERENCES
# =====================================================

st.subheader("💗 Customer Preferences")

customer_recommendations = all_recommendations[
    all_recommendations["user_id"] == selected_customer
].copy()
customer_profile = customer_info.iloc[0] if not customer_info.empty else None

if customer_profile is not None:

    pref1, pref2, pref3 = st.columns(3)

    with pref1:

        st.info(
            f"Favourite Category\n\n"
            f"**{str(customer_profile['favorite_category']).title()}**"
        )

    with pref2:

        st.info(
            f"Favourite Colour\n\n"
            f"**{str(customer_profile['favorite_colour']).title()}**"
        )

    with pref3:

        st.info(
            f"Favourite Style\n\n"
            f"**{str(customer_profile['favorite_style']).title()}**"
        )

    st.caption(
        "Preferences are inferred from weighted browsing and purchase interactions."
    )

else:

    st.warning("No preferences available for this customer.")


st.divider()


# =====================================================
# PERSONALIZED RECOMMENDATIONS
# =====================================================

st.header("✨ Personalized Product Recommendations")

if customer_recommendations.empty:

    st.warning(
        "No recommendations available for this customer."
    )

else:

    max_products = min(
        10,
        len(customer_recommendations)
    )

    recommendation_count = st.slider(
        "Number of products to display",
        min_value=1,
        max_value=max_products,
        value=min(5, max_products)
    )

    customer_recommendations = (
        customer_recommendations.sort_values(
            "recommendation_score",
            ascending=False
        )
    )

    displayed_recommendations = (
        customer_recommendations.head(recommendation_count)
    )

    st.success(
        f"Showing personalized recommendations for {selected_customer}"
    )

    # Display products in two columns

    for start in range(
        0,
        len(displayed_recommendations),
        2
    ):

        cols = st.columns(2)

        for col, (_, product) in zip(
            cols,
            displayed_recommendations.iloc[
                start:start + 2
            ].iterrows()
        ):

            with col:

                st.subheader(
                    "🪑 " + str(product["product_name"])
                )

                st.write(
                    f"**Product ID:** {product['product_id']}"
                )

                st.write(
                    f"**Category:** {product['category']}"
                )

                st.write(
                    f"**Colour:** {product['colour']}"
                )

                st.write(
                    f"**Style:** {product['style']}"
                )

                st.write(
                    (
                        f"**Price:** ₹{product['price']:,.2f}"
                        if pd.notna(product["price"])
                        else "**Price:** Unavailable"
                    )
                )

                st.write(
                    f"**Rating:** ⭐ {product['rating']}"
                )

                score = float(
                    product["recommendation_score"]
                )

                st.write(
                    f"**Recommendation Score:** {score:.2f}"
                )

                st.progress(
                    max(0.0, min(1.0, score))
                )

                st.divider()


# =====================================================
# RECOMMENDATION DETAILS TABLE
# =====================================================

st.header("📋 Recommendation Details")

if not customer_recommendations.empty:

    display_columns = [
        "product_id",
        "product_name",
        "category",
        "colour",
        "style",
        "price",
        "rating",
        "recommendation_score"
    ]

    recommendation_table = displayed_recommendations[
        display_columns
    ].copy()

    st.dataframe(
        recommendation_table,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# =====================================================
# RECOMMENDATION ANALYTICS
# =====================================================

st.header("📈 Recommendation Analytics")

if not customer_recommendations.empty:

    chart1, chart2 = st.columns(2)

    # ---------------------------------------------
    # RECOMMENDATION SCORE CHART
    # ---------------------------------------------

    with chart1:

        st.subheader("Recommendation Scores")

        score_chart = (
            displayed_recommendations.set_index(
                "product_name"
            )["recommendation_score"]
        )

        st.bar_chart(score_chart)

    # ---------------------------------------------
    # PRODUCT PRICE CHART
    # ---------------------------------------------

    with chart2:

        st.subheader("Product Price Comparison")

        price_chart = (
            displayed_recommendations.set_index(
                "product_name"
            )["price"]
        )

        st.bar_chart(price_chart)


st.divider()


# =====================================================
# CUSTOMER SPENDING ANALYSIS
# =====================================================

st.header("💰 Customer Spending Analysis")

spending_chart = customer_features.set_index(
    "user_id"
)["total_spending"].dropna()

if spending_chart.empty:
    st.info("No complete spending values are available from the source data.")
else:
    st.bar_chart(
        spending_chart,
        x_label="Customer ID",
        y_label="Total Spending (₹)"
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">

    <h4>🛋️ Furniture Recommendation System</h4>

    <p>
    AI-Based Customer Segmentation and Personalized
    Product Recommendations
    </p>

    <p>
    Developed using Python, Pandas, Machine Learning
    and Streamlit.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)