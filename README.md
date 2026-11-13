# Furniture Recommendation System

A content-based furniture recommendation application built with Python,
Pandas, and Streamlit. Users select a furniture item and receive similar
product suggestions based on engineered product features. The customer
dashboard uses browsing and purchase history to rank products by each
customer's learned category, colour, and style preferences.

## Project overview

The project demonstrates a simple end-to-end recommendation workflow:

1.  Clean the raw product and user-behaviour data.
2.  Explore the data and create exploratory charts.
3.  Transform product attributes into numerical features.
4.  Calculate product-to-product similarity using cosine similarity.
5.  Present recommendations through a Streamlit web application.

## Features

-   Select a furniture product from the available catalogue.
-   Recommend similar products using cosine similarity.
-   Display available product details such as name, category, material,
    colour, price, and rating.
-   Filter the catalogue by category and budget (where those fields are
    available in the source data).
-   Adjust the number of recommendations displayed.

## Project structure

``` text
furniture/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── products.csv
│   │   └── user_behavior.csv
│   └── processed/
│       ├── products_clean.csv
│       └── user_behavior_clean.csv
├── outputs/
│   ├── product_features.csv
│   └── user_interactions.csv
├── customer_feature_engineering/
│   ├── customer_features.py
│   └── outputs/
│       ├── user_features.parquet
│       ├── customer_features.csv
│       └── customer_spending.csv
└── src/
    ├── data_cleaner.py
    ├── eda.py
    ├── feature_engineering.py
    └── recommender.py
```

## Requirements

-   Python 3.12 (the project was developed in a Python 3.12 virtual
    environment)
-   pip
-   A web browser

Install dependencies from the project folder:

``` bash
python -m pip install -r requirements.txt
```

If you are using Windows, activate your virtual environment first:

``` cmd
.venv312\Scripts\activate
```

## Run the application

From the project root folder, run:

``` bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

``` text
http://localhost:8501
```

Keep the terminal open while using the application. Press `Ctrl+C` in
the terminal to stop the Streamlit server.

## Rebuild the data pipeline

Run these scripts from the project root, in this order, if you need to
regenerate the processed data and features:

``` bash
python src/data_cleaner.py
python src/eda.py
python src/feature_engineering.py
python customer_feature_engineering/customer_features.py
```

Then start the application with:

``` bash
python -m streamlit run app.py
```

The EDA script may display charts. The data-cleaning and
feature-engineering scripts should complete before the application is
launched.

## How recommendations work

The product-to-product recommender compares engineered catalogue feature
vectors using cosine similarity:

\[
`\operatorname{cosine}`{=tex}(A,B)=`\frac{A\cdot B}{\|A\|\|B\|}`{=tex}
\]

A higher score indicates greater similarity between the represented
feature vectors. The selected product itself is excluded from its
recommendation list, and the remaining products are sorted by similarity
score.

The customer dashboard uses `customer_feature_engineering/outputs/user_features.parquet`
to score each catalogue product against the customer's weighted category,
colour, and style preference vectors. Category, colour, and style
contribute 0.4, 0.3, and 0.3 of the score respectively; product rating
breaks ties. Products the customer has purchased are excluded. The profile
file is generated from the raw interaction and product CSV files by
`customer_feature_engineering/customer_features.py`.

The customer feature builder also writes a CSV copy, RFM scores and
segments, purchase-only spending metrics, interaction counts, conversion
rates, and engagement scores. Preference weights are view=1,
wishlist=2, add-to-cart=3, and purchase=5. Clicks are counted for behavior
and engagement, but are not given a preference weight. Material and room
type preferences are generated when those product columns are present.
Wishlist counts are zero when no wishlist events occur in the input.

## Data

The project uses two CSV input files:

-   `data/raw/products.csv`: product catalogue information.
-   `data/raw/user_behavior.csv`: user-behaviour records.

Processed data and generated feature files are stored under
`data/processed/` and `outputs/`.

The sample dataset is small and is suitable for demonstrating the
workflow. Confirm that you have permission to share the source data
before publishing the repository. Do not commit private, licensed, or
personally identifiable data.

## Current scope and limitations

-   Customer recommendations learn preferences from recorded
    interactions, but the sample history is too small to represent
    long-term preferences reliably.
-   Frequency counts purchase events because the sample data does not
    contain a separate order identifier.
-   Spending uses catalogue list price multiplied by purchased quantity;
    actual transaction prices and applied discounts are not available.
    Spending is unavailable when any purchased product has no catalogue
    price, and is flagged rather than estimated.
-   Recommendation quality depends on the available catalogue, feature
    encoding, and feature scaling.
-   A similarity score is a mathematical comparison, not a guarantee
    that a user will like or purchase a product.
-   The project uses a small sample catalogue; larger and more diverse
    data is needed for meaningful evaluation.
-   The current interface displays product details available in the CSV
    data. Product photos and live inventory or purchasing integrations
    are not included unless separately added.

## Suggested future improvements

-   Add image URLs and product photographs.
-   Evaluate recommendations using a larger labelled dataset and
    suitable ranking metrics.
-   Add automated tests for data loading, ID matching, filters, and
    recommendation output.
-   Deploy the Streamlit application and document the deployment
    environment.

## Testing checklist

-   [ ] All input CSV files are present.
-   [ ] Data-cleaning script runs successfully.
-   [ ] Feature-engineering script creates the expected output files.
-   [ ] Customer feature script creates `user_features.parquet`.
-   [ ] Customer feature script creates `customer_features.csv`.
-   [ ] Streamlit app starts without errors.
-   [ ] Product names and details are displayed correctly.
-   [ ] Category and budget filters behave as expected.
-   [ ] Recommendations appear for more than one selected product.
-   [ ] Customer recommendations reflect interaction preferences and
    exclude previously purchased products.
-   [ ] The selected product is not recommended to itself.

## Author and project details

Complete these fields before submission:

-   **Student name:** \[Your name\]
-   **Institution:** \[Institution name\]
-   **Course / department:** \[Course or department\]
-   **Project guide:** \[Guide name\]
-   **Submission date:** \[Date\]
