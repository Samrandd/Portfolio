from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Online Shopper Segmentation Explorer",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
DOC_DIR = BASE_DIR / "docs"


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def show_image(filename: str, caption: str) -> None:
    path = FIG_DIR / filename
    if path.exists():
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {filename}")


def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(exclude="number").columns.tolist()


st.title("Online Shopper Segmentation Explorer")
st.caption("Interactive customer segmentation dashboard using K-means and Gaussian Mixture Models")

st.markdown(
    """
This app turns my final project into a small analytics product. It summarizes the unsupervised learning results,
compares K-means and Gaussian Mixture Models, and lets users explore the original online shopper dataset through
custom visualizations.
"""
)

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Data Explorer",
        "Model Comparison",
        "K-means Results",
        "GMM Results",
        "Cluster Profiles",
        "PCA Visualizations",
        "Business Takeaways",
        "Project Files",
    ],
)

if section == "Project Overview":
    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", "12,330")
    col2.metric("Original Columns", "18")
    col3.metric("Modeled Features", "75")
    col4.metric("Purchase Rate", "15.47%")

    st.subheader("Problem statement")
    st.write(
        """
Online shopping sessions are heterogeneous. Some visitors browse casually, some compare products,
and others show strong purchase intent. This project compares K-means and GMM to identify meaningful shopper
segments and evaluate whether those segments differ in purchase behavior.
"""
    )

    st.subheader("Dataset and preprocessing")
    st.write(
        """
The dataset contains 12,330 sessions and 18 columns. Revenue was removed before clustering and used only as a
post-clustering evaluation variable. Numeric features were standardized, categorical features were one-hot encoded,
and the final modeled feature matrix contained 75 features.
"""
    )

    show_image("revenue_correlation.png", "Correlation of numeric features with Revenue")


elif section == "Data Explorer":
    st.header("Interactive Raw Data Explorer")

    st.write(
        """
This section lets users explore the original online shopper dataset and build their own visualizations.
Users can filter the data, choose attributes, and compare shopper behavior across different groups.
"""
    )

    raw_path = DATA_DIR / "online_shoppers_intention.csv"

    if not raw_path.exists():
        st.error(
            "The raw dataset is missing. Please add `online_shoppers_intention.csv` to the data folder."
        )
        st.stop()

    raw_df = pd.read_csv(raw_path)

    tab1, tab2, tab3 = st.tabs(
        [
            "Build Your Own Visualization",
            "What We Did",
            "Data Dictionary",
        ]
    )

    with tab1:
        st.subheader("Build Your Own Visualization")

        st.write(
            """
Use the filters below and then select the chart type and attributes you want to compare.
For example, you can compare `PageValues` by `Revenue`, inspect `ProductRelated_Duration`,
or review purchase behavior by `Month`, `VisitorType`, and `Weekend`.
"""
        )

        filtered_df = raw_df.copy()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "Revenue" in filtered_df.columns:
                revenue_filter = st.multiselect(
                    "Revenue",
                    options=sorted(filtered_df["Revenue"].astype(str).unique()),
                    default=sorted(filtered_df["Revenue"].astype(str).unique()),
                )
                filtered_df = filtered_df[
                    filtered_df["Revenue"].astype(str).isin(revenue_filter)
                ]

        with col2:
            if "VisitorType" in filtered_df.columns:
                visitor_filter = st.multiselect(
                    "Visitor Type",
                    options=sorted(filtered_df["VisitorType"].astype(str).unique()),
                    default=sorted(filtered_df["VisitorType"].astype(str).unique()),
                )
                filtered_df = filtered_df[
                    filtered_df["VisitorType"].astype(str).isin(visitor_filter)
                ]

        with col3:
            if "Month" in filtered_df.columns:
                month_filter = st.multiselect(
                    "Month",
                    options=sorted(filtered_df["Month"].astype(str).unique()),
                    default=sorted(filtered_df["Month"].astype(str).unique()),
                )
                filtered_df = filtered_df[
                    filtered_df["Month"].astype(str).isin(month_filter)
                ]

        with col4:
            if "Weekend" in filtered_df.columns:
                weekend_filter = st.multiselect(
                    "Weekend",
                    options=sorted(filtered_df["Weekend"].astype(str).unique()),
                    default=sorted(filtered_df["Weekend"].astype(str).unique()),
                )
                filtered_df = filtered_df[
                    filtered_df["Weekend"].astype(str).isin(weekend_filter)
                ]

        st.markdown("### Filtered Data Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{filtered_df.shape[0]:,}")
        c2.metric("Columns", f"{filtered_df.shape[1]:,}")

        if "Revenue" in filtered_df.columns:
            c3.metric("Purchase Rate", f"{filtered_df['Revenue'].mean() * 100:.2f}%")
        else:
            c3.metric("Purchase Rate", "N/A")

        c4.metric("Missing Values", f"{filtered_df.isna().sum().sum():,}")

        with st.expander("Preview filtered data"):
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        numeric_cols = get_numeric_cols(filtered_df)
        categorical_cols = get_categorical_cols(filtered_df)
        all_cols = filtered_df.columns.tolist()

        st.markdown("### Custom Visualization Builder")

        chart_type = st.selectbox(
            "Choose chart type",
            [
                "Histogram",
                "Bar chart",
                "Scatter plot",
                "Box plot",
                "Correlation heatmap",
            ],
        )

        if chart_type == "Histogram":
            if not numeric_cols:
                st.warning("No numeric columns are available for a histogram.")
            else:
                x_col = st.selectbox("Choose numeric attribute", numeric_cols)
                color_col = st.selectbox(
                    "Optional color/grouping",
                    ["None"] + categorical_cols,
                )

                fig = px.histogram(
                    filtered_df,
                    x=x_col,
                    color=None if color_col == "None" else color_col,
                    nbins=40,
                    title=f"Distribution of {x_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar chart":
            x_col = st.selectbox("Choose category or attribute", all_cols)

            value_option = st.radio(
                "Bar chart type",
                ["Count records", "Average of numeric attribute"],
                horizontal=True,
            )

            if value_option == "Count records":
                chart_df = filtered_df[x_col].astype(str).value_counts().reset_index()
                chart_df.columns = [x_col, "Count"]

                fig = px.bar(
                    chart_df,
                    x=x_col,
                    y="Count",
                    title=f"Count by {x_col}",
                )
            else:
                if not numeric_cols:
                    st.warning("No numeric columns are available to average.")
                    st.stop()

                y_col = st.selectbox("Choose numeric attribute to average", numeric_cols)

                chart_df = (
                    filtered_df.groupby(x_col, as_index=False)[y_col]
                    .mean()
                    .sort_values(y_col, ascending=False)
                )

                fig = px.bar(
                    chart_df,
                    x=x_col,
                    y=y_col,
                    title=f"Average {y_col} by {x_col}",
                )

            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter plot":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for a scatter plot.")
            else:
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols, index=1)
                color_col = st.selectbox("Color by", ["None"] + all_cols)

                fig = px.scatter(
                    filtered_df,
                    x=x_col,
                    y=y_col,
                    color=None if color_col == "None" else color_col,
                    opacity=0.6,
                    title=f"{y_col} vs {x_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box plot":
            if not numeric_cols:
                st.warning("Need at least one numeric column for a box plot.")
            else:
                y_col = st.selectbox("Numeric attribute", numeric_cols)
                x_col = st.selectbox("Group by", ["None"] + categorical_cols)

                fig = px.box(
                    filtered_df,
                    x=None if x_col == "None" else x_col,
                    y=y_col,
                    title=f"Box plot of {y_col}",
                )
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Correlation heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for a correlation heatmap.")
            else:
                selected_numeric = st.multiselect(
                    "Choose numeric attributes",
                    numeric_cols,
                    default=numeric_cols[:8],
                )

                if len(selected_numeric) >= 2:
                    corr = filtered_df[selected_numeric].corr()

                    fig = go.Figure(
                        data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.index,
                            colorscale="RdBu",
                            zmin=-1,
                            zmax=1,
                        )
                    )
                    fig.update_layout(title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please choose at least two numeric attributes.")

        st.download_button(
            label="Download filtered data as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_online_shopper_data.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("What We Did in the Project")

        st.markdown(
            """
### 1. Started with exploratory data analysis
I reviewed the dataset structure, missing values, summary statistics, feature distributions, and purchase rate.
The purchase rate was about **15.47%**, meaning most sessions did not end in a purchase.

### 2. Removed Revenue before clustering
`Revenue` was not used to train the clustering models.  
It was only used afterward to evaluate whether the discovered clusters had different purchase behavior.

### 3. Preprocessed the features
Numeric variables were standardized, and categorical variables such as `Month`, `VisitorType`,
`Browser`, and `TrafficType` were one-hot encoded. The final modeled feature matrix had **75 features**.

### 4. Compared K-means and GMM
K-means was used as a baseline hard clustering method.  
Gaussian Mixture Models were used because they allow probabilistic cluster membership and more flexible cluster shapes.

### 5. Evaluated the models
K-means was evaluated using silhouette score and inertia.  
GMM was evaluated using BIC, AIC, silhouette score, revenue rate by cluster, and interpretability.

### Main finding
K-means with `k=3` produced the clearest business interpretation:
high-intent shoppers, moderate browsers, and shallow visitors.  
The full-covariance GMM with 6 components had the best statistical fit by BIC.
"""
        )

    with tab3:
        st.subheader("Data Dictionary")

        dictionary_df = pd.DataFrame(
            {
                "Attribute": [
                    "Administrative",
                    "Administrative_Duration",
                    "Informational",
                    "Informational_Duration",
                    "ProductRelated",
                    "ProductRelated_Duration",
                    "BounceRates",
                    "ExitRates",
                    "PageValues",
                    "SpecialDay",
                    "Month",
                    "OperatingSystems",
                    "Browser",
                    "Region",
                    "TrafficType",
                    "VisitorType",
                    "Weekend",
                    "Revenue",
                ],
                "Explanation": [
                    "Number of administrative/account-management pages visited",
                    "Total time spent on administrative pages",
                    "Number of informational pages visited",
                    "Total time spent on informational pages",
                    "Number of product-related pages visited",
                    "Total time spent on product-related pages",
                    "Average bounce rate of pages visited",
                    "Average exit rate of pages visited",
                    "Average value of pages visited before transaction",
                    "Closeness of the session date to a special shopping day",
                    "Month of the user session",
                    "Operating system used by the visitor",
                    "Browser used by the visitor",
                    "Region of the visitor",
                    "Traffic source that brought the visitor to the site",
                    "Type of visitor: new, returning, or other",
                    "Whether the visit occurred on a weekend",
                    "Whether the session ended with a purchase",
                ],
                "Example": [
                    "2",
                    "80.8 seconds",
                    "1",
                    "34.5 seconds",
                    "32",
                    "1194.7 seconds",
                    "0.02",
                    "0.04",
                    "5.89",
                    "0.0",
                    "Nov",
                    "2",
                    "2",
                    "1",
                    "2",
                    "Returning_Visitor",
                    "True / False",
                    "True / False",
                ],
            }
        )

        st.dataframe(dictionary_df, use_container_width=True, hide_index=True)


elif section == "Model Comparison":
    st.header("Model Comparison Summary")
    model_summary = load_csv("model_summary.csv")
    st.dataframe(model_summary, use_container_width=True, hide_index=True)

    st.info(
        "K-means with k=3 was the easiest to explain as high-, medium-, and low-intent shopper segments. "
        "The full-covariance GMM with 6 components achieved the best statistical fit by BIC."
    )


elif section == "K-means Results":
    st.header("K-means Results")
    st.write(
        """
K-means was used as the baseline hard clustering method. Candidate values of `k` from 2 to 8 were evaluated
using silhouette score and inertia.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image("kmeans_silhouette.png", "K-means silhouette scores")
    with col2:
        show_image("kmeans_elbow.png", "K-means elbow plot")

    st.subheader("Revenue by K-means cluster")
    kmeans_revenue = load_csv("kmeans_revenue.csv")
    st.dataframe(kmeans_revenue, use_container_width=True, hide_index=True)

    st.write(
        """
The `k=3` solution separated shoppers into high-intent, moderate browsing, and shallow visitor groups.
The high-intent cluster had the highest purchase rate, while the shallow visitor cluster had almost no purchase activity.
"""
    )


elif section == "GMM Results":
    st.header("Gaussian Mixture Model Results")
    st.write(
        """
GMM models were tested using spherical, tied, diagonal, and full covariance structures. BIC and AIC were calculated,
with BIC used as the primary model selection criterion.
"""
    )

    show_image("gmm_bic.png", "GMM model selection by BIC")

    st.subheader("Revenue by GMM full-6 cluster")
    gmm_revenue = load_csv("gmm_revenue.csv")
    st.dataframe(gmm_revenue, use_container_width=True, hide_index=True)

    st.info(
        "The full-covariance GMM with 6 components achieved the lowest BIC. The tied-covariance GMM was also useful "
        "because it produced stronger silhouette separation and easier business interpretation."
    )


elif section == "Cluster Profiles":
    st.header("Cluster Profiles")
    st.write(
        """
This table focuses on the K-means `k=3` solution because it produced the clearest business interpretation.
"""
    )

    profile = load_csv("kmeans_profile.csv")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown(
        """
**Interpretation:** The high-intent cluster had the strongest product-related browsing activity, highest `PageValues`,
and lowest bounce/exit rates. The shallow visitor cluster showed almost no `PageValues` and much higher exit behavior.
"""
    )


elif section == "PCA Visualizations":
    st.header("PCA Visualizations")
    st.write(
        """
PCA was used only for visualization. The first two principal components explained approximately **36.2%** of total
variance, so these plots provide a useful but incomplete 2D view of the full 75-dimensional feature space.
"""
    )

    plot_choice = st.selectbox("Choose PCA plot", ["K-means k=3", "GMM full-6", "GMM tied-6"])

    if plot_choice == "K-means k=3":
        show_image("pca_kmeans.png", "PCA visualization of K-means k=3")
    elif plot_choice == "GMM full-6":
        show_image("pca_gmm_full.png", "PCA visualization of GMM full covariance, 6 components")
    else:
        show_image("pca_gmm_tied.png", "PCA visualization of GMM tied covariance, 6 components")


elif section == "Business Takeaways":
    st.header("Business Takeaways")

    st.markdown(
        """
### High-intent shoppers
These sessions show higher `PageValues`, longer product-related browsing, and lower bounce/exit rates.
They may be good candidates for checkout nudges, personalized offers, or remarketing.

### Moderate browsers
These users show some engagement but are less likely to purchase than the high-intent group.
They may benefit from product recommendations or limited-time incentives.

### Shallow visitors
These sessions show limited browsing depth, low `PageValues`, and high exit behavior.
For this group, the business may focus on landing page quality, traffic source quality, or site navigation.

### Final takeaway
K-means gave the clearest business segmentation, while GMM gave a more flexible statistical view of the data.
Together, the models suggest that purchase intent is strongly related to browsing depth, `PageValues`, and exit behavior.
"""
    )


elif section == "Project Files":
    st.header("Project Files")
    st.write("The original report, notebook, and HTML code export are included in the `docs/` folder of this repository.")

    for file_name in [
        "Final_Project_Group_094.pdf",
        "Final_Project_Group_094.ipynb",
        "Final_Project_Group_094_code.html",
        "stoufani3_Final_Report_Project_Group094.docx",
    ]:
        path = DOC_DIR / file_name
        if path.exists():
            with open(path, "rb") as file:
                st.download_button(
                    label=f"Download {file_name}",
                    data=file,
                    file_name=file_name,
                )
