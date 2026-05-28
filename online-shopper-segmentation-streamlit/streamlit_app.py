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

INK = "#111827"
SEGMENT_COLORS = ["#0f766e", "#2563eb", "#f59e0b", "#7c3aed", "#dc2626", "#475569"]

st.markdown(
    """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e5e7eb;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.82rem;
    }
    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 700;
    }
    .app-kicker {
        color: #0f766e;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .app-title {
        color: #111827;
        font-size: 2.55rem;
        font-weight: 760;
        line-height: 1.08;
        margin: 0;
    }
    .app-subtitle {
        color: #475569;
        font-size: 1.05rem;
        line-height: 1.55;
        max-width: 820px;
        margin: 0.75rem 0 1.35rem 0;
    }
    .section-kicker {
        color: #0f766e;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .section-title {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 720;
        line-height: 1.2;
        margin: 0 0 0.35rem 0;
    }
    .section-copy {
        color: #64748b;
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0 0 1rem 0;
    }
    .insight-panel {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f766e;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin: 0.35rem 0 1rem 0;
    }
    .insight-panel strong {
        color: #111827;
    }
    .insight-panel p {
        color: #475569;
        margin: 0.25rem 0 0 0;
        line-height: 1.5;
    }
    div[data-testid="stTabs"] button p {
        font-weight: 650;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def show_image(filename: str, caption: str) -> None:
    path = FIG_DIR / filename
    if path.exists():
        st.image(Image.open(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing figure: {filename}")


def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(exclude="number").columns.tolist()


def page_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="section-kicker">{kicker}</div>
<div class="section-title">{title}</div>
<p class="section-copy">{copy}</p>
""",
        unsafe_allow_html=True,
    )


def insight_panel(title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="insight-panel">
    <strong>{title}</strong>
    <p>{copy}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def style_chart(fig, title=None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_white",
        colorway=SEGMENT_COLORS,
        margin=dict(l=24, r=24, t=58, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color=INK, family="Arial, sans-serif"),
        title=dict(font=dict(size=18, color=INK), x=0.02),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    return fig


st.markdown(
    """
<div class="app-kicker">Customer analytics portfolio</div>
<h1 class="app-title">Online Shopper Segmentation Explorer</h1>
<p class="app-subtitle">
An interactive dashboard for understanding purchase intent, shopper engagement, and segment-level revenue behavior
using K-means and Gaussian Mixture Models.
</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0.4rem 0 1.4rem 0;" />
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
<h2 style="margin-bottom: 0.25rem;">Shopper Segments</h2>
<p style="margin-top: 0; color: #cbd5e1;">Portfolio dashboard</p>
""",
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "Story flow",
    [
        "1. Problem & Data",
        "2. Executive Summary",
        "3. Explore Shopper Behavior",
        "4. Compare Models",
        "5. K-means Segments",
        "6. GMM Deep Dive",
        "7. Segment Profiles",
        "8. PCA Check",
        "9. Business Actions",
        "10. Project Files",
    ],
)

if section == "1. Problem & Data":
    page_intro(
        "Start here",
        "The project starts with a business problem: most online visits do not become purchases.",
        "The goal is to understand which browsing behaviors separate high-intent shoppers from ordinary browsers and shallow visitors, then turn those patterns into usable customer segments.",
    )

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Dataset", "UCI")
    metric_b.metric("Sessions", "12,330")
    metric_c.metric("Fields", "18")
    metric_d.metric("Target Field", "Revenue")

    st.markdown("")
    source_col, objective_col = st.columns([1, 1])

    with source_col:
        st.markdown("#### Where the data comes from")
        st.write(
            "The data is the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository."
        )
        st.markdown(
            "[Open the official UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)"
        )
        st.write(
            "The Streamlit app uses a local copy of the CSV in `online-shopper-segmentation-streamlit/data/online_shoppers_intention.csv`."
        )
        st.markdown(
            """
The dataset describes website sessions using page counts, visit durations, bounce and exit behavior,
traffic source, visitor type, month, weekend status, and whether the session ended in purchase.
"""
        )

    with objective_col:
        st.markdown("#### What we are trying to solve")
        st.write(
            "Instead of only predicting purchase/no purchase, this project asks a more strategic question: **what kinds of shoppers are visiting the site?**"
        )
        st.markdown(
            """
- Identify shopper groups based on behavior before using `Revenue`.
- Compare whether those groups differ in purchase rate.
- Decide whether K-means or GMM gives the better segmentation lens.
- Translate the segments into practical marketing and site-experience actions.
"""
        )

    st.markdown("#### Questions this dashboard answers")
    question_col_1, question_col_2, question_col_3 = st.columns(3)
    with question_col_1:
        insight_panel(
            "Who looks ready to buy?",
            "High-intent shoppers show stronger product browsing, higher PageValues, and lower exit behavior.",
        )
    with question_col_2:
        insight_panel(
            "Which model helps most?",
            "K-means is easier to explain, while GMM can capture overlapping shopper behavior and softer cluster boundaries.",
        )
    with question_col_3:
        insight_panel(
            "What should the business do?",
            "Use segment profiles to decide where to target offers, where to nurture browsers, and where to improve traffic or landing-page quality.",
        )

    st.markdown("#### How to follow the story")
    story_col_1, story_col_2, story_col_3, story_col_4 = st.columns(4)
    with story_col_1:
        insight_panel(
            "1. Frame the problem",
            "Start with the purchase-intent problem, the dataset, and what the project is trying to solve.",
        )
    with story_col_2:
        insight_panel(
            "2. Inspect behavior",
            "Use the data explorer to see how browsing behavior, visitor type, and Revenue relate to one another.",
        )
    with story_col_3:
        insight_panel(
            "3. Compare models",
            "Review why K-means is easier to communicate and why GMM can capture richer cluster structure.",
        )
    with story_col_4:
        insight_panel(
            "4. Decide actions",
            "Use segment profiles and business actions to translate the analysis into practical recommendations.",
        )

    with st.expander("Why clustering instead of only prediction?"):
        st.write(
            """
A prediction model can estimate whether a session will purchase, but it may not explain the types of shoppers behind
that prediction. Clustering groups similar sessions first, then uses `Revenue` afterward to evaluate whether the
groups are commercially meaningful. This makes the result easier to connect to marketing, product, and website decisions.
"""
        )


elif section == "2. Executive Summary":
    page_intro(
        "Executive overview",
        "Purchase intent is concentrated in a small, high-engagement segment.",
        "The dashboard highlights where shoppers differ most: browsing depth, page value, exit behavior, and conversion rate.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", "12,330")
    col2.metric("Purchase Rate", "15.47%")
    col3.metric("Modeled Features", "75")
    col4.metric("Recommended Model", "K-means k=3")

    st.markdown("")
    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        insight_panel(
            "Business signal",
            "High-intent sessions account for a smaller share of traffic but show the strongest purchase rate and deepest product engagement.",
        )
        st.markdown(
            """
#### Segmentation decision
K-means with three clusters provides the clearest business readout:

- **High-intent shoppers:** strong page value, deep product browsing, lower exit behavior.
- **Moderate browsers:** ordinary shopping behavior with meaningful conversion potential.
- **Shallow visitors:** limited engagement and almost no purchase activity.
"""
        )

    with right_col:
        kmeans_revenue_overview = load_csv("kmeans_revenue.csv")
        kmeans_revenue_overview["Revenue Rate Value"] = (
            kmeans_revenue_overview["Revenue Rate"].str.rstrip("%").astype(float)
        )
        fig = px.bar(
            kmeans_revenue_overview,
            x="Interpretation",
            y="Revenue Rate Value",
            color="Interpretation",
            text="Revenue Rate",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, yaxis_title="Revenue rate", xaxis_title="")
        st.plotly_chart(
            style_chart(fig, "Purchase rate by K-means segment"),
            width="stretch",
        )

    st.markdown("#### Revenue correlation")
    show_image("revenue_correlation.png", "Numeric feature correlation with Revenue")

    with st.expander("Modeling scope"):
        st.write(
            """
Revenue was removed before clustering and used only after modeling to evaluate whether the discovered groups
had different purchase behavior. Numeric features were standardized, categorical fields were one-hot encoded,
and the final modeling table contained 75 features.
"""
        )


elif section == "3. Explore Shopper Behavior":
    page_intro(
        "Data explorer",
        "Investigate shopper behavior across traffic, visitor, month, and purchase outcomes.",
        "Filter raw sessions and build quick visual comparisons for conversion patterns, engagement depth, and page value.",
    )

    raw_path = DATA_DIR / "online_shoppers_intention.csv"

    if not raw_path.exists():
        st.error(
            "The raw dataset is missing. Please add `online_shoppers_intention.csv` to the data folder."
        )
        st.stop()

    raw_df = pd.read_csv(raw_path)

    data_metric_1, data_metric_2, data_metric_3, data_metric_4 = st.columns(4)
    data_metric_1.metric("Raw Sessions", f"{raw_df.shape[0]:,}")
    data_metric_2.metric("Columns", f"{raw_df.shape[1]:,}")
    data_metric_3.metric("Purchase Rate", f"{raw_df['Revenue'].mean() * 100:.2f}%")
    data_metric_4.metric("Visitor Types", f"{raw_df['VisitorType'].nunique():,}")

    tab1, tab2, tab3 = st.tabs(
        [
            "Visualization Builder",
            "Method",
            "Data Dictionary",
        ]
    )

    with tab1:
        st.subheader("Visualization Builder")

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
            st.dataframe(filtered_df, width="stretch", hide_index=True)

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
                st.plotly_chart(style_chart(fig), width="stretch")

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

            st.plotly_chart(style_chart(fig), width="stretch")

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
                st.plotly_chart(style_chart(fig), width="stretch")

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
                st.plotly_chart(style_chart(fig), width="stretch")

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
                    st.plotly_chart(style_chart(fig), width="stretch")
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
The full-covariance GMM with 6 components had the best statistical fit by BIC, which suggests it may capture more subtle overlap in shopper behavior than K-means.
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

        st.dataframe(dictionary_df, width="stretch", hide_index=True)


elif section == "4. Compare Models":
    page_intro(
        "Model comparison",
        "K-means gives the clearest business story; GMM gives a more flexible statistical view.",
        "The strongest recommendation is not that one model replaces the other. K-means is best for simple segment labels, while GMM is useful when shopper behavior overlaps across groups.",
    )
    model_summary = load_csv("model_summary.csv")
    st.dataframe(model_summary, width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        insight_panel(
            "Recommended operating model",
            "K-means k=3 is the best choice for stakeholder communication because the segments map cleanly to intent levels.",
        )
    with col_b:
        insight_panel(
            "Where GMM is stronger",
            "The full-covariance GMM with 6 components achieved the strongest BIC score. It can capture softer boundaries, unequal cluster shapes, and mixed shopper behavior better than hard K-means labels.",
        )

    insight_panel(
        "How to position the models",
        "Use K-means k=3 as the executive segmentation because it is easy to explain. Use GMM as the deeper analytical lens when you want probability-based membership, more granular clusters, or evidence that the shopper groups are not perfectly separated.",
    )


elif section == "5. K-means Segments":
    page_intro(
        "K-means results",
        "Three segments provide a practical view of shopper intent.",
        "The K-means solution balances separation, interpretability, and business actionability.",
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image("kmeans_silhouette.png", "K-means silhouette scores")
    with col2:
        show_image("kmeans_elbow.png", "K-means elbow plot")

    st.subheader("Revenue by K-means cluster")
    kmeans_revenue = load_csv("kmeans_revenue.csv")
    kmeans_revenue_chart = kmeans_revenue.copy()
    kmeans_revenue_chart["Revenue Rate Value"] = (
        kmeans_revenue_chart["Revenue Rate"].str.rstrip("%").astype(float)
    )
    fig = px.bar(
        kmeans_revenue_chart,
        x="Interpretation",
        y="Revenue Rate Value",
        color="Interpretation",
        text="Revenue Rate",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Revenue rate", xaxis_title="")
    st.plotly_chart(style_chart(fig, "Revenue rate by cluster"), width="stretch")
    st.dataframe(kmeans_revenue, width="stretch", hide_index=True)

    insight_panel(
        "Interpretation",
        "The high-intent cluster converted at 28.26%, while shallow visitors converted at only 0.57%. This makes the segmentation useful for targeting and site experience decisions.",
    )


elif section == "6. GMM Deep Dive":
    page_intro(
        "GMM results",
        "Gaussian Mixture Models can be better when shopper behavior is blended rather than cleanly separated.",
        "Instead of forcing every session into a hard group, GMM estimates the probability that a session belongs to each cluster and allows clusters to have different shapes.",
    )

    show_image("gmm_bic.png", "GMM model selection by BIC")

    st.subheader("Revenue by GMM full-6 cluster")
    gmm_revenue = load_csv("gmm_revenue.csv")
    gmm_revenue_chart = gmm_revenue.copy()
    gmm_revenue_chart["Revenue Rate Value"] = (
        gmm_revenue_chart["Revenue Rate"].str.rstrip("%").astype(float)
    )
    fig = px.bar(
        gmm_revenue_chart,
        x="Cluster",
        y="Revenue Rate Value",
        color="Cluster",
        text="Revenue Rate",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="Revenue rate", xaxis_title="Cluster")
    st.plotly_chart(style_chart(fig, "GMM cluster revenue rates"), width="stretch")
    st.dataframe(gmm_revenue, width="stretch", hide_index=True)

    gmm_col_1, gmm_col_2 = st.columns(2)
    with gmm_col_1:
        insight_panel(
            "Why GMM can be better",
            "Online shoppers do not always fall into neat groups. A session may look partly like a browser and partly like a high-intent shopper. GMM handles this uncertainty by assigning probabilities instead of only one fixed label.",
        )
    with gmm_col_2:
        insight_panel(
            "Tradeoff",
            "The six-cluster GMM fits the data better statistically, but it is harder to explain to non-technical stakeholders. That makes it a strong analytical model and a useful benchmark, while K-means remains cleaner for presentation.",
        )


elif section == "7. Segment Profiles":
    page_intro(
        "Cluster profiles",
        "The K-means segments differ most on page value, product browsing depth, bounce rate, and exit rate.",
        "These profile metrics translate the model output into observable shopper behavior.",
    )

    profile = load_csv("kmeans_profile.csv")
    st.dataframe(profile, width="stretch", hide_index=True)

    profile_chart = profile[["Interpretation", "PageValues", "ProductRelated", "ExitRates"]].copy()
    profile_chart = profile_chart.melt(
        id_vars="Interpretation",
        value_vars=["PageValues", "ProductRelated", "ExitRates"],
        var_name="Metric",
        value_name="Value",
    )
    fig = px.bar(profile_chart, x="Interpretation", y="Value", color="Metric", barmode="group")
    fig.update_layout(xaxis_title="", yaxis_title="Profile value")
    st.plotly_chart(style_chart(fig, "Segment profile comparison"), width="stretch")

    insight_panel(
        "Profile readout",
        "High-intent shoppers show the strongest product browsing and page value. Shallow visitors show high exit behavior and near-zero page value.",
    )


elif section == "8. PCA Check":
    page_intro(
        "PCA visualizations",
        "Two-dimensional PCA views summarize how the clusters separate in the modeled feature space.",
        "The plots are useful for visual inspection, while the segment profiles and revenue rates provide the business interpretation.",
    )

    plot_choice = st.selectbox("Choose PCA plot", ["K-means k=3", "GMM full-6", "GMM tied-6"])

    if plot_choice == "K-means k=3":
        show_image("pca_kmeans.png", "PCA visualization of K-means k=3")
    elif plot_choice == "GMM full-6":
        show_image("pca_gmm_full.png", "PCA visualization of GMM full covariance, 6 components")
    else:
        show_image("pca_gmm_tied.png", "PCA visualization of GMM tied covariance, 6 components")


elif section == "9. Business Actions":
    page_intro(
        "Business actions",
        "Each segment suggests a different growth lever.",
        "The recommendation is to prioritize high-intent conversion, nurture moderate browsers, and diagnose shallow-visitor traffic quality.",
    )

    action_col_1, action_col_2, action_col_3 = st.columns(3)
    with action_col_1:
        insight_panel(
            "High-intent shoppers",
            "Use checkout nudges, personalized offers, and remarketing because these sessions already show strong engagement and page value.",
        )
    with action_col_2:
        insight_panel(
            "Moderate browsers",
            "Use product recommendations, comparison support, and time-sensitive incentives to move engaged visitors toward purchase.",
        )
    with action_col_3:
        insight_panel(
            "Shallow visitors",
            "Review landing page quality, traffic source relevance, and navigation because this group exits quickly and rarely purchases.",
        )

    st.markdown("#### Final recommendation")
    st.write(
        "Use K-means k=3 as the primary business segmentation for clear stakeholder communication. Use the full-covariance GMM as a deeper analytical layer when you want more granular clusters, probability-based membership, or a better statistical fit. Purchase intent is most strongly associated with browsing depth, PageValues, and exit behavior."
    )


elif section == "10. Project Files":
    page_intro(
        "Project files",
        "Download the supporting report, notebook, and code export.",
        "These files document the analysis workflow behind the dashboard.",
    )

    for file_name in [
        "online_shopper_segmentation_report.pdf",
        "online_shopper_segmentation_notebook.ipynb",
        "online_shopper_segmentation_code_export.html",
        "online_shopper_segmentation_final_report.docx",
    ]:
        path = DOC_DIR / file_name
        if path.exists():
            with open(path, "rb") as file:
                st.download_button(
                    label=f"Download {file_name}",
                    data=file,
                    file_name=file_name,
                )
