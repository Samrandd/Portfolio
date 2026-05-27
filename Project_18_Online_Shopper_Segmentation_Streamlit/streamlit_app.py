from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Online Shopper Segmentation Explorer",
    page_icon="🛒",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
DOC_DIR = BASE_DIR / "docs"


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def show_image(filename: str, caption: str) -> None:
    path = FIG_DIR / filename
    if path.exists():
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing figure: {filename}")


st.title("🛒 Online Shopper Segmentation Explorer")
st.caption("K-means vs Gaussian Mixture Models for online shopper behavior analysis")

st.markdown(
    """
This app summarizes an unsupervised learning project using the **Online Shoppers Purchasing Intention** dataset.
The goal is to identify meaningful shopper behavior segments and compare **K-means clustering** with
**Gaussian Mixture Models (GMM)**.

The `Revenue` variable was excluded from clustering and used only afterward to evaluate whether the resulting
segments showed meaningful differences in purchase behavior.
"""
)

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
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
            st.write(f"- `{file_name}`")
