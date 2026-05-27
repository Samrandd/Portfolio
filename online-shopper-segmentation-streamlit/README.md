# Online Shopper Segmentation Explorer

An interactive Streamlit portfolio app for an ISyE 6740 final project on **probabilistic customer segmentation** using the Online Shoppers Purchasing Intention dataset.

The project compares **K-means clustering** and **Gaussian Mixture Models (GMM)** to identify behavioral segments among online shoppers. The `Revenue` outcome was excluded during clustering and used afterward to evaluate whether the discovered segments differed in purchase behavior.

## Project highlights

- Built an unsupervised segmentation workflow on 12,330 online shopping sessions.
- Compared K-means with Gaussian Mixture Models using silhouette score, inertia/elbow, BIC, AIC, and interpretability.
- Found that K-means with `k=3` produced the clearest business segmentation.
- Found that the full-covariance GMM with 6 components achieved the lowest BIC.
- Used revenue-by-cluster and cluster profiles to connect model outputs to shopper intent.
- Added PCA visualizations to compare how the cluster structures appear in two dimensions.

## App sections

The Streamlit app includes:

1. Project overview
2. Model comparison
3. K-means results
4. GMM results
5. Cluster profiles
6. PCA visualizations
7. Business takeaways

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── data/
│   ├── model_summary.csv
│   ├── kmeans_revenue.csv
│   ├── gmm_revenue.csv
│   └── kmeans_profile.csv
├── figures/
│   ├── revenue_correlation.png
│   ├── kmeans_silhouette.png
│   ├── kmeans_elbow.png
│   ├── gmm_bic.png
│   ├── pca_kmeans.png
│   ├── pca_gmm_full.png
│   └── pca_gmm_tied.png
└── docs/
    ├── Final_Project_Group_094.ipynb
    ├── Final_Project_Group_094.pdf
    ├── Final_Project_Group_094_code.html
    └── stoufani3_Final_Report_Project_Group094.docx
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Choose this GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Deploy the app.

## Main takeaway

K-means gave the clearest business segmentation, while GMM provided a more flexible statistical view of customer behavior. Across models, higher purchase rates were associated with stronger `PageValues`, deeper product-related browsing, and lower bounce/exit behavior.
