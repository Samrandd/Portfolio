# Project 5: Clustering on Vehicle Dataset

## Business Case

An automobile manufacturer has developed prototypes for a new vehicle. Before introducing the new model into its range, the manufacturer wants to determine which existing vehicles on the market are most similar to their new prototype. By understanding the competitive landscape and vehicle clustering patterns, the manufacturer can better position their product and make informed decisions about market entry, pricing, and features.

## Problem Statement

Use clustering methods to find the most distinctive clusters of vehicles in the market. This analysis will summarize existing vehicles and help the manufacturer make data-driven decisions about new model positioning, competitive analysis, and market strategy.

## Dataset

**Source:** 1985 Ward's Automotive Yearbook
- **Records:** Vehicles from 1985
- **Features:** Vehicle specifications and characteristics
- **Attributes Included:**
  - Engine specifications (type, size, horsepower)
  - Fuel system and consumption
  - Dimensions (length, width, height, weight)
  - Performance metrics (acceleration, braking)
  - Safety features
  - Insurance risk ratings
  - Normalized losses

## Methodology

### Clustering Technique: Agglomerative Hierarchical Clustering

**Why Hierarchical Clustering?**
- Creates dendrograms showing vehicle relationships
- Allows multiple levels of granularity
- No need to pre-specify number of clusters
- Provides interpretable hierarchical structure
- Better for exploratory analysis

### Linkage Methods
- Single linkage
- Complete linkage
- Average linkage
- Ward linkage (minimizes within-cluster variance)

## Analysis Pipeline

1. **Data Exploration**
   - Vehicle specifications overview
   - Feature distributions and correlations
   - Missing value handling
   - Outlier detection

2. **Feature Preprocessing**
   - Feature scaling and normalization
   - Standardization for fair distance comparison
   - Dimensionality considerations

3. **Distance Metric Selection**
   - Euclidean distance for hierarchical clustering
   - Testing different linkage methods

4. **Hierarchical Clustering**
   - Building dendrogram
   - Identifying optimal cluster cutoff
   - Cluster interpretation

5. **Cluster Characterization**
   - Analyzing cluster centroids
   - Comparing vehicle types within clusters
   - Identifying distinguishing features per cluster

6. **Business Insights**
   - Vehicle categories and segments
   - Competitive positioning
   - Market opportunities

## Key Findings

- **Distinctive Vehicle Clusters:** Identified separate market segments
- **Competitive Landscape:** Clear categorization of existing vehicles
- **Feature Relationships:** Vehicle specifications that drive clustering
- **Market Gaps:** Potential positioning for new prototype
- **Size and Performance Segmentation:** Vehicles cluster by similar specifications and capabilities

## Deliverables

- `lab_hierarchical-Ass.ipynb` - Complete hierarchical clustering analysis
- `Automobile_data.csv` - Vehicle dataset
- Dendrogram visualizations
- Cluster descriptions and business recommendations

## Skills Demonstrated

- Hierarchical clustering algorithms
- Dendrogram interpretation
- Feature scaling and normalization
- Distance metric selection
- Clustering evaluation
- Business case framing
- Data visualization
- Python libraries: Scikit-learn, Scipy, Pandas, NumPy, Matplotlib
- Domain knowledge in automotive industry

## Applications & Impact

- **Product Positioning:** Identify best market segment for new vehicle
- **Competitive Analysis:** Understand direct competitors in each cluster
- **Pricing Strategy:** Set competitive prices based on cluster analysis
- **Feature Selection:** Design features aligned with target cluster
- **Market Segmentation:** Understand distinct vehicle market categories

## Tools & Libraries

- Python 3.x
- Scikit-learn (clustering algorithms)
- Scipy (hierarchical clustering)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)

## References

- This assignment (except dataset) was prepared by Saeed Aghabozorgi, based on IBM Machine Learning with Python course from Coursera
- Dataset source: 1985 Ward's Automotive Yearbook, available on Kaggle

## Conclusion

Hierarchical clustering provides valuable insights into vehicle market segmentation. The dendrogram reveals natural groupings of vehicles based on specifications, helping the manufacturer identify the most suitable market position for their new prototype and understand competitive dynamics within each vehicle cluster.