# Project 8: Fraud Detection using Unsupervised Learning

## Business Case

Fraud detection is a critical challenge for financial institutions. Banks and payment processors need to identify potentially fraudulent transactions to protect customers and minimize losses. Traditional supervised learning approaches require labeled fraud cases, which can be expensive and time-consuming to obtain.

This project demonstrates the use of unsupervised learning techniques to detect anomalous transaction patterns that may indicate fraudulent activity.

## Objective

Use unsupervised machine learning algorithms (K-means clustering and DBSCAN) to identify fraudulent transactions in the BankSim dataset without relying on pre-labeled fraud cases.

## Dataset

**Source:** BankSim Dataset (banksim1.csv)
- Contains simulated bank transactions
- Multiple transaction features including amount, merchant, location, and time
- Real-world transaction characteristics

## Methodology

### Techniques Applied

1. **K-Means Clustering**
   - Partitioning-based clustering algorithm
   - Identifies clusters of similar transactions
   - Anomalies are identified as outliers from main clusters

2. **DBSCAN (Density-Based Spatial Clustering)**
   - Density-based clustering approach
   - Better at identifying irregular-shaped clusters
   - Points in low-density regions marked as outliers/anomalies
   - More robust for detecting fraud patterns

## Key Analysis Steps

1. Data preprocessing and feature engineering
2. Exploratory Data Analysis (EDA)
3. Feature scaling and normalization
4. Model training and parameter tuning
5. Anomaly detection and fraud identification
6. Results interpretation and visualization

## Skills Demonstrated

- Python data manipulation (Pandas, NumPy)
- Unsupervised learning algorithms
- Data preprocessing and scaling
- Clustering and anomaly detection
- Data visualization (Matplotlib, Seaborn)
- Domain knowledge in fraud detection

## Results

The analysis reveals transaction patterns that deviate from normal behavior. The combination of K-Means and DBSCAN provides complementary perspectives on potential fraud cases, with DBSCAN offering superior performance in identifying density-based anomalies.

## Files

- `Fraud_detection_unsupervised models.ipynb` - Main analysis notebook
- `banksim1.csv` - Transaction dataset

## Tools & Libraries

- Python 3.x
- Scikit-learn (clustering algorithms)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)

## Notes

This project demonstrates practical applications of unsupervised learning in detecting financial fraud without requiring labeled training data, making it applicable to real-world scenarios where fraud labels are expensive or unavailable.