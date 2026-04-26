# Project 4: Capstone Project - Bank Deposit Prediction

## Business Case

A Portuguese banking institution needs to improve its marketing campaign effectiveness. The bank wants to predict whether clients will subscribe to term deposits to optimize their direct marketing campaigns. By identifying which clients are most likely to open a term deposit, the bank can focus resources on high-probability prospects, improving ROI and reducing marketing costs.

## Problem Statement

The main focus of this project is to predict whether clients will open term deposits (variable y) using sociodemographic, social and economic context attributes and account attributes. This is a binary classification problem where the goal is to identify clients most likely to respond positively to the campaign.

## Dataset

- **Source:** Portuguese Bank Marketing Campaign
- **Records:** Multiple bank client records
- **Features:** Sociodemographic attributes, economic indicators, campaign information
- **Target:** Binary variable (y) - Term deposit subscription (yes/no)
- **Challenge:** Class imbalance and multicollinearity in features

## Methodology

### Data Preprocessing
- Exploratory Data Analysis (EDA)
- Handling class imbalance
- Feature scaling and normalization
- Addressing multicollinearity
- Feature engineering and selection

### Classification Models Implemented

1. **Logistic Regression with Ridge Regularization**
   - Best performing model based on ROC score
   - Provides probability estimates
   - Handles multicollinearity effectively

2. **Decision Tree Classification**
   - Captures non-linear relationships
   - Feature importance analysis

3. **Support Vector Machine (SVM)**
   - Effective for complex decision boundaries
   - Handles high-dimensional data

4. **Naive Bayes**
   - Probabilistic approach
   - Fast training and prediction

5. **k-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Local pattern detection

### Key Analysis Steps

1. Data exploration and visualization
2. Missing value and outlier treatment
3. Class imbalance handling (SMOTE, undersampling)
4. Multicollinearity check and resolution
5. Feature scaling and encoding
6. Model training with cross-validation
7. Hyperparameter tuning
8. Model comparison and selection

## Results

**Best Model:** Logistic Regression (Ridge Algorithm)
- **Metric:** ROC Score - Superior performance

**Most Important Features Identified:**
- Consumer Price Index (CPI)
- Employment variation rate
- Number of contacts during campaign
- Previous campaign success
- Days since last contact
- Client age and job type

## Key Findings

- Economic indicators (CPI, employment) are strong predictors
- Previous client interactions significantly influence subscription likelihood
- Certain job categories show higher subscription rates
- Duration of contact is a key predictor of success

## Deliverables

### Notebooks
- `Capstone Project-Check multicollinearity.ipynb` - Multicollinearity analysis
- `Capstone Project-Samrand Toufani-Imbalance.ipynb` - Class imbalance handling
- `Capstone Project-Samrand Toufani-Model prediction.ipynb` - Model predictions
- `Capstone Project-Samrand Toufani-Visualisation-Presentation.ipynb` - Comprehensive visualizations

### Presentations
- `Project_04_Capstone_Technical_Presentation.pdf` - Technical analysis for data scientists
- `Project_04_Capstone_Non_Technical_Presentation.pdf` - Business summary for stakeholders

## Business Impact

- **Campaign Efficiency:** Better targeting reduces wasted marketing spend
- **ROI Improvement:** Higher conversion rates from focused campaigns
- **Risk Mitigation:** Predictive model reduces guesswork in client acquisition
- **Scalability:** Automated prediction for large-scale campaigns
- **Actionable Insights:** Clear feature importance guides strategy

## Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Class imbalance handling techniques
- Multicollinearity detection and resolution
- Multiple classification algorithms
- Model comparison and selection
- Feature importance analysis
- Cross-validation and hyperparameter tuning
- Python libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- Data visualization for technical and non-technical audiences
- Business case framing and communication

## Tools & Libraries

- Python 3.x
- Scikit-learn (machine learning models)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)
- Imbalanced-learn (SMOTE for class imbalance)

## Conclusion

This capstone project demonstrates end-to-end machine learning workflow from data exploration to model deployment. The Logistic Regression model with Ridge regularization provides an optimal balance between model complexity and predictive performance, making it ideal for production use in the bank's marketing campaign optimization system.