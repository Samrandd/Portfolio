# Project 6: Indian Liver Patient Dataset - Classification

## Business Case

Liver diseases are a significant health concern affecting millions globally. Early detection and diagnosis can significantly improve treatment outcomes. This project develops machine learning models to predict liver disease based on medical test results, assisting healthcare professionals in diagnosis and patient stratification.

## Problem Statement

Predicting whether a patient has liver disease based on blood test results and medical attributes. The goal is to develop accurate classification models that can assist medical professionals in identifying patients with liver disease from diagnostic test data.

## Dataset

**Source:** Indian Liver Patient Dataset (ILPD)

**Patient Attributes:**
- Age
- Gender
- Total Bilirubin
- Direct Bilirubin
- Total Proteins
- Albumin
- A/G Ratio (Albumin/Globulin)
- SGPT (Serum Glutamic-Pyruvic Transaminase)
- SGOT (Serum Glutamic-Oxaloacetic Transaminase)
- Alkaline Phosphatase (Alkphos)

**Target Variable:** Liver Patient Status (Yes/No)

## Methodology

### Data Preprocessing
- Exploratory Data Analysis (EDA)
- Missing value handling
- Feature scaling and normalization
- Addressing class imbalance if present
- Feature selection

### Classification Algorithms Evaluated

1. **Logistic Regression**
   - Baseline probabilistic classifier
   - Interpretable feature coefficients

2. **Decision Tree**
   - Captures non-linear relationships
   - Feature importance analysis

3. **Random Forest**
   - Ensemble method with multiple trees
   - Reduced overfitting
   - Feature importance ranking

4. **Bagging Classifier**
   - Bootstrap aggregating
   - Reduced variance
   - Improved stability

5. **k-Nearest Neighbors (KNN)**
   - Instance-based approach
   - Local pattern detection

## Analysis Pipeline

1. **Data Exploration**
   - Statistical summaries of medical parameters
   - Distribution analysis
   - Correlation between medical markers and liver disease

2. **Feature Engineering**
   - Deriving new features from existing markers
   - Feature interaction analysis
   - Addressing multicollinearity

3. **Model Training**
   - Cross-validation
   - Hyperparameter tuning
   - Class imbalance handling

4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC-AUC curves
   - Confusion matrices
   - Comparative analysis

5. **Feature Importance**
   - Identifying key diagnostic markers
   - Clinical relevance of features

## Key Findings

- **Most Predictive Markers:** Identified which blood test results best predict liver disease
- **Model Performance:** Comparative accuracy across different algorithms
- **Clinical Insights:** Which medical parameters are most important for diagnosis
- **Feature Relationships:** How medical markers correlate with disease presence

## Decision Tree Analysis

The project specifically evaluates which decision tree-based algorithm provides the best prediction of liver disease presence. Decision trees offer:
- Clear interpretation of decision rules
- Easy visualization of diagnostic thresholds
- Practical application for clinical decision-making

## Deliverables

- `Ensemble_imbalanced_the Indian Liver Patient Dataset_Hard_Voting.ipynb` - Ensemble voting classifier
- `Indian Liver Patient Dataset (ILPD).csv` - Medical dataset
- Model comparison report
- Feature importance analysis

## Skills Demonstrated

- Medical data preprocessing
- Multiple classification algorithms
- Ensemble methods (Bagging, Voting)
- Decision tree analysis
- Feature importance interpretation
- Class imbalance handling
- Model evaluation and comparison
- Python libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- Healthcare domain knowledge

## Clinical Impact

- **Diagnostic Assistance:** Support for medical professionals in patient screening
- **Risk Stratification:** Identify high-risk patients needing specialist consultation
- **Early Detection:** Enable preventive interventions
- **Resource Optimization:** Efficient use of specialist resources
- **Data-Driven Decisions:** Evidence-based diagnostic thresholds

## Tools & Libraries

- Python 3.x
- Scikit-learn (machine learning algorithms)
- Pandas (data manipulation)
- NumPy (numerical operations)
- Matplotlib & Seaborn (visualization)
- Imbalanced-learn (handling class imbalance)

## Important Disclaimer

**This model is designed as a diagnostic support tool and should not be used as a substitute for professional medical diagnosis. All medical decisions should be made by qualified healthcare professionals based on comprehensive clinical evaluation.**

## Conclusion

This project demonstrates machine learning application in healthcare. By comparing multiple classification algorithms including decision trees, bagging classifiers, and random forests, we identify the most effective approach for predicting liver disease from medical test results, providing healthcare professionals with a data-driven diagnostic support tool.