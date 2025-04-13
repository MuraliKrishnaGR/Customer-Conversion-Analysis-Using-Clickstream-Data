# **Customer Conversion Analysis Using Clickstream Data**

## **Project Overview**
This project focuses on analyzing online shopping customer behavior using clickstream data. The goal is to predict key metrics, classify user actions, and provide actionable recommendations to improve customer engagement and conversion rates.

---

## **Key Objectives**
1. **Regression:** Predict item prices within main categories using **Random Forest Regressor**.
2. **Classification:** Determine if a user made a purchase using **Random Forest Classifier** and **Logistic Regression**.
3. **Clustering:** Provide customer recommendations by grouping session data with **K-Means Clustering**.

---

## **Data Preprocessing**
1. **Handling Missing Values:**
   - Imputed or removed missing data to ensure model robustness.
2. **Outlier Removal:**
   - Applied **IQR Clipping** to handle extreme outliers.
3. **Feature Scaling:**
   - Standardized features for clustering and classification using `StandardScaler`.
4. **Feature Engineering:**
   - Derived session-based features, including **session length**, **main categories**, and **continent data**.

---

## **Visualizations**
- **Exploratory Data Analysis (EDA):**
  - Distribution of session lengths and purchase behaviors.
  - Correlation heatmaps to identify relationships between features.
  - Geographic user distribution by continent.
- **Clustering Analysis Visualization:**
  - Visualized clusters with **PCA (2D projections)** for interpretability.
  - Used **Silhouette Scores** to determine the optimal number of clusters.

---

## **Models and Techniques**
1. **Regression (Random Forest Regressor):**
   - Predict item prices for specific categories.
   - Metrics: MAE, R2 Score, RMSE.

2. **Classification (Random Forest Classifier & Logistic Regression):**
   - Classify user behavior (purchase or not).
   - Metrics: Accuracy, Precision, Recall, ROC-AUC.

3. **Clustering (K-Means):**
   - Group customers by session data for recommendations.
   - Metric: Silhouette Score.

---

## **MLflow Integration**
- Tracked experiments, logged metrics, and registered models in **MLflow**.
- Registered and deployed models for inference.

---

### Streamlit Application

The app provides the following functionalities:

- **Clustering**: Recommends customer groups based on session data such as session length, main category preferences, and continent of origin.
- **Regression**: Predicts clothing item prices using features like clothing model number, color, and browsing preferences.
- **Classification**: Predicts purchase likelihood based on user behavior, such as session length, category clicks, and geographic data.

---

## **Tools and Libraries**
- **Data Manipulation & Analysis:** pandas, numpy
- **Machine Learning Models:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Model Tracking & Deployment:** MLflow, Streamlit

© 2025 MuraliKrishnaGR. All rights reserved.


---

## **Conclusion**
This project combines regression, classification, and clustering techniques with interactive tools to provide actionable insights. E-commerce businesses can use these findings to enhance customer experience, optimize product pricing, and increase conversions.
