# python-project6_datacleaning_and_unsupervised_machine_learning

# Uber Trip Data - Unsupervised Learning Project

## Project Description
This project performs comprehensive data preprocessing, clustering, dimensionality reduction, and anomaly detection on Uber trip data using Python and Scikit-learn.

## Dataset
- **File:** `Uber-Jan-Feb-FOIL.csv`
- **Type:** Tabular data loaded using pandas
- **Columns:** dispatching_base_number, date, active_vehicles, trips
- **Records:** 354 rows

## Steps Performed

### 1. Data Loading & Initial Exploration
- Imported pandas, numpy, matplotlib, and multiple Scikit-learn libraries
- Mounted Google Drive to access the dataset
- Loaded CSV dataset using `pd.read_csv()`
- Displayed data overview with `.head()`

### 2. Data Cleaning & Transformation
- **Base Number:** Removed the letter 'B' from dispatching base numbers and converted them to integers
- **Date Conversion:** Used `pd.to_datetime()` on the date column
- **Feature Extraction:** Extracted `year`, `day`, `week` (day of week), and `month` into separate columns
- **Dropped Unnecessary Data:** Removed the original `date` column after extraction

### 3. Feature Preparation
- **X (Features):** Selected all columns as the feature matrix
- **Feature Scaling:** Applied `StandardScaler` to normalize the data so all features are on the same scale

### 4. Finding Optimal Clusters (KMeans)
- **Inertia Plot:** Looped K from 1 to 10 and plotted inertia to find the elbow point
- **Silhouette Score:** Looped K from 2 to 10 and plotted silhouette scores to check cluster separation
- **Conclusion:** Determined K=2 as the best number of clusters

### 5. Model Training - KMeans
- **KMeans (K=2):** Trained the model and assigned cluster labels to each row
- **`groupby('k,labels').mean()`:** Compared average stats per cluster
  - Cluster 0: ~835 active vehicles, ~7,526 trips (low volume)
  - Cluster 1: ~3,718 active vehicles, ~32,800 trips (high volume)

### 6. Model Training - DBSCAN
- **DBSCAN (eps=2):** Ran density-based clustering
- Found 3 clusters and 0 noise points
- **`groupby('DBSCAN').mean()`:** Compared averages per cluster, identifying one massive cluster and two smaller ones

### 7. Dimensionality Reduction - PCA
- **PCA (n_components=3):** Reduced the dataset from 7 columns to 3 principal components
- **`explained_variance_ratio_`:** Checked variance captured by each component (~39.6%, ~18.3%, ~16.6%)
- **Cumulative Plot:** Visualized total variance retained (~74.4%)
- **Created `new_uber_data`:** New DataFrame with the 3 compressed components

### 8. Anomaly Detection - Isolation Forest
- **IsolationForest (contamination=0.5):** Trained to identify anomalies
- **Counted Results:**
  - Normal points: 177
  - Anomalies: 177
- *Note:* Setting contamination to 0.5 split the data exactly in half, which is unusually high for real-world anomaly detection

## Key Findings
- Uber dispatching bases naturally split into two groups: **low-volume** and **high-volume** operations
- High-volume bases have roughly **4x more active vehicles** and **4x more trips** than low-volume ones
- 3 principal components capture about **74% of the total variance** in the dataset

## Libraries Used
```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
