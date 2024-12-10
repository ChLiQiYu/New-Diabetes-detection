# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from joblib import parallel_backend
import os

# Set temporary folder path for joblib (using an English path)
os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/Temp/joblib'

# Load the dataset from CSV file
data = pd.read_csv("./data.csv")

# Check for missing values and fill them with the median value if any are found
if data.isnull().sum().any():
    data.fillna(data.median(), inplace=True)

# Separate the features (X) and the target variable (y)
X = data.drop(columns=['id', 'target'])  # Drop 'id' and 'target' columns
y = data['target']  # Target variable

# Standardize the features by scaling them (mean = 0, variance = 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced k_neighbors to reduce sample size
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ----------------- Model Optimization -----------------
# Model 1: Random Forest Classifier optimization
rf = RandomForestClassifier(random_state=42)

# Hyperparameters for Random Forest model
rf_params = {
    'n_estimators': [100],         # Limit the number of trees in the forest
    'max_depth': [10],             # Limit the depth of each tree
    'min_samples_split': [5],      # Limit the number of samples required to split a node
    'min_samples_leaf': [2],       # Limit the number of samples required at a leaf node
    'max_features': ['sqrt']      # Limit the maximum number of features to consider for each split
}

# Use GridSearchCV to perform hyperparameter tuning with cross-validation
rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=3, scoring='f1_macro', n_jobs=1)  # cv=3 for 3-fold cross-validation

# Use parallel backend for faster processing
with parallel_backend('loky'):
    rf_grid.fit(X_train, y_train)

# Predict and evaluate the performance of Random Forest model
y_pred_rf = rf_grid.predict(X_test)
rf_f1 = f1_score(y_test, y_pred_rf, average='macro')
print("Optimized Random Forest F1-score:", rf_f1)

# Model 2: K-Nearest Neighbors (KNN) optimization
knn = KNeighborsClassifier()

# Hyperparameters for KNN model
knn_params = {
    'n_neighbors': [3],             # Limit the number of neighbors to consider
    'weights': ['uniform'],         # Use uniform weight for neighbors
    'metric': ['euclidean']         # Use Euclidean distance as the distance metric
}

# Use GridSearchCV to perform hyperparameter tuning with cross-validation
knn_grid = GridSearchCV(estimator=knn, param_grid=knn_params, cv=3, scoring='f1_macro', n_jobs=1)  # cv=3 for 3-fold cross-validation
knn_grid.fit(X_train, y_train)

# Predict and evaluate the performance of KNN model
y_pred_knn = knn_grid.predict(X_test)
knn_f1 = f1_score(y_test, y_pred_knn, average='macro')
print("Optimized KNN F1-score:", knn_f1)

# Model 3: Gradient Boosting Decision Trees (GBDT) optimization using LightGBM
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(np.unique(y)),  # Set the number of classes in the target variable
    'learning_rate': 0.1,            # Set learning rate to reduce number of boosting rounds
    'num_leaves': 20,                # Limit the number of leaves per tree
    'max_depth': 8,                  # Limit the depth of each tree
    'feature_fraction': 0.7,         # Limit the fraction of features used per split
    'bagging_fraction': 0.7,         # Limit the fraction of data used per boosting round
    'bagging_freq': 5,               # Reduce the frequency of bagging
    'verbose': -1                    # Suppress output
}

# Create training and validation datasets for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set the maximum number of boosting rounds
num_boost_round = 300  # Limit the number of boosting rounds

# Train the LightGBM model
gbm = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=num_boost_round,  # Limit the number of boosting rounds
    valid_sets=[train_data, valid_data],  # Add validation set to monitor overfitting
    valid_names=['train', 'valid'],  # Set names for the validation sets
)

# Predict and evaluate the performance of LightGBM model
y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred_gbdt = np.argmax(y_test_pred, axis=1)

# Calculate and print the F1-score for LightGBM model
gbdt_f1 = f1_score(y_test, y_pred_gbdt, average='macro')
print("Optimized LightGBM F1-score:", gbdt_f1)
