# Import necessary libraries
from imblearn.over_sampling import SMOTE  # For handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
import pandas as pd  # For data manipulation and reading CSV files
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset into train and test sets
from sklearn.preprocessing import StandardScaler  # For feature scaling (standardization)
from sklearn.ensemble import RandomForestClassifier  # For Random Forest classification model
from sklearn.neighbors import KNeighborsClassifier  # For K-Nearest Neighbors classification model
from sklearn.metrics import f1_score  # For calculating F1 score to evaluate models
import lightgbm as lgb  # For LightGBM classification model (Gradient Boosting Decision Trees)

# Load dataset from a CSV file
data = pd.read_csv("./data.csv")

# Check for missing values and handle them by filling with the median of each column
data.fillna(data.median(), inplace=True)

# Separate features (X) and target variable (y)
X = data.drop(columns=['id', 'target'])  # Drop 'id' and 'target' columns from the features
y = data['target']  # Set 'target' column as the target variable

# Standardize features (scale to have mean=0 and variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42, k_neighbors=3)  # Use SMOTE with k_neighbors set to 3
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)  # Resample the dataset to balance the classes

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Model 1: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the RandomForestClassifier with 100 trees
rf.fit(X_train, y_train)  # Train the model using training data
y_pred_rf = rf.predict(X_test)  # Predict the target for the test data
rf_f1 = f1_score(y_test, y_pred_rf, average='macro')  # Calculate F1-score (macro-average)
print(f"Random Forest F1-score: {rf_f1:.4f}")  # Output the F1-score for Random Forest

# Model 2: K-Nearest Neighbors (KNN) Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN with 5 neighbors
knn.fit(X_train, y_train)  # Train the model
y_pred_knn = knn.predict(X_test)  # Predict the target for the test data
knn_f1 = f1_score(y_test, y_pred_knn, average='macro')  # Calculate F1-score (macro-average)
print(f"KNN F1-score: {knn_f1:.4f}")  # Output the F1-score for KNN

# Model 3: Gradient Boosting Decision Trees using LightGBM
trainX = lgb.Dataset(X_train, y_train)  # Prepare the training data for LightGBM

# Set parameters for LightGBM (Gradient Boosting Decision Trees)
params = {
    'boosting_type': 'gbdt',  # Use the Gradient Boosting Decision Tree algorithm
    'objective': 'multiclass',  # Specify that this is a multiclass classification problem
    'metric': 'multi_logloss',  # Use multi-class log loss as the evaluation metric
    'max_depth': 5,  # Limit the maximum depth of trees
    'num_leaves': 3,  # Limit the maximum number of leaves in each tree
    'learning_rate': 0.01,  # Set the learning rate
    'feature_fraction': 0.7,  # Use 70% of features for each iteration
    'bagging_fraction': 0.7,  # Use 70% of data samples for each iteration
    'bagging_freq': 15,  # Perform bagging every 15 iterations
    'verbose': -1,  # Suppress LightGBM's output
    'num_class': 3  # Number of classes in the target variable
}

# Train the LightGBM model
gbm = lgb.train(params, trainX, 1000, feature_name=list(X.columns))

# Predict the target for the test data using the trained model
y_test_pred = gbm.predict(X_test)
y_pred_gbdt = np.argmax(y_test_pred, axis=1)  # Convert probability predictions to class labels

# Calculate F1-score (macro-average) for LightGBM model
gbdt_f1 = f1_score(y_test, y_pred_gbdt, average='macro')
print(f"LightGBM F1-score: {gbdt_f1:.4f}")  # Output the F1-score for LightGBM
