## Project Overview

This project aims to build and evaluate multiple classification models for predicting a target variable from a given dataset. The models used include Random Forest, K-Nearest Neighbors (KNN), and LightGBM (Gradient Boosting Decision Trees). The project also handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and scales the features using StandardScaler.

## Steps Involved:

1. **Data Loading and Preprocessing**:
    - Load a CSV dataset containing the target and feature variables.
    - Handle missing values by filling them with the median of each feature column.
    - Standardize features to have a mean of 0 and variance of 1.
2. **Handling Class Imbalance**:
    - Apply SMOTE to balance the class distribution in the target variable.
3. **Model Training and Evaluation**:
    - **Random Forest Classifier**: A tree-based ensemble learning method for classification.
    - **K-Nearest Neighbors (KNN)**: A simple, non-parametric classification algorithm.
    - **LightGBM**: A gradient boosting decision tree model optimized for speed and accuracy.
4. **Performance Evaluation**:
    - All models are evaluated using the **F1-score** (macro-average), which is a metric that considers both precision and recall, making it particularly useful for imbalanced datasets.

## Installation Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn (imblearn)
- lightgbm
- matplotlib (for optional visualization)

To install the required libraries, you can use pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn lightgbm matplotlib
```

## Running the Program

1. **Prepare the dataset**: Place your dataset file (`data.csv`) in the same directory as the script.
    
2. **Run the Python script**: Execute the script to load the data, train the models, and print the F1-scores for each model.
    

bash

复制代码

`python classification_model.py`

3. **Expected Output**:
    - The script will output the F1-scores for the Random Forest, KNN, and LightGBM models, which represent their performance on the test set.

Example output:

```yaml
Random Forest F1-score: 0.8321 KNN F1-score: 0.8147 LightGBM F1-score: 0.8412
```

## Conclusion

This project provides a simple pipeline for training and evaluating multiple classification models on an imbalanced dataset. By using SMOTE to balance the dataset and evaluating the models using the F1-score, it ensures reliable performance metrics.