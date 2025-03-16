# Formative-2_Data-Preprocessing_Group-9

# Customer Transactions Data Preprocessing and Augmentation

## Overview
This project focuses on preprocessing and augmenting a customer transactions dataset to improve its quality for machine learning applications. The process involves handling missing values, feature transformations, data augmentation, and applying SMOTE for balancing categorical data.

## Project Structure
```
├── data_preprocessing.ipynb  # Jupyter Notebook with preprocessing steps
├── customer_transactions.csv  # Original dataset
├── customer_transactions_augmented.csv  # Augmented dataset
├── final_dataset_ready_group1.csv  # Preprocessed final dataset
├── README.md  # Documentation
```

## Steps Involved

### 1️⃣ Data Loading
We begin by loading the dataset and inspecting its structure.
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('customer_transactions.csv')
df.head()
```

### 2️⃣ Handling Missing Values
We check for missing values and apply appropriate imputation techniques.
```python
print(df.isnull().sum())

# Define imputers
imputer_mean = SimpleImputer(strategy='mean')
imputer_median = SimpleImputer(strategy='median')
imputer_mode = SimpleImputer(strategy='most_frequent')

# Apply imputations
df['customer_rating'] = imputer_mean.fit_transform(df[['customer_rating']])
df['purchase_amount'] = imputer_median.fit_transform(df[['purchase_amount']])

print(df.isnull().sum())  # Verify missing values are handled
```

### 3️⃣ Data Augmentation
#### Applying Random Noise
```python
np.random.seed(42)
df['purchase_amount_augmented'] = df['purchase_amount'] * (1 + np.random.uniform(-0.05, 0.05, df.shape[0]))
```

#### Log Transformation
```python
df['log_purchase_amount'] = np.log1p(df['purchase_amount_augmented'])
```

#### Synthetic Transformations
```python
num_new_rows = 500
new_transactions = df.sample(n=num_new_rows, replace=True).copy()
new_transactions['purchase_amount_augmented'] *= np.random.uniform(0.9, 1.1, num_new_rows)
new_transactions['transaction_id'] = np.arange(df['transaction_id'].max() + 1, df['transaction_id'].max() + 1 + num_new_rows)

df_augmented = pd.concat([df, new_transactions], ignore_index=True)
```

#### Balancing Data Using SMOTE
```python
if 'product_category' in df_augmented.columns:
    le_category = LabelEncoder()
    df_augmented['product_category_encoded'] = le_category.fit_transform(df_augmented['product_category'])
    
    X_features = df_augmented[['purchase_amount', 'customer_rating']]
    y_target = df_augmented['product_category_encoded']
    
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_features, y_target)
    
    balanced_df = pd.DataFrame(X_smote, columns=['purchase_amount', 'customer_rating'])
    balanced_df['product_category'] = le_category.inverse_transform(y_smote)
    
    customer_ids = df_augmented['customer_id_legacy'].sample(len(balanced_df), replace=True).reset_index(drop=True)
    balanced_df['customer_id_legacy'] = customer_ids
    balanced_df['transaction_id'] = np.arange(1, len(balanced_df) + 1)
    balanced_df['purchase_amount_augmented'] = balanced_df['purchase_amount'] * (1 + np.random.uniform(-0.05, 0.05, len(balanced_df)))
    balanced_df['log_purchase_amount'] = np.log1p(balanced_df['purchase_amount_augmented'])
    
    df_augmented = balanced_df
    print(f"Dataset after SMOTE balancing: {df_augmented.shape}")
```

### 4️⃣ Feature Selection
```python
X = df_augmented.drop(columns=['product_category'])
y = df_augmented['product_category']

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y_imputed = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).flatten())

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_imputed, y_imputed)
```

### 5️⃣ Data Export
```python
df_augmented.to_csv("customer_transactions_augmented.csv", index=False)
final_merged_df.to_csv('final_dataset_ready_group1.csv', index=False)
```

### 6️⃣ Model Training (Random Forest Regressor)
```python
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_imputed, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
```

## Results
- **Dataset augmentation added 500 synthetic rows.**
- **SMOTE applied to balance categorical data.**
- **Feature selection reduced dimensionality to improve model performance.**
- **Random Forest model trained and evaluated.**

## Usage
1. Ensure all dependencies are installed:  
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
   ```
2. Run the preprocessing script in a Jupyter Notebook or Python environment.
3. Train a machine learning model using the prepared dataset.

## Contributors
- **[Your Name]** - Data Preprocessing, Augmentation, Model Training

## License
This project is licensed under the MIT License.
