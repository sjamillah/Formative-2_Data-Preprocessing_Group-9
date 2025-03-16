# Formative-2_Data-Preprocessing_Group-9

# Customer Transactions Data Preprocessing and Augmentation

## Overview
This project focuses on preprocessing and augmenting a customer transactions dataset to improve its quality for machine learning applications. The process involves handling missing values, feature transformations, data augmentation, and applying SMOTE for balancing categorical data.

## Preprocessing Steps
1. **Handling Missing Values**:
   - Removed entries with a high percentage of missing values.
   - Used median imputation for numerical features and mode imputation for categorical features.
   
2. **Synthetic Data Generation**:
   - Applied SMOTE for oversampling underrepresented categories.
   - Created synthetic records based on existing patterns.

3. **Data Transformation**:
   - Converted categorical features to numerical using Label Encoding and One-Hot Encoding.
   - Normalized `purchase_amount` using MinMaxScaler.

4. **Merging Datasets**:
   - Combined customer and transaction data using `customer_id`.
   - Utilized an ID mapping file for transitive relationship handling.

5. **Feature Engineering**:
   - Extracted time-based features from `purchase_date` (e.g., day of the week, month).
   - Derived spending patterns and customer segmentation attributes.

6. **Quality Assurance and Consistency Checks**:
   - Verified data integrity, uniqueness, and consistency.
   - Identified and removed duplicates.

## Results
- **Dataset augmentation added 500 synthetic rows.**
- **SMOTE applied to balance categorical data.**
- **Feature selection reduced dimensionality to improve model performance.**
- **Random Forest model trained and evaluated.**


## Contributors
- **Abubakar Ahmed Umar** 
- **Peris Nyawira Wangui**
- **Jamillah Ssozi**

## Youtube Video Link 
-- https://youtu.be/oaXCrsv0Ric
