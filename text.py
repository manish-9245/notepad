import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
from fuzzywuzzy import process

def find_closest_column(target, columns, threshold=80):
    closest_match, score = process.extractOne(target, columns)
    return closest_match if score >= threshold else None

class IndicatorEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = None
        self.reverse_mapping = None
    
    def fit(self, X, y=None):
        unique_values = pd.unique(X)
        self.mapping = {val: i for i, val in enumerate(unique_values)}
        self.reverse_mapping = {i: val for val, i in self.mapping.items()}
        return self
    
    def transform(self, X):
        # Convert to Series if it's a DataFrame to use apply
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Handle new, unseen categories
        for val in pd.unique(X):
            if val not in self.mapping:
                max_idx = max(self.mapping.values()) + 1 if self.mapping else 0
                self.mapping[val] = max_idx
                self.reverse_mapping[max_idx] = val
        
        # Use map instead of applymap for Series
        return X.map(lambda x: self.mapping.get(x, max(self.mapping.values()) + 1 if self.mapping else 0)).values.reshape(-1, 1)
    
    def inverse_transform(self, X):
        # Convert 2D array to Series for easier mapping
        X_series = pd.Series(X.ravel())
        return X_series.map(lambda x: self.reverse_mapping.get(x, 'Unknown')).values.reshape(X.shape)

# Load the data
input_file = 'input_2023-06-05.csv'  # Replace with your actual file name
output_file = 'output_2023-06-05.csv'  # Replace with your actual file name

input_df = pd.read_csv(input_file, index_col=0)
output_df = pd.read_csv(output_file, index_col=0)

# Map column names
output_mapping = {
    'principal_amt': find_closest_column('principal_amt', output_df.columns),
    'full_commission': find_closest_column('full_commission', output_df.columns),
    'indicator_long': find_closest_column('indicator_long', output_df.columns)
}

# Use fuzzy matching to rename columns
for target, source in output_mapping.items():
    if source in output_df.columns:
        output_df.rename(columns={source: target}, inplace=True)

# Find and rename key columns
indicator_col = find_closest_column('indicator', input_df.columns)
comm_type_col = find_closest_column('commission_type', input_df.columns)

# Apply transformers
indicator_encoder = IndicatorEncoder().fit(pd.concat([input_df[indicator_col], output_df['indicator_long']]))
comm_type_encoder = OneHotEncoder(handle_unknown='ignore')

input_df[indicator_col] = indicator_encoder.transform(input_df[indicator_col])
comm_types = comm_type_encoder.fit_transform(input_df[comm_type_col].values.reshape(-1, 1)).toarray()

# Apply encoder to output as well
output_df['indicator_long'] = indicator_encoder.transform(output_df['indicator_long'])

# Split columns into different types
numerical_cols = [col for col in input_df.columns if col not in [indicator_col, comm_type_col] and pd.api.types.is_numeric_dtype(input_df[col])]

# Split the data into features (X) and targets (y)
X_num = input_df[numerical_cols].values
X_ind = input_df[indicator_col].values
X_comm = comm_types
X = np.column_stack([X_num, X_ind, X_comm])

feature_names = list(numerical_cols) + [indicator_col] + [f'comm_type_{i}' for i in range(comm_types.shape[1])]

y_numerical = output_df[['principal_amt', 'full_commission']]
y_categorical = output_df[['indicator_long']]

# Split into training and testing sets
X_train, X_test, y_train_num, y_test_num, y_train_cat, y_test_cat = train_test_split(
    X, y_numerical, y_categorical, test_size=0.2, random_state=42
)

# Initialize models
scaler = StandardScaler()
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create multi-output models
multi_regressor = MultiOutputRegressor(regressor)

# Create and train pipelines
reg_pipeline = Pipeline(steps=[('scaler', scaler), ('regressor', multi_regressor)])
reg_pipeline.fit(X_train, y_train_num)

clf_pipeline = Pipeline(steps=[('classifier', classifier)])
clf_pipeline.fit(X_train, y_train_cat)

# Make predictions on test set
y_pred_num = reg_pipeline.predict(X_test)
y_pred_cat = clf_pipeline.predict(X_test)

# Convert predicted categories back to original labels
y_pred_cat_original = indicator_encoder.inverse_transform(y_pred_cat)

# Create DataFrames for predicted values and combine them
y_pred_num_df = pd.DataFrame(y_pred_num, columns=output_numerical_cols, index=y_test_num.index)
y_pred_cat_df = pd.DataFrame(y_pred_cat_original, columns=output_categorical_cols, index=y_test_cat.index)
y_pred_df = pd.concat([y_pred_num_df, y_pred_cat_df], axis=1)

# Function to make predictions for new entries
def predict_new_entries(new_data):
    if isinstance(new_data, dict):
        new_data = pd.DataFrame.from_dict(new_data, orient='index')
    
    # Rename columns if needed
    for col in input_cols:
        closest_match = find_closest_column(col, new_data.columns)
        if closest_match and closest_match != col:
            new_data.rename(columns={closest_match: col}, inplace=True)
    
    # Fill missing columns with default values
    for col in input_cols:
        if col not in new_data.columns:
            new_data[col] = 0 if col in numerical_cols else 'Unknown'
    
    # Transform data
    new_data_num = new_data[numerical_cols].fillna(0).values
    new_data_ind = indicator_encoder.transform(new_data[indicator_col].fillna('Unknown'))
    new_data_comm = comm_type_encoder.transform(new_data[comm_type_col].fillna('Unknown').values.reshape(-1, 1)).toarray()
    
    # Combine features
    new_data_combined = np.column_stack([new_data_num, new_data_ind, new_data_comm])
    
    # Make predictions
    new_pred_num = reg_pipeline.predict(new_data_combined)
    new_pred_cat = clf_pipeline.predict(new_data_combined)
    
    # Convert predicted categories back to original labels
    new_pred_cat_original = indicator_encoder.inverse_transform(new_pred_cat)
    
    # Create DataFrames for predicted values
    new_pred_num_df = pd.DataFrame(new_pred_num, columns=output_numerical_cols, index=new_data.index)
    new_pred_cat_df = pd.DataFrame(new_pred_cat_original, columns=output_categorical_cols, index=new_data.index)
    
    # Combine predictions
    new_pred_df = pd.concat([new_pred_num_df, new_pred_cat_df], axis=1)
    
    return new_pred_df

# Example usage for new entries
new_entries = {
    0: {'commission type': 'R', 'price': 50000, 'quantity': 100, 'commission amount': 500, 'factor': 0.1, 'indicator short': 'B'},
    1: {'commission type': 'F', 'price': 75000, 'quantity': 50, 'commission amount': 1000, 'factor': 0.2, 'indicator short': 'S'},
    2: {'commission type': 'R', 'price': 60000, 'quantity': 75, 'commission amount': 750, 'factor': 0.15, 'indicator short': 'SS'},
    3: {'commission type': 'B', 'price': 80000, 'quantity': 60, 'commission amount': 800, 'factor': 0.12, 'indicator short': 'Short Sell'}
}

new_predictions = predict_new_entries(new_entries)
print("\nPredictions for new entries:")
print(new_predictions)

# Print feature importance (for numerical predictions)
importances = reg_pipeline.named_steps['regressor'].estimators_[0].feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature importance for principal_amt:")
for f, idx in enumerate(indices):
    print(f"{feature_names[idx]}: {importances[idx]}")
