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
        unique_values = pd.unique(X.values.ravel())
        self.mapping = {val: i for i, val in enumerate(unique_values)}
        self.reverse_mapping = {i: val for val, i in self.mapping.items()}
        return self
    
    def transform(self, X):
        # Handle new, unseen categories
        for val in pd.unique(X.values.ravel()):
            if val not in self.mapping:
                max_idx = max(self.mapping.values()) + 1
                self.mapping[val] = max_idx
                self.reverse_mapping[max_idx] = val
        return X.applymap(lambda x: self.mapping.get(x, max(self.mapping.values()) + 1)).values
    
    def inverse_transform(self, X):
        return pd.DataFrame(X).applymap(lambda x: self.reverse_mapping.get(x, 'Unknown')).values

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

output_df.columns = [output_mapping.get(col, col) for col in output_mapping]

# Find and rename key columns
indicator_col = find_closest_column('indicator', input_df.columns)
comm_type_col = find_closest_column('commission_type', input_df.columns)

# Split columns into different types
numerical_cols = [col for col in input_df.columns if col not in [indicator_col, comm_type_col] and pd.api.types.is_numeric_dtype(input_df[col])]
categorical_cols = [col for col in input_df.columns if col not in numerical_cols]

# Initialize transformers
indicator_encoder = IndicatorEncoder()
comm_type_encoder = OneHotEncoder(handle_unknown='ignore')

# Apply transformers
input_df[indicator_col] = indicator_encoder.fit_transform(input_df[indicator_col])
comm_types = comm_type_encoder.fit_transform(input_df[comm_type_col].values.reshape(-1, 1)).toarray()

# Apply encoder to output as well
output_df[output_mapping['indicator_long']] = indicator_encoder.transform(output_df[output_mapping['indicator_long']])

# Split the data into features (X) and targets (y)
X_num = input_df[numerical_cols]
X_cat = np.column_stack([input_df[indicator_col], comm_types])
X = np.column_stack([X_num, X_cat])

y_numerical = output_df[[output_mapping['principal_amt'], output_mapping['full_commission']]]
y_categorical = output_df[[output_mapping['indicator_long']]]

# Split into training and testing sets
X_train, X_test, y_train_num, y_test_num, y_train_cat, y_test_cat = train_test_split(
    X, y_numerical, y_categorical, test_size=0.2, random_state=42
)

# Initialize models and create pipelines
scaler = StandardScaler()
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

multi_regressor = MultiOutputRegressor(regressor)
multi_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create and train pipelines
reg_pipeline = Pipeline(steps=[('scaler', scaler), ('regressor', multi_regressor)])
reg_pipeline.fit(X_train[:, :len(numerical_cols)], y_train_num)

clf_pipeline = Pipeline(steps=[('classifier', multi_classifier)])
clf_pipeline.fit(X_train[:, len(numerical_cols):], y_train_cat)

# Make predictions on test set
y_pred_num = reg_pipeline.predict(X_test[:, :len(numerical_cols)])
y_pred_cat = clf_pipeline.predict(X_test[:, len(numerical_cols):])

# Convert predicted categories back to original labels
y_pred_cat_original = indicator_encoder.inverse_transform(y_pred_cat)

# Create DataFrames for predicted values and combine them
y_pred_num_df = pd.DataFrame(y_pred_num, columns=y_numerical.columns, index=y_test_num.index)
y_pred_cat_df = pd.DataFrame(y_pred_cat_original, columns=y_categorical.columns, index=y_test_cat.index)
y_pred_df = pd.concat([y_pred_num_df, y_pred_cat_df], axis=1)

# Function to make predictions for new entries
def predict_new_entries(new_data):
    if isinstance(new_data, dict):
        new_data = pd.DataFrame.from_dict(new_data, orient='index')
    
    # Map column names
    new_data = map_columns(new_data, input_mapping)
    new_data.fillna(0, inplace=True)
    
    # Transform data
    new_data[indicator_col] = indicator_encoder.transform(new_data[indicator_col])
    new_comm_types = comm_type_encoder.transform(new_data[comm_type_col].values.reshape(-1, 1)).toarray()
    
    # Prepare the data
    new_data_num = new_data[numerical_cols].values
    new_data_cat = np.column_stack([new_data[indicator_col], new_comm_types])
    
    # Make predictions
    new_pred_num = reg_pipeline.predict(new_data_num)
    new_pred_cat = clf_pipeline.predict(new_data_cat)
    
    # Convert predicted categories back to original labels
    new_pred_cat_original = indicator_encoder.inverse_transform(new_pred_cat)
    
    # Create DataFrames for predicted values
    new_pred_num_df = pd.DataFrame(new_pred_num, columns=y_numerical.columns, index=new_data.index)
    new_pred_cat_df = pd.DataFrame(new_pred_cat_original, columns=y_categorical.columns, index=new_data.index)
    
    # Combine predictions and rename columns
    new_pred_df = pd.concat([new_pred_num_df, new_pred_cat_df], axis=1)
    new_pred_df.columns = [list(output_mapping.keys())[list(output_mapping.values()).index(col)] if col in output_mapping.values() else col for col in new_pred_df.columns]
    
    return new_pred_df

# Example usage for new entries
new_entries = {
    0: {'commission_type': 'R', 'price': 50000, 'quantity': 100, 'commission_amount': 500, 'factor': 0.1, 'indicator_short': 'B'},
    1: {'commission_type': 'F', 'price': 75000, 'quantity': 50, 'commission_amount': 1000, 'factor': 0.2, 'indicator_short': 'S'},
    2: {'commission_type': 'R', 'price': 60000, 'quantity': 75, 'commission_amount': 750, 'factor': 0.15, 'indicator_short': 'SS'},
    3: {'commission_type': 'B', 'price': 80000, 'quantity': 60, 'commission_amount': 800, 'factor': 0.12, 'indicator_short': 'Short Sell'}
}

new_predictions = predict_new_entries(new_entries)
print("\nPredictions for new entries:")
print(new_predictions)
