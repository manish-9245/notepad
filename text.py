import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from datetime import datetime
from fuzzywuzzy import process

def find_closest_column(target, columns, threshold=80):
    closest_match, score = process.extractOne(target, columns)
    return closest_match if score >= threshold else None

# Load the data
input_file = 'input_2023-06-05.csv'  # Replace with your actual file name
output_file = 'output_2023-06-05.csv'  # Replace with your actual file name

input_df = pd.read_csv(input_file, index_col=0)
output_df = pd.read_csv(output_file, index_col=0)

# Handle typos in column names
output_mapping = {
    'principal_amt': find_closest_column('principal_amt', output_df.columns),
    'full_commission': find_closest_column('full_commission', output_df.columns),
    'indicator_long': find_closest_column('indicator_long', output_df.columns)
}

output_df.columns = [output_mapping.get(col, col) for col in output_mapping]

# Map 'SS' to 'Short Sell' in the input DataFrame
indicator_col = find_closest_column('indicator', input_df.columns)
input_df[indicator_col] = input_df[indicator_col].replace('SS', 'Short Sell')

# Extract column names directly from the DataFrames
input_cols = input_df.columns
output_cols = output_df.columns

print("Input columns:", input_cols)
print("Output columns:", output_cols)

# Identify commission type column
comm_type_col = find_closest_column('commission_type', input_df.columns)

# Split columns into different types
numerical_cols = [col for col in input_df.columns if col not in [indicator_col, comm_type_col] and input_df[col].dtype in ['int64', 'float64']]
categorical_cols = [col for col in input_df.columns if col not in numerical_cols]

print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Initialize label encoders and one-hot encoders
label_encoders = {}
onehot_encoder = OneHotEncoder(handle_unknown='ignore')

# Handle indicator column with LabelEncoder
unique_values = pd.unique(pd.concat([input_df[indicator_col], output_df[output_mapping['indicator_long']]]))
label_encoders[indicator_col] = LabelEncoder()
label_encoders[indicator_col].fit(unique_values)
input_df[indicator_col] = label_encoders[indicator_col].transform(input_df[indicator_col])
output_df[output_mapping['indicator_long']] = label_encoders[indicator_col].transform(output_df[output_mapping['indicator_long']])

# Handle commission type with OneHotEncoder
commission_types = input_df[comm_type_col].values.reshape(-1, 1)
onehot_encoder.fit(commission_types)

# Split the data into features (X) and targets (y)
X_num = input_df[numerical_cols]
X_cat = onehot_encoder.transform(input_df[comm_type_col].values.reshape(-1, 1)).toarray()
X = np.hstack([X_num, X_cat])

y_numerical = output_df[[output_mapping['principal_amt'], output_mapping['full_commission']]]
y_categorical = output_df[[output_mapping['indicator_long']]]

# Split into training and testing sets
X_train, X_test, y_train_num, y_test_num, y_train_cat, y_test_cat = train_test_split(
    X, y_numerical, y_categorical, test_size=0.2, random_state=42
)

# Initialize models
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create multi-output models
multi_regressor = MultiOutputRegressor(regressor)
multi_classifier = MultiOutputClassifier(classifier)

# Train models
multi_regressor.fit(X_train, y_train_num)
multi_classifier.fit(X_train, y_train_cat)

# Make predictions on test set
y_pred_num = multi_regressor.predict(X_test)
y_pred_cat = multi_classifier.predict(X_test)

# Convert predicted categories back to original labels
y_pred_cat_original = y_pred_cat.copy()
y_pred_cat_original[:, 0] = label_encoders[indicator_col].inverse_transform(y_pred_cat[:, 0])

# Create DataFrames for predicted values
y_pred_num_df = pd.DataFrame(y_pred_num, columns=y_numerical.columns, index=y_test_num.index)
y_pred_cat_df = pd.DataFrame(y_pred_cat_original, columns=y_categorical.columns, index=y_test_cat.index)

# Combine numerical and categorical predictions
y_pred_df = pd.concat([y_pred_num_df, y_pred_cat_df], axis=1)

# Function to make predictions for new entries
def predict_new_entries(new_data):
    # Convert new_data to DataFrame if it's a dictionary
    if isinstance(new_data, dict):
        new_data = pd.DataFrame.from_dict(new_data, orient='index')
    
    # Replace 'SS' with 'Short Sell' in new data
    new_data[indicator_col] = new_data[indicator_col].replace('SS', 'Short Sell')
    
    # Prepare the data
    new_data_num = new_data[numerical_cols].fillna(0)
    new_data_cat = onehot_encoder.transform(new_data[comm_type_col].values.reshape(-1, 1)).toarray()
    new_data_combined = np.hstack([new_data_num, new_data_cat])
    
    # Make predictions
    new_pred_num = multi_regressor.predict(new_data_combined)
    new_pred_cat = multi_classifier.predict(new_data_combined)
    
    # Convert predicted categories back to original labels
    new_pred_cat_original = new_pred_cat.copy()
    new_pred_cat_original[:, 0] = label_encoders[indicator_col].inverse_transform(new_pred_cat[:, 0])
    
    # Create DataFrames for predicted values
    new_pred_num_df = pd.DataFrame(new_pred_num, columns=y_numerical.columns, index=new_data.index)
    new_pred_cat_df = pd.DataFrame(new_pred_cat_original, columns=y_categorical.columns, index=new_data.index)
    
    # Combine numerical and categorical predictions
    new_pred_df = pd.concat([new_pred_num_df, new_pred_cat_df], axis=1)
    
    # Rename columns back to their original names
    new_pred_df.columns = [list(output_mapping.keys())[list(output_mapping.values()).index(col)] if col in output_mapping.values() else col for col in new_pred_df.columns]
    
    return new_pred_df

# Example usage for new entries
new_entries = {
    0: {'commission_type': 'R', 'price': 50000, 'quantity': 100, 'commission_amount': 500, 'factor': 0.1, 'indicator_short': 'B'},
    1: {'commission_type': 'F', 'price': 75000, 'quantity': 50, 'commission_amount': 1000, 'factor': 0.2, 'indicator_short': 'S'},
    2: {'commission_type': 'R', 'price': 60000, 'quantity': 75, 'commission_amount': 750, 'factor': 0.15, 'indicator_short': 'SS'},
    3: {'commission_type': 'B', 'price': 80000, 'quantity': 60, 'commission_amount': 800, 'factor': 0.12, 'indicator_short': 'B'}
}

new_predictions = predict_new_entries(new_entries)
print("\nPredictions for new entries:")
print(new_predictions)
