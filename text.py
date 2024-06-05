import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from datetime import datetime

# Load the data
input_file = 'input_2023-06-05.csv'  # Replace with your actual file name
output_file = 'output_2023-06-05.csv'  # Replace with your actual file name

input_df = pd.read_csv(input_file, index_col=0)
output_df = pd.read_csv(output_file, index_col=0)

# Extract column names directly from the DataFrames
input_cols = input_df.columns
output_cols = output_df.columns

print("Input columns:", input_cols)
print("Output columns:", output_cols)

# Identify numerical and categorical columns
numerical_cols = input_df.select_dtypes(include=[np.number]).columns
categorical_cols = input_df.select_dtypes(include=[object]).columns

# Determine numerical and categorical output columns
output_numerical_cols = output_df.select_dtypes(include=[np.number]).columns
output_categorical_cols = output_df.select_dtypes(include=[object]).columns

print("Numerical output columns:", output_numerical_cols)
print("Categorical output columns:", output_categorical_cols)

# Initialize label encoders for each categorical column
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    input_df[col] = label_encoders[col].fit_transform(input_df[col])

# Initialize label encoders for output categorical columns
output_label_encoders = {}
for col in output_categorical_cols:
    output_label_encoders[col] = LabelEncoder()
    output_df[col] = output_label_encoders[col].fit_transform(output_df[col])

# Split the data into features (X) and targets (y)
X = input_df
y_numerical = output_df[output_numerical_cols]
y_categorical = output_df[output_categorical_cols]

# Split into training and testing sets
X_train, X_test, y_train_num, y_test_num, y_train_cat, y_test_cat = train_test_split(
    X, y_numerical, y_categorical, test_size=0.2, random_state=42
)

# Create column transformers for numerical and categorical columns
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols)
    ])

# Initialize models
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Create multi-output models
multi_regressor = MultiOutputRegressor(regressor)
multi_classifier = MultiOutputClassifier(classifier)

# Create pipelines
reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', multi_regressor)
])

clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', multi_classifier)
])

# Train models
reg_pipeline.fit(X_train, y_train_num)
clf_pipeline.fit(X_train, y_train_cat)

# Make predictions on test set
y_pred_num = reg_pipeline.predict(X_test)
y_pred_cat = clf_pipeline.predict(X_test)

# Convert predicted categories back to original labels
y_pred_cat_original = y_pred_cat.copy()
for i, col in enumerate(output_categorical_cols):
    y_pred_cat_original[:, i] = output_label_encoders[col].inverse_transform(y_pred_cat[:, i])

# Create DataFrames for predicted values
y_pred_num_df = pd.DataFrame(y_pred_num, columns=output_numerical_cols, index=y_test_num.index)
y_pred_cat_df = pd.DataFrame(y_pred_cat_original, columns=output_categorical_cols, index=y_test_cat.index)

# Combine numerical and categorical predictions
y_pred_df = pd.concat([y_pred_num_df, y_pred_cat_df], axis=1)

# Save predictions to CSV
y_pred_df.to_csv(f'predictions_{datetime.today().strftime("%Y-%m-%d")}.csv')

# Function to make predictions for new entries
def predict_new_entries(new_data):
    # Convert new_data to DataFrame if it's a dictionary
    if isinstance(new_data, dict):
        new_data = pd.DataFrame.from_dict(new_data, orient='index')
    
    # Ensure all columns are present and in the correct order
    missing_cols = set(input_cols) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0  # or any appropriate default value

    new_data = new_data[input_cols]
    
    # Encode categorical columns
    for col in categorical_cols:
        new_data[col] = label_encoders[col].transform(new_data[col].astype(str))
    
    # Make predictions
    new_pred_num = reg_pipeline.predict(new_data)
    new_pred_cat = clf_pipeline.predict(new_data)
    
    # Convert predicted categories back to original labels
    new_pred_cat_original = new_pred_cat.copy()
    for i, col in enumerate(output_categorical_cols):
        new_pred_cat_original[:, i] = output_label_encoders[col].inverse_transform(new_pred_cat[:, i])
    
    # Create DataFrames for predicted values
    new_pred_num_df = pd.DataFrame(new_pred_num, columns=output_numerical_cols, index=new_data.index)
    new_pred_cat_df = pd.DataFrame(new_pred_cat_original, columns=output_categorical_cols, index=new_data.index)
    
    # Combine numerical and categorical predictions
    new_pred_df = pd.concat([new_pred_num_df, new_pred_cat_df], axis=1)
    
    return new_pred_df

# Example usage for new entries
new_entries = {
    0: {'commission_type': 'R', 'price': 50000, 'quantity': 100, 'commission_amount': 500, 'factor': 0.1, 'indicator_short': 'B'},
    1: {'commission_type': 'F', 'price': 75000, 'quantity': 50, 'commission_amount': 1000, 'factor': 0.2, 'indicator_short': 'S'}
}

new_predictions = predict_new_entries(new_entries)
print("\nPredictions for new entries:")
print(new_predictions)
