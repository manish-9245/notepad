# Load the data
input_file = 'input_{}.csv'.format(datetime.today().strftime('%Y-%m-%d'))
output_file = 'output_{}.csv'.format(datetime.today().strftime('%Y-%m-%d'))

inp_df = pd.read_csv(input_file)
out_df = pd.read_csv(output_file)

# Define preprocessing for numeric and categorical features
numeric_features = ['price', 'quantity', 'commission_amount', 'factor']
categorical_features = ['commission_type', 'indicator_short']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the models
regressor = RandomForestRegressor(random_state=42)
classifier = RandomForestClassifier(random_state=42)

# Combine preprocessing and model into a pipeline
pipeline_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(regressor))
])

pipeline_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Prepare the target variables
y_reg = out_df[['principal_amt', 'full_commission']]
y_class = out_df['indicator_long']

# Encode the target variable for classification
label_enc = LabelEncoder()
y_class_encoded = label_enc.fit_transform(y_class)

# Split the data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(inp_df, y_reg, test_size=0.2, random_state=42)
X_train, X_test, y_train_class, y_test_class = train_test_split(inp_df, y_class_encoded, test_size=0.2, random_state=42)

# Train the models
pipeline_regressor.fit(X_train, y_train_reg)
pipeline_classifier.fit(X_train, y_train_class)

# Predict and evaluate
y_pred_reg = pipeline_regressor.predict(X_test)
y_pred_class = pipeline_classifier.predict(X_test)

mse_principal_amt = mean_squared_error(y_test_reg['principal_amt'], y_pred_reg[:, 0])
mse_full_commission = mean_squared_error(y_test_reg['full_commission'], y_pred_reg[:, 1])
accuracy_indicator_long = accuracy_score(y_test_class, y_pred_class)

print("MSE Principal Amt:", mse_principal_amt)
print("MSE Full Commission:", mse_full_commission)
print("Accuracy Indicator Long:", accuracy_indicator_long)

# To make predictions on new data
def predict_new_data(new_data):
    new_data_processed = pipeline_regressor['preprocessor'].transform(new_data)
    new_principal_amt, new_full_commission = pipeline_regressor['regressor'].predict(new_data_processed)
    new_indicator_long = pipeline_classifier['classifier'].predict(new_data_processed)
    new_indicator_long_decoded = label_enc.inverse_transform(new_indicator_long)
    return new_principal_amt, new_full_commission, new_indicator_long_decoded

# Example new data
new_data = pd.DataFrame({
    'commission_type': ['R', 'F'],
    'price': [500000, 750000],
    'quantity': [10000, 20000],
    'commission_amount': [5000, 7000],
    'factor': [0.05, 0.1],
    'indicator_short': ['B', 'S']
})

# Make predictions
new_principal_amt, new_full_commission, new_indicator_long_decoded = predict_new_data(new_data)

# Print predictions
print("New Principal Amt Predictions:", new_principal_amt)
print("New Full Commission Predictions:", new_full_commission)
print("New Indicator Long Predictions:", new_indicator_long_decoded)