import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the data
input_file = 'input_{}.csv'.format(datetime.today().strftime('%Y-%m-%d'))
output_file = 'output_{}.csv'.format(datetime.today().strftime('%Y-%m-%d'))

inp_df = pd.read_csv(input_file)
out_df = pd.read_csv(output_file)

# Preprocess the data
label_enc = LabelEncoder()
inp_df['commission_type'] = label_enc.fit_transform(inp_df['commission_type'])
inp_df['indicator_short'] = label_enc.fit_transform(inp_df['indicator_short'])

# Standardize the numerical features
scaler = StandardScaler()
inp_df[['price', 'quantity', 'commission_amount', 'factor']] = scaler.fit_transform(inp_df[['price', 'quantity', 'commission_amount', 'factor']])

# Prepare training data
X = inp_df.values
y_principal_amt = out_df['principal_amt'].values
y_full_commission = out_df['full_commission'].values
y_indicator_long = label_enc.fit_transform(out_df['indicator_long'])

# Split the data
X_train, X_test, y_train_principal_amt, y_test_principal_amt = train_test_split(X, y_principal_amt, test_size=0.2, random_state=42)
X_train, X_test, y_train_full_commission, y_test_full_commission = train_test_split(X, y_full_commission, test_size=0.2, random_state=42)
X_train, X_test, y_train_indicator_long, y_test_indicator_long = train_test_split(X, y_indicator_long, test_size=0.2, random_state=42)

# Train the models
model_principal_amt = RandomForestRegressor(random_state=42)
model_full_commission = RandomForestRegressor(random_state=42)
model_indicator_long = RandomForestClassifier(random_state=42)

model_principal_amt.fit(X_train, y_train_principal_amt)
model_full_commission.fit(X_train, y_train_full_commission)
model_indicator_long.fit(X_train, y_train_indicator_long)

# Predict and evaluate
y_pred_principal_amt = model_principal_amt.predict(X_test)
y_pred_full_commission = model_full_commission.predict(X_test)
y_pred_indicator_long = model_indicator_long.predict(X_test)

mse_principal_amt = mean_squared_error(y_test_principal_amt, y_pred_principal_amt)
mse_full_commission = mean_squared_error(y_test_full_commission, y_pred_full_commission)
accuracy_indicator_long = accuracy_score(y_test_indicator_long, y_pred_indicator_long)

print("MSE Principal Amt:", mse_principal_amt)
print("MSE Full Commission:", mse_full_commission)
print("Accuracy Indicator Long:", accuracy_indicator_long)