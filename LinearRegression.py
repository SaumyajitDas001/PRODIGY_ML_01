import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = train_data[features].copy()
y = train_data[target]

X = X.fillna(X.mean())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

test_X = test_data[features].copy()
test_X = test_X.fillna(test_X.mean())

test_predictions = model.predict(test_X)

test_data['PredictedPrice'] = test_predictions
test_data[['Id', 'PredictedPrice']].to_csv('predicted_prices.csv', index=False)
print("Predictions saved to 'predicted_prices.csv'")
