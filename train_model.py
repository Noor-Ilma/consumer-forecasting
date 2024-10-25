import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('consumer_forecasting_data_1000_entries')

# Define a target variable (e.g., predict whether Consumer_Spend exceeds a threshold)
threshold = 600000
data['High_Spend'] = (data['Consumer_Spend'] > threshold).astype(int)  # Target: 1 if spending > threshold, else 0

# Data preprocessing (you can add more based on your dataset)
data['debt_income_ratio'] = data['Retail_Sales'] / data['Consumer_Spend']  # Example feature engineering

# Prepare feature and target variables
X = data[['Consumer_Confidence_Index', 'Retail_Sales', 'Interest_Rate']]
y = data['High_Spend']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'consumer_forecasting_model.pkl')
