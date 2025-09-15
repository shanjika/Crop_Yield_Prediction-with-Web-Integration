import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("yield_df.csv")

# Drop unwanted index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Encode categorical columns
le_area = LabelEncoder()
le_item = LabelEncoder()
df["Area"] = le_area.fit_transform(df["Area"])
df["Item"] = le_item.fit_transform(df["Item"])

# Define features and target
feature_cols = ["Year", "average_rain_fall_mm_per_year",
                "pesticides_tonnes", "avg_temp", "Area", "Item"]
X = df[feature_cols]
y = df["hg/ha_yield"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and encoders using Pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_area, open("le_area.pkl", "wb"))
pickle.dump(le_item, open("le_item.pkl", "wb"))

print("✅ Model trained and saved successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print metrics
print(f"🔍 R² Score: {r2:.4f}")
print(f"📉 Mean Absolute Error: {mae:.2f}")
print(f"📊 Root Mean Squared Error: {rmse:.2f}")
print(f"✅ Model Accuracy (R² Score): {r2:.4f}")
