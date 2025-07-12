# AI for Percision Agriculture (Data science & AI PROJECT).py
# crop_yield_prediction.py
# crop_production_india_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("Crop Production data.csv")

# Rename columns for consistency
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Display basic info
print("Initial shape:", df.shape)
print(df.head())

# Drop missing values and remove rows with zero production
df.dropna(inplace=True)
df = df[(df['Production'] > 0) & (df['Area'] > 0)]

# Label Encoding for categorical columns
le = LabelEncoder()
df['State_Code'] = le.fit_transform(df['State_Name'])
df['District_Code'] = le.fit_transform(df['District_Name'])
df['Crop_Code'] = le.fit_transform(df['Crop'])
df['Season_Code'] = le.fit_transform(df['Season'])

# Final feature set
X = df[['State_Code', 'District_Code', 'Crop_Year', 'Season_Code', 'Crop_Code', 'Area']]
y = df['Production']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Performance:")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Top 10 crops by total production
top_crops = df.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_crops.values, y=top_crops.index, palette="viridis")
plt.title("Top 10 Crops by Total Production in India")
plt.xlabel("Total Production (Tonnes)")
plt.show()

# Top 10 states by production
top_states = df.groupby("State_Name")["Production"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_states.values, y=top_states.index, palette="coolwarm")
plt.title("Top 10 States by Total Crop Production")
plt.xlabel("Total Production (Tonnes)")
plt.show()

# Trend of production over years
plt.figure(figsize=(12,5))
sns.lineplot(data=df.groupby("Crop_Year")["Production"].sum().reset_index(), x="Crop_Year", y="Production")
plt.title("Crop Production in India Over the Years")
plt.xlabel("Year")
plt.ylabel("Total Production (Tonnes)")
plt.grid(True)
plt.show()
