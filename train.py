from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import json

# Reading the Dataset
df = pd.read_csv("data.csv")

# Selecting Data
df = df[["age","study_hours","class_attendance","sleep_hours","internet_access","exam_score"]]

# Renaming the Columns
df.rename(columns={"study_hours":"hours","class_attendance":"attendance","exam_score":"score"},inplace=True)

# Duplicates
print("-"*30)
print("No. of Duplicates : ",df.duplicated().sum())
df = df.drop_duplicates()
print("-"*30)
print("No. of Duplicates (After Removing it): ",df.duplicated().sum())
print("-"*30)

# Label Encoding
label = LabelEncoder()
df["internet_access"] = label.fit_transform(df["internet_access"])
# no -> 0 and yes -> 1

# Features
X = df[['age', 'hours', 'attendance', 'sleep_hours', 'internet_access']]
Y = df['score']

# Split the Data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# Model Initialization
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=2,
    random_state=42
)

# Model Training
model.fit(X_train, Y_train)

# =========================
# Model Testing and Evaluation
# =========================

Y_pred = model.predict(X_test)

# Testing Metrics
r2 = r2_score(y_true=Y_test, y_pred=Y_pred)
mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
rmae = np.sqrt(mae)
mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
rmse = np.sqrt(mse)

print("-"*30)
print("\nAccuracy Score For Testing Data")
print("-"*30)
print(f"R² Score : {r2}")
print(f"MAE      : {mae}")
print(f"RMAE      : {rmae}")
print(f"MSE      : {mse}")
print(f"RMSE     : {rmse}")
print("-"*30)


# =========================
# Training Data Evaluation
# =========================

y_train_pred = model.predict(X_train)

r2_train = r2_score(y_true=Y_train, y_pred=y_train_pred)
mae_train = mean_absolute_error(y_true=Y_train, y_pred=y_train_pred)
rmae_train = np.sqrt(mae_train)
mse_train = mean_squared_error(y_true=Y_train, y_pred=y_train_pred)
rmse_train = np.sqrt(mse_train)

print("-"*30)
print("\nAccuracy Score For Training Data")
print("-"*30)
print(f"R² Score : {r2_train}")
print(f"MAE      : {mae_train}")
print(f"RMAE      : {rmae_train}")
print(f"MSE      : {mse_train}")
print(f"RMSE     : {rmse_train}")
print("-"*30)


# =========================
# Store ONLY Testing Accuracy
# =========================

accuracy = {
    "r2_score": float(r2),
    "mae": float(mae),
    "rmae":float(rmae),
    "mse": float(mse),
    "rmse": float(rmse)
}

with open("accuracy.json", "w") as f:
    json.dump(accuracy, f, indent=4)

# Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("-"*30)
print("Feature Importance :")
print(importance)
print("-"*30)

# Saving the Model

with open("model.pkl","wb") as file:
    pkl.dump(model,file)
    print("Model is Saved.")
