import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import pandas as pd
import joblib
import matplotlib.pyplot as plt

scaler = joblib.load('model_final/scaler_india.pkl')
imputer = joblib.load('model_final/imputer_india.pkl')
model = keras.models.load_model('model_final/model_india.keras')

data = pd.read_csv('data/data_usa.csv')

X_df = data.drop(columns=['price_usd_normalized', 'calories'])
X_df = imputer.transform(X_df)
X_df = scaler.transform(X_df)
y_df = data['price_usd_normalized']

y = y_df.values
y_pred = model.predict(X_df).flatten()


plt.figure(figsize=(12, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.title('Predictions vs True Values (USA)')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid()

plt.show()