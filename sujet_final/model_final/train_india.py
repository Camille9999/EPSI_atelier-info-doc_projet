import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
# imputer
from sklearn.impute import SimpleImputer

scaler = StandardScaler()
imputer = SimpleImputer(strategy='median')

# 1. Chargement et préparation des données
data = pd.read_csv('data/data_india.csv')

# Imputation des valeurs manquantes


corr = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation des Caractéristiques (India)')

X_df = data.drop(columns=['price_usd_normalized', 'calories'])
y_df = data['price_usd_normalized']

X_df = imputer.fit_transform(X_df)
y = y_df.values

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Architecture du modèle
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.MeanSquaredError(),
    metrics=['mean_absolute_error']
)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=30, 
    batch_size=32
)

# convergence plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Convergence du Modèle (India)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('MAE du Modèle (India)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.grid()

y_pred = model.predict(X_test).flatten()
# convergence plot
plt.figure(figsize=(12, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Predictions vs True Values (India)')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid()

# export
model.save('model_final/model_india.keras')
# export du scaler
joblib.dump(scaler, 'model_final/scaler_india.pkl')  

joblib.dump(imputer, 'model_final/imputer_india.pkl')

plt.show()