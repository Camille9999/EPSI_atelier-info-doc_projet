# Prétraitement des données - `sig`

- Suppression des colonnes inutiles ('title', 'date_time', 'net', 'magType', 'latitude', 'longitude', 'location', 'continent', 'country', 'alert')
- Normalisation : StandardScaler (moyenne=0, écart-type=1) sur colonnes numériques
- Export : earthquake_data_clean_sig.csv
