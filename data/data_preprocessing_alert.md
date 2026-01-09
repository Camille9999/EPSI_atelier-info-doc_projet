# Prétraitement des données - `alert`

- Suppression des colonnes inutiles ('title', 'date_time', 'net', 'magType', 'latitude', 'longitude', 'location', 'continent', 'country')
- Normalisation : StandardScaler (moyenne=0, écart-type=1) sur colonnes numériques
- Lignes initiales : 782
- Lignes après suppression des valeurs manquantes sur `alert` : 415
- Export : earthquake_data_clean_alert.csv
