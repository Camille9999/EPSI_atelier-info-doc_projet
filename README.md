# EPSI_atelier-info-doc_projet


## Installation

Pour utiliser et travailler sur le projet il est fortement conseillé de suivre note [procédure installation](/doc/procedure_installation.adoc)

## Structure du repo

* Le dossier [data](/data/) contient les datasets bruts utilisés, des exports des versions prétraitées dites "clean" ainsi que des fichiers markdown explicitant les transformations effectuées.

* Le dossier [doc](/doc/) contient procédures d'installation, data cards et model cards ainsi qu'un rapport html avec de la visualisation de données interactive.

* Le dossier [models](/models/) comprend les deux modèles entrainés et exportés.

* Le dossier [notebooks](/notebooks/) contient touts les notebooks utilisés au long du projet dont celui avec toutes les étapes de data exploration, transformation de données et entrainement.

```
├── data
│   ├── data_preprocessing_alert.md
│   ├── data_preprocessing_sig.md
│   ├── earthquake_1995-2023.csv
│   ├── earthquake_data_clean_alert.csv
│   ├── earthquake_data_clean_sig.csv
│   └── earthquake_data.csv
├── doc
│   ├── data_card_earthquake_data.json
│   ├── earthquake_data_report.html
│   ├── model_card_alert.md
│   ├── model_card_sig.md
│   └── procedure_installation.adoc
├── models
│   ├── model_alert_logreg.joblib
│   └── model_sig_rf.joblib
├── notebooks
│   ├── data_exploration.ipynb
│   └── rapport_html.ipynb
├── README.md
└── requirements.txt

```
