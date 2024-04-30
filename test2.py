import pandas as pd
from joblib import load
import numpy as np

# Chargement des données
chemin_fichier_csv = "E:/Formation/MLOPS/projet/simulation/premiere_moitie_2021-22.csv"
donnees = pd.read_csv(chemin_fichier_csv)
print("Données chargées :\n", donnees.head())

# Chargement du LabelEncoder sauvegardé
label_encoder = load('E:/Formation/MLOPS/projet/Modeles/label_encoder.joblib')

# Fonction pour gérer les équipes non reconnues par le LabelEncoder
def encode_teams(data, column, label_encoder):
    known_labels = set(label_encoder.classes_)
    unknown_label = 'Unknown_Team'
    if unknown_label not in label_encoder.classes_:
        label_encoder.classes_ = np.append(label_encoder.classes_, unknown_label)
    # Ajout des équipes manquantes aux classes si elles ne sont pas déjà présentes
    missing_teams = set(data[column]) - known_labels
    label_encoder.classes_ = np.append(label_encoder.classes_, list(missing_teams))
    # Encodage
    return data[column].apply(lambda x: label_encoder.transform([x])[0] if x in known_labels else label_encoder.transform([unknown_label])[0])

donnees['HomeTeam'] = encode_teams(donnees, 'HomeTeam', label_encoder)
donnees['AwayTeam'] = encode_teams(donnees, 'AwayTeam', label_encoder)

# Filtrer et trier les données pour Brentford à domicile et Arsenal à l'extérieur
Brentford_id = label_encoder.transform(['Brentford'])[0]
Arsenal_id = label_encoder.transform(['Arsenal'])[0]
donnees_Brentford_home = donnees[donnees['HomeTeam'] == Brentford_id]
donnees_Arsenal_away = donnees[donnees['AwayTeam'] == Arsenal_id]

print("Données pour Brentford à domicile :\n", donnees_Brentford_home.head())
print("Données pour Arsenal à l'extérieur :\n", donnees_Arsenal_away.head())

# Calcul des moyennes des cinq derniers matchs
moyennes_Brentford_home = donnees_Brentford_home[['FTHG', 'HTHG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']].tail(5).mean()
moyennes_Arsenal_away = donnees_Arsenal_away[['FTAG', 'HTAG', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']].tail(5).mean()

# Fusionner les moyennes en une seule ligne de DataFrame pour la prédiction
moyennes = pd.concat([moyennes_Brentford_home, moyennes_Arsenal_away]).to_frame().T
print("Moyennes pour la prédiction :\n", moyennes)

# Assurer que toutes les colonnes nécessaires sont là et dans le bon ordre
# colonnes_modele = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
# moyennes = moyennes.reindex(columns=colonnes_modele).fillna(0)

# Charger le modèle
modele_rf = load("E:/Formation/MLOPS/projet/Modeles/modele_rf.joblib")

# Prédiction
# prediction = modele_rf.predict(moyennes)
# print("Prédiction pour le match Brentford vs Arsenal :", prediction)
# Créer une nouvelle ligne de données pour la prédiction
colonnes_modele = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
nouvelle_ligne = pd.DataFrame([[Brentford_id, Arsenal_id, moyennes_Brentford_home['FTHG'], moyennes_Arsenal_away['FTAG'], moyennes_Brentford_home['HTHG'], moyennes_Arsenal_away['HTAG'], moyennes_Brentford_home['HS'], moyennes_Arsenal_away['AS'], moyennes_Brentford_home['HST'], moyennes_Arsenal_away['AST'], moyennes_Brentford_home['HF'], moyennes_Arsenal_away['AF'], moyennes_Brentford_home['HC'], moyennes_Arsenal_away['AC'], moyennes_Brentford_home['HY'], moyennes_Arsenal_away['AY'], moyennes_Brentford_home['HR'], moyennes_Arsenal_away['AR']]], columns=colonnes_modele)

# Prédiction
prediction = modele_rf.predict(nouvelle_ligne)
print("Prédiction pour le match Brentford vs Arsenal :", prediction)
