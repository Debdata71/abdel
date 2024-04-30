import pandas as pd
from joblib import load
import numpy as np

# Chargement des données
chemin_fichier_csv = "E:/Formation/MLOPS/projet/simulation/premiere_moitie_2021-22.csv"
donnees = pd.read_csv(chemin_fichier_csv)
print("Données chargées :\n", donnees.head())

# Chargement du mapping des équipes
label_encoder_path = "E:/Formation/MLOPS/projet/Modeles/equivalence_equipes.joblib"
label_encoder = load(label_encoder_path)

# Fonction pour gérer les équipes non reconnues par le mapping
def encode_teams(data, column, label_encoder):
    known_labels = set(label_encoder.keys())
    unknown_label = 'Unknown_Team'
    if unknown_label not in label_encoder:
        label_encoder[unknown_label] = len(label_encoder)
    # Ajout des équipes manquantes au mapping si elles ne sont pas déjà présentes
    missing_teams = set(data[column]) - known_labels
    for team in missing_teams:
        label_encoder[team] = len(label_encoder)
    # Encodage
    return data[column].apply(lambda x: label_encoder[x] if x in known_labels else label_encoder[unknown_label])

donnees['HomeTeam'] = encode_teams(donnees, 'HomeTeam', label_encoder)
donnees['AwayTeam'] = encode_teams(donnees, 'AwayTeam', label_encoder)

# Filtrer les données pour l'équipe à domicile
donnees_home = donnees[donnees['HomeTeam'] == label_encoder['Brentford']]
if donnees_home.empty:
    print("Aucun match trouvé pour l'équipe à domicile : Brentford")
else:
    print("Données pour l'équipe à domicile :\n", donnees_home.head())

# Filtrer les données pour l'équipe à l'extérieur
donnees_away = donnees[donnees['AwayTeam'] == label_encoder['Arsenal']]
if donnees_away.empty:
    print("Aucun match trouvé pour l'équipe à l'extérieur : Arsenal")
else:
    print("Données pour l'équipe à l'extérieur :\n", donnees_away.head())

# Calcul des moyennes des cinq derniers matchs
moyennes_Brentford_home = donnees_home[['FTHG', 'HTHG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']].tail(5).mean()
moyennes_Arsenal_away = donnees_away[['FTAG', 'HTAG', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']].tail(5).mean()

# Fusionner les moyennes en une seule ligne de DataFrame pour la prédiction
moyennes = pd.concat([moyennes_Brentford_home, moyennes_Arsenal_away]).to_frame().T
print("Moyennes pour la prédiction :\n", moyennes)

# Assurer que toutes les colonnes nécessaires sont là et dans le bon ordre
colonnes_modele = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
moyennes = moyennes.reindex(columns=colonnes_modele).fillna(0)

# Charger le modèle
modele_rf = load("E:/Formation/MLOPS/projet/Modeles/modele_rf.joblib")

# Créer une nouvelle ligne de données pour la prédiction
nouvelle_ligne = pd.DataFrame([[label_encoder['Brentford'], label_encoder['Arsenal'], moyennes_Brentford_home['FTHG'], moyennes_Arsenal_away['FTAG'], moyennes_Brentford_home['HTHG'], moyennes_Arsenal_away['HTAG'], moyennes_Brentford_home['HS'], moyennes_Arsenal_away['AS'], moyennes_Brentford_home['HST'], moyennes_Arsenal_away['AST'], moyennes_Brentford_home['HF'], moyennes_Arsenal_away['AF'], moyennes_Brentford_home['HC'], moyennes_Arsenal_away['AC'], moyennes_Brentford_home['HY'], moyennes_Arsenal_away['AY'], moyennes_Brentford_home['HR'], moyennes_Arsenal_away['AR']]], columns=colonnes_modele)

# Prédiction
prediction = modele_rf.predict(nouvelle_ligne)
print("Prédiction pour le match Brentford vs Arsenal :", prediction)
