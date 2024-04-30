import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump


# Chargement des données
donnees = pd.read_csv("E:/Formation/MLOPS/projet/english-premier-league/2020-2021.csv")
chemin_donnees = "E:/Formation/MLOPS/projet/english-premier-league/2020-2021.csv"
print(f"Chargement des données depuis : {chemin_donnees}")
donnees = pd.read_csv(chemin_donnees)


# Supprimer les colonnes non pertinentes
colonnes_a_garder = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
donnees = donnees[colonnes_a_garder]
print("Colonnes retenues pour le modèle :", donnees.columns.tolist())

# Encodage des variables catégorielles 'HomeTeam' et 'AwayTeam'
print("Encodage des variables 'HomeTeam' et 'AwayTeam'...")
# Importer le mapping
equivalence_equipes = {
    'Arsenal': 0,
    'Aston Villa': 1,
    'Brentford': 2,
    'Brighton': 3,
    'Burnley': 4,
    'Chelsea': 5,
    'Crystal Palace': 6,
    'Everton': 7,
    'Fulham': 8,
    'Leeds': 9,
    'Leicester': 10,
    'Liverpool': 11,
    'Man City': 12,
    'Man United': 13,
    'Newcastle': 14,
    'Norwich': 15,
    'Southampton': 16,
    'Tottenham': 17,
    'Watford': 18,
    'West Ham': 19,
    'Wolves': 20,
    'Bournemouth': 21,
    'Swansea': 22,
    'Stoke': 23,
    'West Brom': 24,
    'Huddersfield': 25,
    'Sheffield United': 26,
    'Reading': 27,
    'Blackburn': 28,
    'Derby': 29,
    'Cardiff': 30,
    'Middlesbrough': 31,
    'Sunderland': 32,
    'Birmingham': 33,
    'Wigan': 34,
    'Portsmouth': 35,
    'Fulham': 36,
    'Bolton': 37,
    'Blackpool': 38,
    'Hull': 39,
    'QPR': 40,
    'Burnley': 41,
    'Brighton': 42,
    'Huddersfield': 43,
    'Swansea': 44,
    'Norwich': 45,
    'Bournemouth': 46,
    'Watford': 47,
    'Stoke': 48,
    'Reading': 49
}

# Utiliser le mapping pour encoder les noms d'équipes dans votre ensemble de données
donnees['HomeTeam'] = donnees['HomeTeam'].map(equivalence_equipes)
donnees['AwayTeam'] = donnees['AwayTeam'].map(equivalence_equipes)

print("Encodage terminé.")

# Vérifier s'il y a des valeurs manquantes dans les données
if donnees.isnull().values.any():
    print("\nAttention : Il y a des valeurs manquantes dans les données. Traitement en cours...")
    # Remplacer les valeurs manquantes par la moyenne des colonnes
    donnees.fillna(donnees.mean(), inplace=True)
    print("Les valeurs manquantes ont été traitées.")

# Sélection des caractéristiques et de la cible
X = donnees.drop(['FTR'], axis=1)
y = donnees['FTR']

# Répéter la division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Répéter la création et l'entraînement du modèle
modele_rf = RandomForestClassifier(random_state=42)
modele_rf.fit(X_train, y_train)

# Prédiction et évaluation
predictions = modele_rf.predict(X_test)
print("\nPrécision :", accuracy_score(y_test, predictions))

# Enregistrer le modèle
modele_chemin = "E:/Formation/MLOPS/projet/Modeles/modele_rf.joblib"
dump(modele_rf, modele_chemin)
print(f"Modèle enregistré à : {modele_chemin}")


# Enregistrer le tableau d'équivalence des équipes
dump(equivalence_equipes, "E:/Formation/MLOPS/projet/Modeles/equivalence_equipes.joblib")