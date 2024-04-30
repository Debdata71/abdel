from flask import Flask, request, jsonify, render_template
from joblib import load, dump
import pandas as pd
import numpy as np
import logging
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from logging.handlers import RotatingFileHandler
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)

# Configurer la clé secrète JWT
app.config['JWT_SECRET_KEY'] = 'votre_super_secret'  # Changez ceci pour une vraie clé secrète
jwt = JWTManager(app)

logger = logging.getLogger(__name__)
# Configuration de la journalisation
log_file = "app.log"
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Chemins des fichiers de données
chemin_fichier_csv = "E:/Formation/MLOPS/projet/simulation/premiere_moitie_2021-22.csv"
modele_chemin = "E:/Formation/MLOPS/projet/Modeles/modele_rf.joblib"
chemin_fichier_excel = "E:/Formation/MLOPS/projet/simulation/seconde_moitie_2021-22.xlsx"
label_encoder_path = "E:/Formation/MLOPS/projet/Modeles/equivalence_equipes.joblib"

# Chargement du modèle et du LabelEncoder
print("Chargement du modèle et du LabelEncoder...")
try:
    modele = load(modele_chemin)
    label_encoder = load(label_encoder_path)
    print("Classes du LabelEncoder :", label_encoder.classes_)
    donnees_seconde_moitie = pd.read_excel(chemin_fichier_excel)
    logger.info("Modèle et LabelEncoder chargés avec succès.")
except FileNotFoundError as e:
    logger.error("Erreur de chargement des fichiers : %s", e)
    raise

# Simuler une base de données d'utilisateurs
users = {
    "user1": "password1",
    "user2": "password2"
}

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    user_password = users.get(username, None)

    if user_password and password == user_password:
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token), 200

    return jsonify("Invalid credentials"), 401

@app.route('/')
def home():
    print("Rendu de la page d'accueil.")
    jours = sorted(donnees_seconde_moitie['Journée'].unique())
    return render_template('formulaire.html', jours=jours)

@app.route('/get_matches/<int:jour>')
def get_matches(jour):
    try:
        matchs_du_jour = donnees_seconde_moitie[donnees_seconde_moitie['Journée'] == jour]
        logger.info("Matches pour la journée %s obtenus avec succès.", jour)
    except KeyError as e:
        logger.error("Journée invalide : %s", e)
        return jsonify({'error': 'Journée invalide'}), 400

    liste_matchs = [{'HomeTeam': row['HomeTeam'], 'AwayTeam': row['AwayTeam']} for _, row in matchs_du_jour.iterrows()]
    return jsonify(liste_matchs)

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    home_team = request.form.get('HomeTeam')
    away_team = request.form.get('AwayTeam')

    if not home_team or not away_team:
        logger.error("Nom des équipes manquants.")
        return jsonify({'error': 'Nom des équipes manquants'}), 400

    # Chargement des données
    donnees = pd.read_csv(chemin_fichier_csv)

    # Encodage sécurisé des équipes
    try:
        home_team_encoded = label_encoder.transform([home_team])[0]
        away_team_encoded = label_encoder.transform([away_team])[0]
    except KeyError as e:
        logger.error("Erreur d'encodage : %s", e)
        return jsonify({'error': 'Erreur encodage'}), 500

        # Filtrer les données pour l'équipe à domicile
    donnees_home = donnees[donnees['HomeTeam'] == home_team_encoded]
    # Filtrer les données pour l'équipe à l'extérieur
    donnees_away = donnees[donnees['AwayTeam'] == away_team_encoded]

    # Calcul des moyennes des statistiques pour les cinq derniers matchs pour chaque équipe
    moyennes_home = donnees_home.tail(5).select_dtypes(include=[np.number]).mean().fillna(0)
    moyennes_away = donnees_away.tail(5).select_dtypes(include=[np.number]).mean().fillna(0)

    # Préparation des données pour la prédiction
    features = pd.concat([moyennes_home, moyennes_away], axis=1).reset_index(drop=True)
    required_columns = [
        'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY',
        'AY', 'HR', 'AR'
    ]
    features = features.reindex(columns=required_columns).fillna(0)

    # Assurer que les features sont dans le bon ordre attendu par le modèle
    try:
        prediction = modele.predict(features)
        result = str(prediction[0])
        logger.info(f"Résultat de la prédiction: {result}")
    except NotFittedError as e:
        logger.error("Erreur de prédiction : %s", e)
        return jsonify({'error': 'Modèle non formé'}), 500

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
