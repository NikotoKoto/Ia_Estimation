import pandas as pd
import requests
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

# Utiliser le backend Agg pour éviter les problèmes d'affichage graphique
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error

# 1. Téléchargement et nettoyage des données
url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2020/full.csv.gz"
response = requests.get(url)
df = pd.read_csv(BytesIO(response.content), compression='gzip', low_memory=False)

columns_to_use = [
    'surface_reelle_bati', 
    'nombre_pieces_principales', 
    'code_postal', 
    'valeur_fonciere', 
    'type_local', 
    'surface_terrain'
]
df = df[columns_to_use]
df = df.dropna()
df = df[df['valeur_fonciere'] > 0]
df = df[df['valeur_fonciere'] < df['valeur_fonciere'].quantile(0.95)]
df_sampled = df.sample(frac=0.1, random_state=42)
df_sampled.to_csv('data/dvf_ile_de_france_2020_cleaned.csv', index=False)

# Afficher les informations sur les données nettoyées
print("Download and cleaning completed!")
print("Données nettoyées et échantillonnées:")
print(df_sampled.describe())
print(df_sampled.head())

# 2. Préparation des données
X = df_sampled.drop('valeur_fonciere', axis=1)
y = df_sampled['valeur_fonciere']
encoder = OrdinalEncoder()
X[['code_postal', 'type_local']] = encoder.fit_transform(X[['code_postal', 'type_local']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Squared Error (MSE): {mse}')

# 5. Visualisation des résultats

# Histogramme des prix des maisons
plt.figure(figsize=(10, 6))
sns.histplot(df_sampled['valeur_fonciere'], bins=50, kde=True)
plt.title('Distribution des prix des maisons')
plt.xlabel('Prix des maisons (euros)')
plt.ylabel('Fréquence')
plt.savefig('output/histogram_price_distribution.png')
plt.close()

# Matrice de corrélation - Exclure les colonnes non numériques
numeric_df = df_sampled.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 6))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de corrélation des caractéristiques')
plt.savefig('output/correlation_matrix.png')
plt.close()

# Comparaison des prédictions aux valeurs réelles
comparison_df = pd.DataFrame({'Valeurs réelles': y_test, 'Prédictions': y_pred})
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Valeurs réelles', y='Prédictions', data=comparison_df)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Comparaison des prédictions aux valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.savefig('output/comparison_real_vs_pred.png')
plt.close()
