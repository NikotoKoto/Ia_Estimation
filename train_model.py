import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import pickle

# Charger les données nettoyées
df = pd.read_csv('data/dvf_ile_de_france_2020_cleaned.csv')

# Préparer les données
X = df.drop('valeur_fonciere', axis=1)
y = df['valeur_fonciere']

# Utiliser l'encodage ordinal pour les variables catégorielles
encoder = OrdinalEncoder()
X[['code_postal', 'type_local']] = encoder.fit_transform(X[['code_postal', 'type_local']])

print(X.head())

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Squared Error: {mse}')
print(f'Predictions: {y_pred[:10]}')
print(f'Actual: {y_test[:10].values}')

# Sauvegarder le modèle et l'encodeur
with open('models/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/ordinal_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
