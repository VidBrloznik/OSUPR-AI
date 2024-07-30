import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Branje podatkov
train_df = pd.read_csv('C:\OSUPR-AI\Training.csv')
test_df = pd.read_csv('C:\OSUPR-AI\Testing.csv')

# Odstranjevanje nepotrebnih stolpcev
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Preverjanje podatkov
print(train_df.head())
print(test_df.head())
print(train_df.info())
print(test_df.info())
print(train_df.describe())
print(test_df.describe())
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Priprava podatkov
X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definiranje posameznih modelov
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scaled, y_train)
y_pred_tree = decision_tree.predict(X_test_scaled)

random_forest = RandomForestClassifier()
random_forest.fit(X_train_scaled, y_train)
y_pred_forest = random_forest.predict(X_test_scaled)

svc = SVC()
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

# Ensemble Model - Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('log_reg', log_reg),
    ('decision_tree', decision_tree),
    ('random_forest', random_forest),
    ('svc', svc),
    ('mlp', mlp)
], voting='hard')

ensemble_model.fit(X_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(X_test_scaled)

# Evaluacija posameznih modelov
models = {
    'Logična regresija': y_pred_log_reg,
    'Odločitveno drevo': y_pred_tree,
    'Naključni gozd': y_pred_forest,
    'Podporni vektorski klasifikator': y_pred_svc,
    'Nevronska mreža': y_pred_mlp,
    'Ensemble Model': y_pred_ensemble
}

for model_name, y_pred in models.items():
    print(f'Evalvacija za: {model_name}')
    print(f'Natančnost: {accuracy_score(y_test, y_pred)}')
    print(f'Matrika zmede:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Klasifikacijsko poročilo:\n{classification_report(y_test, y_pred)}')
    print('\n')

# Hiperparametrična optimizacija za Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

print(f"Najboljši parametri: {grid_search.best_params_}")

best_random_forest = grid_search.best_estimator_
y_pred_best_rf = best_random_forest.predict(X_test_scaled)

print('Evalvacija za: Best Random Forest')
print(f'Natančnost: {accuracy_score(y_test, y_pred_best_rf)}')
print(f'Matrika zmede:\n{confusion_matrix(y_test, y_pred_best_rf)}')
print(f'Klasifikacijsko poročilo:\n{classification_report(y_test, y_pred_best_rf)}')

# Shranjevanje najboljšega modela
joblib.dump(best_random_forest, 'best_random_forest.pkl')

# Nalaganje modela
loaded_model = joblib.load('best_random_forest.pkl')
