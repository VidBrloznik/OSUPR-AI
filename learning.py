import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

train_df = pd.read_csv('C:\OSUPR-AI\Training.csv')
test_df = pd.read_csv('C:\OSUPR-AI\Testing.csv')

train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]


print(train_df.head())
print(test_df.head())

print(train_df.info())
print(test_df.info())

# Povzetek podatkov na kratko
print(train_df.describe())
print(test_df.describe())

print(train_df.isnull().sum())
print(test_df.isnull().sum())

# porazdelitev primerov različnih bolezni
#sns.countplot(y=train_df['prognosis'])
#plt.show()


X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistična regresija
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Odločitveno drevo
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_scaled, y_train)
y_pred_tree = decision_tree.predict(X_test_scaled)

# Naključni gozd
random_forest = RandomForestClassifier()
random_forest.fit(X_train_scaled, y_train)
y_pred_forest = random_forest.predict(X_test_scaled)

# Podporni vektorji
svc = SVC()
svc.fit(X_train_scaled, y_train)
y_pred_svc = svc.predict(X_test_scaled)

# Nevronska mreža
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

models = {
    'Logična regresija': y_pred_log_reg,
    'Odločitveno drevo': y_pred_tree,
    'Naključni gozd': y_pred_forest,
    'Podporni vektorski klasifikator': y_pred_svc,
    'Nevronska mreža': y_pred_mlp
}

for model_name, y_pred in models.items():
    print(f'Evalvacija za: {model_name}')
    print(f'Natančnost: {accuracy_score(y_test, y_pred)}')
    print(f'Matrika zmede:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Klacifikacijsko poročilo:\n{classification_report(y_test, y_pred)}')
    print('\n')
