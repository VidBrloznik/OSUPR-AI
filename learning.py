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
sns.countplot(y=train_df['prognosis'])
plt.show()

X_train = train_df.drop('prognosis', axis=1)
y_train = train_df['prognosis']
X_test = test_df.drop('prognosis', axis=1)
y_test = test_df['prognosis']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)