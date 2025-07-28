import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os

# 1. Citirea fișierului Excel
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Modificat.xlsx")

data = pd.read_excel(file_path_excel)

X = data.drop(columns=['Rasa'])
y = data['Rasa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Utilizarea SelectKBest cu testul ANOVA F pentru a selecta cele mai bune caracteristici
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X_train, y_train)
scores = selector.scores_

feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
print("Scorurile caracteristicilor folosind testul ANOVA F:")
print(feature_scores.sort_values(by='Score', ascending=False))

#Utilizarea Recursive Feature Elimination (RFE) cu RandomForestClassifier
model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train, y_train)

# Crearea unui DataFrame pentru afișarea rezultatelor RFE
rfe_results = pd.DataFrame({'Feature': X.columns, 'Ranking': rfe.ranking_})
print("Clasamentul caracteristicilor folosind RFE:")
print(rfe_results.sort_values(by='Ranking'))

# Identificarea și eliminarea caracteristicilor nefolositoare
unnecessary_features = rfe_results[rfe_results['Ranking'] > 10]['Feature']
#X_train_reduced = X_train.drop(columns=unnecessary_features)
#X_test_reduced = X_test.drop(columns=unnecessary_features)

print("Caracteristicile nefolositoare eliminate:")
print(unnecessary_features)
