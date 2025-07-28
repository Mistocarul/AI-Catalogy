import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-ColoaneSterse.xlsx")

data = pd.read_excel(file_path_excel)

X = data.drop(columns=['Rasa'])
y = data['Rasa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['Rasa'] = y_resampled
resampled_data = resampled_data.drop_duplicates()

print(resampled_data['Rasa'].value_counts())

if 'Nr_rand' in resampled_data.columns:
    resampled_data = resampled_data.drop(columns=['Nr_rand'])

output_file_path = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
resampled_data.to_excel(output_file_path, index=False)

print(f"Datele echilibrate au fost salvate în {output_file_path}")