import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Modificat.xlsx")
df = pd.read_excel(file_path_excel)

df_without_rasa = df.drop(columns=['Rasa'])

correlation_matrix = df_without_rasa.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Echilibrat fara Rasa")
plt.show()

threshold = 0.7
grouped_attributes = {}

for column in correlation_matrix.columns:
    highly_correlated = correlation_matrix.index[correlation_matrix[column].abs() > threshold].tolist()
    highly_correlated.remove(column)

    if highly_correlated:
        grouped_attributes[column] = highly_correlated

for attribute, similar_attributes in grouped_attributes.items():
    print(f"Attribute '{attribute}' is highly correlated with: {similar_attributes}")


def calculate_mean_of_columns(file_path_excel, column1, column2, output_file_path):
    df = pd.read_excel(file_path_excel)

    df[column1] = df[column1] * 2
    df[column2] = df[column2] * 2

    df['Frecventa-atac'] = df[[column1, column2]].mean(axis=1)

    df.to_excel(output_file_path, index=False)
    print(f"Fișierul Excel a fost modificat și salvat la: {output_file_path}")


current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-ColoaneSterse.xlsx")
output_file_path = os.path.join(base_directory, "Dataset-Pisici-Grupat.xlsx")

#calculate_mean_of_columns(file_path_excel, 'Frecventa-atac-pasari', 'Frecventa-atac-mamifere-mici', output_file_path)