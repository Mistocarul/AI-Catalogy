import os
import pandas as pd
import numpy as np
from scipy import stats

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
data = pd.read_excel(file_path_excel)

def identify_outliers_zscore(data):
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))  # Numai coloanele numerice
    outliers_z = data[(z_scores > 3).any(axis=1)]
    return outliers_z

outliers_z = identify_outliers_zscore(data)

print("Zgomote detectate folosind scorul Z:")
print(outliers_z)

cleaned_data = data[~data.index.isin(outliers_z.index)].reset_index(drop=True)

print(f"Numărul de rânduri originale: {data.shape[0]}")
print(f"Numărul de rânduri după eliminare: {cleaned_data.shape[0]}")

output_file_path = os.path.join(base_directory, "Dataset-Pisici-Curat.xlsx")
cleaned_data.to_excel(output_file_path, index=False)

print(f"Setul de date curățat a fost salvat în {output_file_path}")
