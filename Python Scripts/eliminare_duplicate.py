import pandas as pd
import os

def eliminate_duplicates(file_path_excel, file_path_save_duplicates):
    df = pd.read_excel(file_path_excel)
    duplicate_rows = df[df.duplicated(subset=df.columns[1:], keep=False)]

    if duplicate_rows.empty:
        print('No duplicates found')
        return

    with open(file_path_save_duplicates, 'w') as f:
        for index, row in duplicate_rows.iterrows():
            f.write(f'Rând duplicat (Nr_rand {row["Nr_rand"]}):\n')
            f.write(row.to_string())
            f.write('\n\n')

    df_without_duplicates = df.drop_duplicates(subset=df.columns[1:], keep='first')
    df_without_duplicates.to_excel(file_path_excel, index=False)
    print('Finished eliminating duplicates')

def search_file(file_path):
    try:
        with open(file_path, 'r'):
            return True
    except FileNotFoundError:
        return False

current_directory = os.getcwd()
print(f'Directorul curent: {current_directory}')

base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
print(f'Directorul de bază: {base_directory}')

file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Modificat.xlsx")
file_path_save_duplicates = os.path.join(base_directory, "DuplicatesFound.txt")
print(f'Calea fișierului Excel: {file_path_excel}')
print(f'Calea fișierului pentru salvarea duplicatelor: {file_path_save_duplicates}')

if search_file(file_path_excel):
    print('File found')
    eliminate_duplicates(file_path_excel, file_path_save_duplicates)
else:
    print('File not found')