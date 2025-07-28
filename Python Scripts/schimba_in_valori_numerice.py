import os
import pandas as pd
from pathlib import Path

from openpyxl.styles.builtins import currency

def modifyExcel(file_path, base_directory) :

    try:
        df = pd.read_excel(file_path)

        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].replace({'M': 1, 'F': 2, 'NSP': 0})
        else:
            print("Sex not found")
            return

        if 'Varsta' in df.columns:
            df['Varsta'] = df['Varsta'].replace({'Moinsde1': 1,
                                                 '1a2': 2,
                                                 '2a10': 3,
                                                 'Plusde10': 4,
                                                 'Adaugata-nou': 0})
        else:
            print("Varsta not found")
            return

        if 'Rasa' in df.columns:

            valid_rasa_values = ['BEN', 'SBI', 'BRI', 'CHA', 'EUR', 'MCO', 'PER', 'RAG', 'SPH', 'ORI', 'TUV', 'Autre',
                                 'NSP']

            df = df[df['Rasa'].isin(valid_rasa_values)]

            df['Rasa'] = df['Rasa'].replace({'BEN': 1,
                                             'SBI': 2,
                                             'BRI': 3,
                                             'CHA': 4,
                                             'EUR': 5,
                                             'MCO': 6,
                                             'PER': 7,
                                             'RAG': 8,
                                             'SPH': 9,
                                             'ORI': 10,
                                             'TUV': 11,
                                             'Autre': 12,
                                             'NSP': 0})
        else:
            print("Rasa not found")
            return

        if 'Numar-pisici-gospodarie' in df.columns:
            df['Numar-pisici-gospodarie'] = df['Numar-pisici-gospodarie'].replace({'Plusde5': 6})
        else:
            print("Numar-pisici-gospodarie not found")
            return

        if 'Tip-locuinta' in df.columns:
            df['Tip-locuinta'] = df['Tip-locuinta'].replace({'ASB': 1,
                                                             'AAB': 2,
                                                             'ML': 5,
                                                             'MI': 6})
        else:
            print("Tip-locuinta not found")
            return

        if 'Zona-locuinta' in df.columns:
            df['Zona-locuinta'] = df['Zona-locuinta'].replace({'U': 1,
                                                               'PU': 2,
                                                               'R': 5})
        else:
            print("Zona-locuinta not found")
            return

        if 'Timp-petrecut-afara' in df.columns:
            df['Timp-petrecut-afara'] = df['Timp-petrecut-afara'].replace({'0': 1,
                                                                           '1': 2,
                                                                           '2': 3,
                                                                           '3': 4,
                                                                           '4': 5})
        else:
            print("Timp-petrecut-afara not found")
            return

        if 'Timp-petrecut-cu-pisica' in df.columns:
            df['Timp-petrecut-cu-pisica'] = df['Timp-petrecut-cu-pisica'].replace({'0': 1,
                                                                                   '1': 2,
                                                                                   '2': 3,
                                                                                   '3': 4,})
        else:
            print("Timp-petrecut-cu-pisica not found")
            return

        if 'Abundenta-naturii' in df.columns:
            df['Abundenta-naturii'] = df['Abundenta-naturii'].replace({'NSP': 0})
        else:
            print("Abundenta-naturii not found")
            return

        if 'Frecventa-atac-pasari' in df.columns:
            df['Frecventa-atac-pasari'] = df['Frecventa-atac-pasari'].replace({'0': 1,
                                                                               '1': 2,
                                                                               '2': 3,
                                                                               '3': 4,
                                                                               '4': 5})
        else:
            print("Frecventa-atac-pasari not found")
            return

        if 'Frecventa-atac-mamifere-mici' in df.columns:
            df['Frecventa-atac-mamifere-mici'] = df['Frecventa-atac-mamifere-mici'].replace({'0': 1,
                                                                                             '1': 2,
                                                                                             '2': 3,
                                                                                             '3': 4,
                                                                                             '4': 5})
        else:
            print("Frecventa-atac-mamifere-mici not found")
            return

        output = os.path.join(base_directory, "Dataset-Pisici-Modificat.xlsx")
        df.to_excel(output, index=False)
        print("Excel file modified successfully")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except Exception as e:
        print("Error: ", e)
        return

def search_file(file_path):
    try:
        with open(file_path, 'r'):
            return True
    except FileNotFoundError:
        return False

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
excel_path = os.path.join(base_directory, "Dataset-Pisici.xlsx")

if search_file(excel_path):
    print('File found')
    modifyExcel(excel_path, base_directory)
else:
    print('File not found')