import pandas as pd
import os
import numpy as np
from scipy.stats import chi2_contingency

def calculeaza_frecvente(file_path):
    df = pd.read_excel(file_path)
    df = df.sort_values(by='Rasa')
    num_instante_rasa = df['Rasa'].value_counts()
    print("Numărul de instanțe pentru fiecare clasă (Rasă):")
    num_instante_rasa = num_instante_rasa.sort_index()
    print(num_instante_rasa.to_string())

    print("\nValori distincte și frecvențe pentru fiecare atribut:")
    for column in df.columns[0: ]:
        valori_distincte = df[column].value_counts()
        print(f"\nAtribut: {column}")
        print(valori_distincte.to_string())

        for rasa in df['Rasa'].unique():
            print(f"\nFrecvențele pentru rasa {rasa}:")
            frecvente_rasa = df[df['Rasa'] == rasa][column].value_counts()
            print(frecvente_rasa.to_string())

def calculeaza_corelatii(file_path):
    df = pd.read_excel(file_path)
    categorical_attributes = ['Sex', 'Varsta', 'Numar-pisici-gospodarie', 'Tip-locuinta', 'Zona-locuinta',
                       'Timp-petrecut-afara', 'Timp-petrecut-cu-pisica', 'Timiditate', 'Calmitate',
                       'Sperietura', 'Inteligenta', 'Vigilenta', 'Perseverenta', 'Afectivitate',
                       'Prietenie', 'Singuratate', 'Brutalitate', 'Dominanta', 'Agresivitate',
                       'Impulsivitate', 'Previzibilitate', 'Cat-de-distras', 'Abundenta-naturii',
                       'Frecventa-atac-pasari', 'Frecventa-atac-mamifere-mici']
    chi_square_results = []
    for race in df['Rasa'].unique():
        for attr in categorical_attributes:
            contingency_table = pd.crosstab(df[attr], df['Rasa'] == race)
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            chi_square_results.append({
                'Rasa': race,
                'Atribut': attr,
                'Chi2 Statistic': chi2,
                'P-value': p
            })

    chi_square_results = sorted(chi_square_results, key=lambda x: x['Rasa'])
    print("Rezultatele testului Chi-pătrat:")
    for result in chi_square_results:
        print(
            f"Rasa {result['Rasa']} - Atribut: {result['Atribut']} - Chi2: {result['Chi2 Statistic']}, P-value: {result['P-value']}")

    results_df = pd.DataFrame(chi_square_results)
    results_df['Rasa'] = results_df['Rasa'].astype(int)
    results_df = results_df.sort_values(by='Rasa')
    output_file_path = os.path.join(base_directory, "Corelatie_clase_atribute.xlsx")
    results_df.to_excel(output_file_path, index=False)
    print(f"Rezultatele au fost salvate în: {output_file_path}")


#Main
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path = os.path.join(base_directory, "Dataset-Pisici-Modificat.xlsx")

# Apelarea funcțiilor
calculeaza_frecvente(file_path)
#calculeaza_corelatii(file_path)
