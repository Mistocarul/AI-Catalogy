import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")

data = pd.read_excel(file_path_excel)


def plot_boxplot(data, attribute):
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Rasa', y=attribute, data=data, palette='Set2', linewidth=1)

    for line in ax.lines[4::6]:
        line.set_linewidth(3)

    plt.title(f'Boxplot pentru {attribute} pe Rasele de Pisici')
    plt.xlabel('Rasa')
    plt.ylabel(attribute)
    plt.xticks(rotation=45)
    plt.show()

plot_boxplot(data, 'Perseverenta')
