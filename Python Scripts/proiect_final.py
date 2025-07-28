import spacy
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from antrenare_retea_manual import prezice_rasa, antreneaza_retea
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.metrics.cluster import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

nlp = spacy.load('en_core_web_sm')

atribute_mapare = {
    "Varsta": {
        "sub 1 an": 1,
        "1-2 ani": 2,
        "2-10 ani": 3,
        "peste 10 ani": 4,
        "nu stiu": 0
    },
    "Numar-pisici-gospodarie": {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "mai mult de 5": 6
    },
    "Tip-locuinta": {
        "Apartament fara balcon": 1,  # Apartament fără balcon
        "Apartament cu balcon sau terasa": 2,  # Apartament cu balcon sau terasă
        "Casa cartier rezidential": 5,   # Casă cartier rezidențial
        "Casa individuala": 6   # Casă individuală
    },
    "Zona-locuinta": {
        "Urban": 1,    # Urban
        "Periurban": 2,   # Periurban
        "Rural": 3     # Rural
    },
    "Timp-petrecut-afara": {
        "Niciodata": 1,    # Niciodată
        "Limitat sub 1 ora": 2,    # Limitat sub 1 oră
        "Moderat 1-5 ore": 3,    # Moderat 1-5 ore
        "Lung peste 5 ore": 4,    # Lung peste 5 ore
        "Tot timpul": 5     # Tot timpul
    },
    "Timiditate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Calmitate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Sperietura": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Vigilenta": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Perseveranta": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Afectivitate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Prietenie": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Singuratate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Dominanta": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Agresivitate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Impulsivitate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Previzibilitate": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Cat-de-distras": {
        "Deloc": 1,    # Deloc
        "Usor": 2,    # Ușor
        "Moderat": 3,    # Moderat
        "Foarte": 4,    # Foarte
        "Extrem": 5     # Extrem
    },
    "Abundenta-naturii": {
        "Scazuta": 1,    # Scăzută
        "Moderata": 2,    # Moderată
        "Ridicata": 3,    # Ridicată
        "Nu stiu": 0   # Nu știu
    },
    "Frecventa-atac-pasari": {
        "Niciodata": 1,    # Niciodată
        "Rar 1-5 ori pe an": 2,    # Rar 1-5 ori pe an
        "Uneori 5-10 ori pe an": 3,    # Uneori 5-10 ori pe an
        "Des 12-36 ori pe an": 4,    # Des 12-36 ori pe an
        "Foarte des 48-144 ori pe an": 5     # Foarte des 48-144 ori pe an
    }
}

text = ("Pisica mea este foarte prietenoasa si deloc agresiva. Ea este usor distrasa si usor previzibila."
        "Nu este deloc dominanta si nu este deloc singuratica. Ea este foarte afectuoasa si usor perseveranta."
        "Apartament fara balcon este locuinta. Pisica mea are 2 ani si petrece sub 1 ora timp afara."
        "Pisica are varsta de sub 1 an si am un numar de peste 10 pisici in gospodarie."
        "Zona locuintei este urbana. Este extrem de timida. Este foarte calma si nu se sperie usor."
        "Este moderat impulsiva. De asemenea, pisica se afla intr-o zona cu abundenta in natura moderata."
        )
model = SentenceTransformer('all-MiniLM-L6-v2')
atribute_chei = list(atribute_mapare.keys())
cuvinte_text = text.split()

embeddings_text = model.encode(cuvinte_text, convert_to_tensor=True)
embeddings_atribute = model.encode(atribute_chei, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings_text, embeddings_atribute)

atribute_extrase = {}

def verifica_cuvinte_in_jur(i, cuvinte_text, atribut_similar, model, atribut):
    start = max(i - 2, 0)
    end = min(i + 2 + 1, len(cuvinte_text))
    for idx in range(start, end):
        cuvant = cuvinte_text[idx]
        for valoare, _ in atribut_similar.items():
            embedding_cuvant = model.encode(cuvant, convert_to_tensor=True)
            cosine_scores_cuvant = util.cos_sim(embedding_cuvant, model.encode([valoare], convert_to_tensor=True))
            if cosine_scores_cuvant[0][0] > 0.5:
                if atribut not in atribute_extrase:
                    atribute_extrase[atribut] = atribut_similar[valoare]

for i in range(len(cuvinte_text)):
    for j in range(len(atribute_chei)):
        if cosine_scores[i][j] > 0.5:
            atribut_similar = atribute_mapare[atribute_chei[j]]
            verifica_cuvinte_in_jur(i, cuvinte_text, atribut_similar, model, atribute_chei[j])



for chei, valori in atribute_mapare.items():
    if chei not in atribute_extrase:
        atribute_extrase[chei] = 0

print(atribute_extrase)

input_retea = []
for cheie in atribute_chei:
    input_retea.append(atribute_extrase[cheie])

print(input_retea)

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output = antreneaza_retea(file_path_excel)
predictie = prezice_rasa(np.array([input_retea]), weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)

rase_pisici ={
    0: "Necunoscuta",
    1: "Bengal",
    2: "Birmaneza",
    3: "British Shorthair",
    4: "Chartreux",
    5: "European",
    6: "Maine Coon",
    7: "Persian",
    8: "Ragdoll",
    9: "Sphynx",
    10: "Siameza",
    11: "Angora-turceasca",
    12: "Alta rasa/Necunoscut"
}

print("\n")
print(f"Pisica ta are rasa prezisa: {rase_pisici[predictie]}")

#Descrierea unei rase de pisici
rasa_pisica_descriere = input("Alegeti o rasa de pisica pentru a face o descriere: ")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = f"Write me a description about the {rasa_pisica_descriere} cat breed:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=150,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_k=10,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


#Comparatia intre doua rase de pisici
rasa_pisica_descriere_1 = input("Alegeti o rasa de pisica pentru a face o comparare: ")
rasa_pisica_descriere_2 = input("Alegeti o a doua rasa pentru a face compararea: ")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = f"Make a comparison between cat breeds {rasa_pisica_descriere_1} and {rasa_pisica_descriere_2}:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=150,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_k=10,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)



#Comparare cu K-Means
print("\n")
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
df = pd.read_excel(file_path_excel)
atribute = df.drop(columns=['Rasa'])

inertia = []
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(atribute)
    inertia.append(kmeans.inertia_)

# Vizualizăm metoda Elbow pentru a alege numărul optim de clustere
plt.plot(range(1, 30), inertia)
plt.title("Metoda Elbow")
plt.xlabel("Numărul de clustere (k)")
plt.ylabel("Inertia")
plt.show()


k = 13

kmeans = KMeans(n_clusters=k, random_state=42)
atribute['Cluster'] = kmeans.fit_predict(atribute)

pd.set_option('display.max_columns', None)  # Afișează toate coloanele
pd.set_option('display.width', 1000)       # Crește lățimea totală pentru afișare

print("\nRezultatele clustering-ului (fiecare pisică atribuită unui cluster):")
print(atribute.head())

centroids = kmeans.cluster_centers_
print("\nCentroids (coordonate ale centroidurilor pentru fiecare cluster):")
print(centroids)

plt.scatter(atribute.iloc[:, 0], atribute.iloc[:, 1], c=atribute['Cluster'], cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')  # Centroid
plt.title("Vizualizare clustering - Centroiduri")
plt.xlabel(atribute.columns[0])
plt.ylabel(atribute.columns[1])
plt.show()

# Afișăm caracteristicile medii ale fiecărui cluster
cluster_means = atribute.groupby('Cluster').mean()
print("\nCaracteristicile medii ale fiecărui cluster:")
print(cluster_means)

df_heatmap = cluster_means
plt.figure(figsize=(13, 8))
sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Media caracteristicilor'})
plt.title("Caracteristicile medii ale fiecărui cluster")
plt.ylabel("Cluster")
plt.xlabel("Caracteristică")
plt.xticks(rotation=45, ha="right")
plt.show()

df['Cluster'] = atribute['Cluster']
rasa_per_cluster = df.groupby('Cluster')['Rasa'].value_counts()
rasa_table = df.groupby(['Cluster', 'Rasa']).size().unstack(fill_value=0)


plt.figure(figsize=(13, 8))
sns.heatmap(rasa_table, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Număr de pisici'})
plt.title("Distribuția raselor în fiecare cluster")
plt.ylabel("Cluster")
plt.xlabel("Rasa")
plt.xticks(rotation=45, ha="right")
plt.show()

# Calculăm ARI (Adjusted Rand Index) pentru a evalua calitatea clustering-ului
ari_score = adjusted_rand_score(df['Rasa'], df['Cluster'])
print(f"Adjusted Rand Index (ARI): {ari_score:.2f}")

# Calculăm Silhouette Score pentru a evalua calitatea clustering-ului
silhouette_avg = silhouette_score(atribute.drop(columns=['Cluster']), df['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")


#Importanta caracteristicilor
rase_pisici = {
    0: "Necunoscuta",
    1: "Bengal",
    2: "Birmaneza",
    3: "British Shorthair",
    4: "Chartreux",
    5: "European",
    6: "Maine Coon",
    7: "Persian",
    8: "Ragdoll",
    9: "Sphynx",
    10: "Siameza",
    11: "Angora-turceasca",
    12: "Alta rasa/Necunoscut"
}

current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
df = pd.read_excel(file_path_excel)

X = df.drop(columns=['Rasa'])
y = df['Rasa']

importanta_caracteristici_per_rasa = {}

for rasa in df['Rasa'].unique():
    y_rasa = (y == rasa).astype(int)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y_rasa)
    importanta_caracteristici_per_rasa[rase_pisici[rasa]] = rf.feature_importances_

importanta_df = pd.DataFrame(importanta_caracteristici_per_rasa, index=X.columns)

plt.figure(figsize=(12, 8))
sns.heatmap(importanta_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Importanța caracteristicilor'}, fmt=".2f")
plt.title("Importanța caracteristicilor pentru fiecare rasă de pisică")
plt.xlabel("Rase")
plt.ylabel("Caracteristici")
plt.xticks(rotation=45, ha="right")
plt.show()



#Comparare cu AdaBoost
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")
df = pd.read_excel(file_path_excel)

X = df.drop(columns=['Rasa'])
y = df['Rasa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_classifier = DecisionTreeClassifier(max_depth=16)
ada_boost = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=42)

ada_boost.fit(X_train, y_train)
y_pred = ada_boost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratețea modelului AdaBoost: {accuracy * 100:.2f}%")


