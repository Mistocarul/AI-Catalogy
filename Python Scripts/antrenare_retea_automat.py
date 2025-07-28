import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# 1. Citirea fișierului Excel
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")

data = pd.read_excel(file_path_excel)

# 2. Pregătirea datelor
X = data.drop('Rasa', axis=1)
y = data['Rasa']

# Codificarea etichetelor
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalizarea caracteristicilor
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 3. Crearea modelului de rețea neuronală
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')  # Clasificare multiclasă
])

# 4. Compilarea modelului
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adăugăm EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 5. Antrenarea modelului și salvarea istoricului
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# 6. Vizualizarea convergenței (erorilor)
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eroare pe antrenament')
plt.plot(history.history['val_loss'], label='Eroare pe validare')
plt.title('Convergența erorii în funcție de numărul de epoci')
plt.xlabel('Epoci')
plt.ylabel('Eroare')
plt.legend()
plt.grid(True)
plt.show()

# 7. Evaluarea modelului pe setul de testare
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Acuratețea pe setul de testare: {accuracy:.2f}')

# 8. Predicții pe setul de testare și afișarea matricei de confuzie
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Calcularea matricei de confuzie
cm = confusion_matrix(y_test, y_pred_classes)

# Vizualizarea matricei de confuzie
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matricea de confuzie')
plt.show()

# 9. Vizualizarea punctelor clasificate greșit
misclassified_indices = np.where(y_test != y_pred_classes)[0]

# Reducem datele la 2D folosind PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(10, 7))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', alpha=0.6, label='Clasificate corect')
plt.scatter(X_test_pca[misclassified_indices, 0], X_test_pca[misclassified_indices, 1], c='red', edgecolor='k', label='Clasificate greșit')
plt.title('Punctele clasificate greșit')
plt.xlabel('Componenta principală 1')
plt.ylabel('Componenta principală 2')
plt.legend()
plt.show()
