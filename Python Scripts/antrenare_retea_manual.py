import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def standard_scale(X):
    means = np.mean(X, axis=0) # Calculăm media pentru fiecare caracteristică
    stds = np.std(X, axis=0) # Calculăm deviația standard pentru fiecare caracteristică
    return (X - means) / stds # Normalizăm datele

# 1. Citirea fișierului Excel
current_directory = os.getcwd()
base_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
file_path_excel = os.path.join(base_directory, "Dataset-Pisici-Echilibrat.xlsx")

data = pd.read_excel(file_path_excel)

# 2. Pregătirea datelor
X = data.drop('Rasa', axis=1).values  # Convertim la array numpy
y = data['Rasa'].values  # Convertim la array numpy

# Normalizarea caracteristicilor
X = standard_scale(X)

# Împărțirea datelor în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Inițializarea parametrilor rețelei
input_size = X_train.shape[1] # Numărul de caracteristici
hidden_1_size = 512 # Numărul de neuroni din primul strat ascuns
hidden_2_size = 256 # Numărul de neuroni din al doilea strat ascuns
output_size = len(np.unique(y)) # Numărul de clase
learning_rate = 0.1 # Rata de învățare
epochs = 25 # Numărul de epoci
batch_size = 32 # Dimensiunea batch-ului

# Inițializarea ponderilor folosind He initialization (ajuta la convergența mai rapidă)
weights_input_hidden_1 = np.random.randn(input_size, hidden_1_size) * np.sqrt(2. / input_size)
weights_hidden_1_hidden_2 = np.random.randn(hidden_1_size, hidden_2_size) * np.sqrt(2. / hidden_1_size)
weights_hidden_2_output = np.random.randn(hidden_2_size, output_size) * np.sqrt(2. / hidden_2_size)

# Inițializarea biasele
bias_hidden_1 = np.zeros((1, hidden_1_size))
bias_hidden_2 = np.zeros((1, hidden_2_size))
bias_output = np.zeros((1, output_size))

# Funcțiile de activare și derivatele acestora
def relu(x): # Funcția de activare ReLU
    return np.maximum(0, x)

def relu_derivative(x): # Derivata funcției de activare ReLU
    return np.where(x > 0, 1, 0)

def softmax(x): # Funcția de clasificare pentru ultimul strat(output)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilizare pentru overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Funcția de eroare (cross-entropy)
def cross_entropy_loss(y_true, y_pred): # y_true - etichetele reale, y_pred - predicțiile rețelei
    m = y_true.shape[0] # Numărul de exemple
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # Adăugăm o valoare mică pentru a evita log(0)

# Propagarea înainte
def forward_propagation(X):
    global weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output
    global bias_hidden_1, bias_hidden_2, bias_output

    hidden_1 = relu(np.dot(X, weights_input_hidden_1) + bias_hidden_1) # Propagarea înainte pentru primul strat ascuns
    hidden_2 = relu(np.dot(hidden_1, weights_hidden_1_hidden_2) + bias_hidden_2) # Propagarea înainte pentru al doilea strat ascuns
    output = softmax(np.dot(hidden_2, weights_hidden_2_output) + bias_output) # Propagarea înainte pentru stratul de ieșire

    return hidden_1, hidden_2, output

# Propagarea înapoi (backpropagation)
def backward_propagation(X, y, hidden_1, hidden_2, output):
    global weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output
    global bias_hidden_1, bias_hidden_2, bias_output

    m = X.shape[0] # Numărul de exemple

    # Calculul erorilor pentru fiecare strat
    d_output = output - y # Eroarea pentru stratul de ieșire
    d_hidden_2 = np.dot(d_output, weights_hidden_2_output.T) * relu_derivative(hidden_2) # Eroarea pentru al doilea strat ascuns
    d_hidden_1 = np.dot(d_hidden_2, weights_hidden_1_hidden_2.T) * relu_derivative(hidden_1) # Eroarea pentru primul strat ascuns

    # Calculul gradientului pentru ponderi și biase
    d_weights_hidden_2_output = np.dot(hidden_2.T, d_output) / m # Gradientul pentru ponderile dintre al doilea strat ascuns și stratul de ieșire
    d_weights_hidden_1_hidden_2 = np.dot(hidden_1.T, d_hidden_2) / m # Gradientul pentru ponderile dintre primul și al doilea strat ascuns
    d_weights_input_hidden_1 = np.dot(X.T, d_hidden_1) / m # Gradientul pentru ponderile dintre stratul de intrare și primul strat ascuns

    d_bias_output = np.sum(d_output, axis=0, keepdims=True) / m # Gradientul pentru biasele stratului de ieșire
    d_bias_hidden_2 = np.sum(d_hidden_2, axis=0, keepdims=True) / m # Gradientul pentru biasele stratului ascuns 2
    d_bias_hidden_1 = np.sum(d_hidden_1, axis=0, keepdims=True) / m # Gradientul pentru biasele stratului ascuns 1

    # Actualizarea ponderilor folosind gradient descent
    weights_hidden_2_output -= learning_rate * d_weights_hidden_2_output # Actualizarea ponderilor dintre al doilea strat ascuns și stratul de ieșire
    weights_hidden_1_hidden_2 -= learning_rate * d_weights_hidden_1_hidden_2 # Actualizarea ponderilor dintre primul și al doilea strat ascuns
    weights_input_hidden_1 -= learning_rate * d_weights_input_hidden_1 # Actualizarea ponderilor dintre stratul de intrare și primul strat ascuns

    bias_output -= learning_rate * d_bias_output # Actualizarea biasei stratului de ieșire
    bias_hidden_2 -= learning_rate * d_bias_hidden_2 # Actualizarea biasei stratului ascuns 2
    bias_hidden_1 -= learning_rate * d_bias_hidden_1 # Actualizarea biasei stratului ascuns 1

train_losses = []
test_losses = []

# Antrenarea rețelei
for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = np.eye(output_size)[y_train[i:i+batch_size]]

        # Propagarea înainte și înapoi
        hidden_1, hidden_2, output = forward_propagation(X_batch)
        backward_propagation(X_batch, y_batch, hidden_1, hidden_2, output)

    # Calcularea pierderii pe setul de antrenament
    _, _, output_train = forward_propagation(X_train)
    train_loss = cross_entropy_loss(np.eye(output_size)[y_train], output_train)
    train_losses.append(train_loss)

    # Calcularea pierderii pe setul de testare
    _, _, output_test = forward_propagation(X_test)
    test_loss = cross_entropy_loss(np.eye(output_size)[y_test], output_test)
    test_losses.append(test_loss)

    print(f'Epoca {epoch+1}/{epochs} - Pierdere antrenament: {train_loss:.4f} - Pierdere testare: {test_loss:.4f}')

# Evaluarea modelului pe setul de testare
_, _, output_test = forward_propagation(X_test)
y_pred_classes = np.argmax(output_test, axis=1)
accuracy = np.mean(y_pred_classes == y_test)
print(f'Acuratețea pe setul de testare: {accuracy:.2f}')

# Afișarea matricei de confuzie
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues', values_format='d')
plt.title('Matricea de confuzie')
plt.show()

# Vizualizarea convergenței erorilor
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Eroare pe antrenament')
plt.plot(test_losses, label='Eroare pe testare')
plt.title('Convergența erorii în funcție de numărul de epoci')
plt.xlabel('Epoci')
plt.ylabel('Eroare')
plt.legend()
plt.grid(True)
plt.show()

# Vizualizarea punctelor clasificate greșit
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




























def standard_scale(X, epsilon=1e-8):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return (X - means) / (stds + epsilon)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

# Funcțiile de propagare înainte și înapoi
def forward_propagation(X, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output):
    hidden_1 = relu(np.dot(X, weights_input_hidden_1) + bias_hidden_1)
    hidden_2 = relu(np.dot(hidden_1, weights_hidden_1_hidden_2) + bias_hidden_2)
    output = softmax(np.dot(hidden_2, weights_hidden_2_output) + bias_output)
    return hidden_1, hidden_2, output

def backward_propagation(X, y, hidden_1, hidden_2, output, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output, learning_rate):
    m = X.shape[0]
    d_output = output - y
    d_hidden_2 = np.dot(d_output, weights_hidden_2_output.T) * relu_derivative(hidden_2)
    d_hidden_1 = np.dot(d_hidden_2, weights_hidden_1_hidden_2.T) * relu_derivative(hidden_1)

    d_weights_hidden_2_output = np.dot(hidden_2.T, d_output) / m
    d_weights_hidden_1_hidden_2 = np.dot(hidden_1.T, d_hidden_2) / m
    d_weights_input_hidden_1 = np.dot(X.T, d_hidden_1) / m

    d_bias_output = np.sum(d_output, axis=0, keepdims=True) / m
    d_bias_hidden_2 = np.sum(d_hidden_2, axis=0, keepdims=True) / m
    d_bias_hidden_1 = np.sum(d_hidden_1, axis=0, keepdims=True) / m

    weights_hidden_2_output -= learning_rate * d_weights_hidden_2_output
    weights_hidden_1_hidden_2 -= learning_rate * d_weights_hidden_1_hidden_2
    weights_input_hidden_1 -= learning_rate * d_weights_input_hidden_1

    bias_output -= learning_rate * d_bias_output
    bias_hidden_2 -= learning_rate * d_bias_hidden_2
    bias_hidden_1 -= learning_rate * d_bias_hidden_1

    return weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output

# Funcția de antrenare
def antreneaza_retea(file_path_excel, epochs=25, learning_rate=0.1, batch_size=32):
    # Citirea fișierului Excel
    data = pd.read_excel(file_path_excel)

    # Pregătirea datelor
    X = data.drop('Rasa', axis=1).values
    y = data['Rasa'].values

    # Normalizarea caracteristicilor
    X = standard_scale(X)

    # Împărțirea datelor în seturi de antrenament și testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inițializarea parametrilor rețelei
    input_size = X_train.shape[1]
    hidden_1_size = 512
    hidden_2_size = 256
    output_size = len(np.unique(y))

    # Inițializarea ponderilor și biasele
    weights_input_hidden_1 = np.random.randn(input_size, hidden_1_size) * np.sqrt(2. / input_size)
    weights_hidden_1_hidden_2 = np.random.randn(hidden_1_size, hidden_2_size) * np.sqrt(2. / hidden_1_size)
    weights_hidden_2_output = np.random.randn(hidden_2_size, output_size) * np.sqrt(2. / hidden_2_size)

    bias_hidden_1 = np.zeros((1, hidden_1_size))
    bias_hidden_2 = np.zeros((1, hidden_2_size))
    bias_output = np.zeros((1, output_size))

    train_losses = []
    test_losses = []

    # Antrenarea rețelei
    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = np.eye(output_size)[y_train[i:i+batch_size]]

            hidden_1, hidden_2, output = forward_propagation(X_batch, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)
            weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output = backward_propagation(X_batch, y_batch, hidden_1, hidden_2, output, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output, learning_rate)

        # Pierderea pe seturile de antrenament și testare
        _, _, output_train = forward_propagation(X_train, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)
        train_loss = cross_entropy_loss(np.eye(output_size)[y_train], output_train)
        train_losses.append(train_loss)

        _, _, output_test = forward_propagation(X_test, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)
        test_loss = cross_entropy_loss(np.eye(output_size)[y_test], output_test)
        test_losses.append(test_loss)

        print(f'Epoca {epoch+1}/{epochs} - Pierdere antrenament: {train_loss:.4f} - Pierdere testare: {test_loss:.4f}')

    # Evaluarea modelului
    _, _, output_test = forward_propagation(X_test, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)
    y_pred_classes = np.argmax(output_test, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    print(f'Acuratețea pe setul de testare: {accuracy:.2f}')

    # Returnează modelele (ponderile și biasele)
    return weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output

# Funcția pentru prezicerea rasei
def prezice_rasa(atribute_extrase, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output):
    atribut_extras_normalizat = standard_scale(atribute_extrase)  # Aplicăm normalizarea
    _, _, output_nou = forward_propagation(atribut_extras_normalizat, weights_input_hidden_1, weights_hidden_1_hidden_2, weights_hidden_2_output, bias_hidden_1, bias_hidden_2, bias_output)
    rasa_predicție = np.argmax(output_nou, axis=1)  # Indexul clasei prezise
    return rasa_predicție[0]
