#!pip install ucimlrepo
#
# Análise de sequências de nucleotídeos ATGC de DNA
#
import locale
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


# Define padrões de Data e Hora para o Brasil
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')

#
# Carregar o conjunto de dados
molecular_biology_splice_junction_gene_sequences = fetch_ucirepo(id=69)
X = molecular_biology_splice_junction_gene_sequences.data.features
y = molecular_biology_splice_junction_gene_sequences.data.targets


# Verificando dados
print(molecular_biology_splice_junction_gene_sequences.metadata)
print(molecular_biology_splice_junction_gene_sequences.variables)

# Codificação One-hot para sequências de DNA
def encode_sequences(X):
    # Flatten the sequences into a single array
    sequences = np.array(X)
    flat_sequences = sequences.reshape(-1)

    # One-hot encode the characters
    encoder = OneHotEncoder(categories='auto')
    one_hot_encoded = encoder.fit_transform(flat_sequences[:, None]).toarray()

    return one_hot_encoded.reshape(sequences.shape[0], -1)


X_encoded = encode_sequences(X)

# Codificação das classes
label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# y_encoded = label_encoder.fit_transform(y.ravel())
y_encoded = label_encoder.fit_transform(
    y.to_numpy().ravel())  # Problemas na função ravel() do numpy vindo de dataframe Pandas

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

#Treinamento com RandomForest

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

# Avaliação
print("Random Forest Results:")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print("Accuracy:", accuracy_score(y_test, rf_y_pred))

#
# Treinamento em SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
svm_y_pred = svm_classifier.predict(X_test)

# Avaliação
print("\nSVM Results:")
print(confusion_matrix(y_test, svm_y_pred))
print(classification_report(y_test, svm_y_pred))
print("Accuracy:", accuracy_score(y_test, svm_y_pred))

#
# Treinamento em Rede Neural MLP
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_classifier.fit(X_train, y_train)
mlp_y_pred = mlp_classifier.predict(X_test)

# Avaliação
print("\nResultados pela Rede Neural MLPClassifier:")
print(confusion_matrix(y_test, mlp_y_pred))
print(classification_report(y_test, mlp_y_pred))
print("Accuracy:", accuracy_score(y_test, mlp_y_pred))


#
# Treinamento e análise com uma rede KBNN (Knowledge-Based Artificial Neural Network),
# ou Rede Neural Artificial Baseada em Conhecimento

# Definindo o modelo KBANN simples usando Keras
def create_kbann_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Camada de entrada usando Input
    model.add(Dense(128, activation='relu'))  # Camada oculta
    model.add(Dropout(0.5))  # Dropout para reduzir overfitting, é uma das técnicas de regularização para combater o overfitting.
    model.add(Dense(64, activation='relu'))  # Outra camada oculta
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(32, activation='relu'))  # Outra camada oculta
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Camada de saída

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Criar e treinar o modelo
kbann_model = create_kbann_model(X_train.shape[1])
kbann_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
kbann_model.summary()

# Avaliação
kbann_y_pred = np.argmax(kbann_model.predict(X_test), axis=-1)

# Resultados
print("\nResultados do KBANN:")
print(confusion_matrix(y_test, kbann_y_pred))
print(classification_report(y_test, kbann_y_pred))
print("Accuracy:", accuracy_score(y_test, kbann_y_pred))
