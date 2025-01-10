#!pip install ucimlrepo
#
# Análise de sequências de nucleotídeos ATGC de DNA
#
import locale
import os.path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desativa o uso da GPU
from tensorflow import get_logger, random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

#  Define o path dos arquivos Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define padrões de Data e Hora para o Brasil
locale.setlocale(locale.LC_ALL, 'pt_BR.utf8')
# Suprimir todos os warnings
warnings.filterwarnings('ignore')
# Definições de nível de Log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1: INFO, 2: WARNING, 3: ERROR
get_logger().setLevel('ERROR')  # Configura apenas erros a serem exibidos

# Definindo a semente para numpy
np.random.seed(42)

# Definindo a semente para TensorFlow
random.set_seed(42)

#
# Carregar o conjunto de dados
molecular_biology_splice_junction_gene_sequences = fetch_ucirepo(id=69)
X = molecular_biology_splice_junction_gene_sequences.data.features
y = molecular_biology_splice_junction_gene_sequences.data.targets

# Garanta que X e y são arrays NumPy
if isinstance(X, pd.DataFrame):
    X = X.values  # Converte para array NumPy se necessário

if isinstance(y, (pd.DataFrame, pd.Series)):
    y = y.values  # Converte para array NumPy se necessário


# Codificação One-hot para sequências de DNA
def encode_sequences(X):
    # Achatar (flatten) as sequências em um array simples
    sequences = np.array(X)
    flat_sequences = sequences.reshape(-1)

    # One-hot encode the characters
    encoder = OneHotEncoder(categories='auto')
    one_hot_encoded = encoder.fit_transform(flat_sequences[:, None]).toarray()
    return one_hot_encoded.reshape(sequences.shape[0], -1)


X_encoded = encode_sequences(X)

# Codificação das classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # .to_numpy().ravel()) # Problemas na função ravel() do numpy vindo de dataframe Pandas

# Divisão em treino e teste (20% para teste e 80% para treinamento)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Mostrando índices dos rótulos e valores
print("Rótulos codificados:", y_encoded)
# print("Tipo de dado dos rótulos: ", type(y))
print("Rótulos originais:", np.unique(y))  # y são os rótulos originais


# Treinamento com RandomForest
def random_forest():
    # Hiperparâmetros: n_estimators=100 -> árvores de decisão a serem utilizadas e,
    # random_state=42 -> Valor da semente de aleatoriedade do estimador (ideal manter constante)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    rf_y_pred = rf_classifier.predict(X_test)

    # Avaliação
    print("Random Forest Results:")
    print(confusion_matrix(y_test, rf_y_pred))
    print(classification_report(y_test, rf_y_pred))
    print("Accuracy:", accuracy_score(y_test, rf_y_pred))


#
# Treinamento em SVM - Support Vector Machine
def svm_exec():
    # Hiperparâmetros: kernel=linear, para classes que podem ser separadas linearmente
    # random_state=42 -> Valor da semente de aleatoriedade do estimador (ideal manter constante)
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
def mlp_exec():
    # Hiperparâmetros:
    #  hidden_layer_sizes=(100,) usando 100 neurônios (perceptrons) na camada oculta única
    #  max_iter=300 = máximo de iterações (épocas) para o algoritmo de otimização durante o treinamento.
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp_classifier.fit(X_train, y_train)
    mlp_y_pred = mlp_classifier.predict(X_test)

    # Avaliação
    print("\nResultados pela Rede Neural MLPClassifier:")
    print(confusion_matrix(y_test, mlp_y_pred))
    print(classification_report(y_test, mlp_y_pred))
    print("Accuracy:", accuracy_score(y_test, mlp_y_pred))


#
# Treinamento e análise com uma rede KBANN (Knowledge-Based Artificial Neural Network),
# ou Rede Neural Artificial Baseada em Conhecimento
# Definindo/criando o modelo KBANN simples usando Keras
def modelo_kbann(input_dim):
    # Hiperparâmetros:
    #  add(Dense(128, activation='relu') -> Define 128 neurônios totalmente conectados, usando a função de ativação 'relu'
    #  Rectified Linear Unit (unidade linear retificada) - Camadas Dense, são hidden layers, camadas internas da rede neural.
    #  add(Dropout(0.5) -> Camada de dropout, que desativa aleatoriamente 50% dos neurônios durante o treinamento.
    #  add(Dense(64, activation='relu') -> Cria mais uma camada densa, com 64 neurônios e função de ativação ReLU.
    #  add(Dense(32, activation='relu') -> Cria mais uma camada densa, com 32 neurônios e função de ativação ReLU.
    #  add(Dense(len(np.unique(y_encoded)), activation='softmax') -> Camada de saída com número de neurônios igual
    #  ao número de classes únicas no conjunto de dados de saída (y_encoded).
    #  activation='softmax' -> A função de ativação softmax, que transforma as saídas em probabilidades que somam a 1,
    #  importante para tarefas de classificação multi-classe, como no problema em análise.
    #  loss='sparse_categorical_crossentropy' -> Função de perda (sparse_categorical_crossentropy) usada para problemas de
    #  classificação com classes de valores inteiros. Para saídas não codificadas one-hot.
    #  optimizer='adam' -> Adam é um algoritmo de otimização eficiente que ajusta as taxas de aprendizado durante o treinamento.
    #  metrics=['accuracy'] -> métrica a ser monitorada durante o treinamento e a avaliação, neste caso, a acurácia do modelo.
    model = Sequential()  # empilha camadas de forma linear, onde a saída de uma camada é a entrada da próxima.
    model.add(Input(shape=(input_dim,)))  # Camada de entrada usando Input
    #model.add(Dense(256, activation='relu'))  #, kernel_regularizer=l2(0.01)))  # Camada oculta
    #model.add(Dropout(0.5))  # Dropout para reduzir overfitting, é uma das técnicas de regularização para combater o overfitting.
    #model.add(Dense(128, activation='relu'))  # , kernel_regularizer=l2(0.01)))  # Outra camada oculta
    #model.add(Dropout(0.5))  # Dropout
    model.add(Dense(64, activation='relu'))  # , kernel_regularizer=l2(0.01)))  # Outra camada oculta
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(32, activation='relu')) # , kernel_regularizer=l2(0.01)))  # Outra camada oculta
    model.add(Dropout(0.5))  # Dropout
    #model.add(Dense(32, activation='relu'))  # Outra camada oculta
    #model.add(Dropout(0.5))  # Dropout
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))  # Camada de saída
    adam_param = Adam(learning_rate=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_param, metrics=['accuracy'])

    return model


# Executar o treinamento e aprendizado do modelo
def kbann_exe():
    # Hiperparâmetros:
    #  X_train e y_train -> Dados de entrada (X_train) e rótulos correspondentes (y_train) para treinamento.
    #  epochs=50: O número de iterações completas através do conjunto de dados de treinamento. Cada época permite que o
    #  modelo aprenda mais sobre os dados.
    #  batch_size=64: Número de amostras usadas antes de atualizar os pesos no modelo. Um tamanho de lote menor evita
    #  sobrecarga da memória, mas pode tornar o treinamento mais ruidoso.
    #  verbose=1: Define nível de detalhes durante o treinamento. O valor de 1 mostra a barra de progresso.

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    folds = 20  # Nro. vezes para a validação cruzada
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    accuracies = []
    error_rates = {0: [], 1: [], 2: []}  # Dic. para armazenar erros por rótulos IE, EI e NEITHER
    total_conf_matrix = np.zeros((len(np.unique(y_encoded)), len(np.unique(y_encoded))))  # Para a matriz de confusão final

    # Taxas de erros por rótulos IE, EI e NEITHER
    labels = {0: 'EI', 1: 'IE', 2: 'NEITHER'}  # Assumindo que 0, 1, e 2 são os rótulos das classes

    for train_index, test_index in kfold.split(X):

        # Dividindo os dados em treino e teste
        X_train, X_test = X_encoded[train_index], X_encoded[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Criar e treinar o modelo
        kbann_model = modelo_kbann(X_train.shape[1])
        kbann_model.fit(X_train, y_train, epochs=25, batch_size=4, verbose=0,
                        validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Avaliação
        y_pred = np.argmax(kbann_model.predict(X_test), axis=-1)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        conf_matrix = confusion_matrix(y_test, y_pred)
        total_conf_matrix += conf_matrix

        for label_index in range(len(labels)):
            true_positives = conf_matrix[label_index, label_index]  # Verdadeiros positivos
            false_negatives = conf_matrix[label_index, :].sum() - true_positives  # Falsos negativos
            false_positives = conf_matrix[:, label_index].sum() - true_positives  # Falsos positivos

            # Cálculo
            total = true_positives + false_negatives + false_positives
            if total > 0:
                error_rate = (false_positives + false_negatives) / total
                error_rates[label_index].append(error_rate)  # Armazena a taxa de erro
            else:
                error_rates[label_index].append(None)  # Para evitar divisão por zero se total for 0

        # Relatório de classificação
        print("\nResultados do KBANN:")
        print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    print(f"Acurácias em cada fold: {accuracies}")
    print(f"Acurácia média: {np.mean(accuracies):.2f}")
    print(f"Desvio padrão da acurácia: {np.std(accuracies):.2f}")

    # Cálculos e impressão das taxas de erro finais por rótulo
    for label_index, label in labels.items():
        if len(error_rates[label_index]) > 0:
            avg_error_rate = np.mean(error_rates[label_index])
            std_error_rate = np.std(error_rates[label_index])
            print(f'Taxa de erro média para {label}: {avg_error_rate:.2%} (Desvio Padrão: {std_error_rate:.2%})')
        else:
            print(f'Taxa de erro para {label}: Não disponível')

    # Impressão da matriz de confusão total
    print("\nMatriz de Confusão Total:\n", total_conf_matrix)


kbann_exe()