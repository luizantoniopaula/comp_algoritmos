import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.optim import Adam
import torch.nn.functional as F
from ucimlrepo import fetch_ucirepo


class KABNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.5):
        super(KABNN, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Definindo dicionário de mapeamento
        encoding_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'D': 5, 'R': 6, 'S': 7}
        encoded_sequences = []
        for sequence in x:
            encoded_sequence = [encoding_dict[char] for char in sequence]  # Converte cada letra para seu valor numérico
            encoded_sequences.append(encoded_sequence)
        x_encoded = np.array(encoded_sequences)

        x_tensor = torch.from_numpy(x_encoded).float()

        #print("Forma de x_tensor:", x_tensor.shape)  # Deve ser (n_samples, input_size)
        #print("Dados de x_tensor:", x_tensor)
        # pd.DataFrame(x).to_csv('x_original.csv', sep=";", index=False, header=False)  # Salva X
        # pd.DataFrame(x_encoded).to_csv('x_encoded.csv', sep=';', index=False, header=False)  # Salva X_encoded

        for layer in self.layers[:-1]:
            x = F.relu(layer(x_tensor))
            x = self.dropout(x_tensor)

        x = self.layers[-1](x_tensor)
        return x


class KABNNOptimizer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = StandardScaler()

    # Codificação One-hot para sequências de DNA
    def encode_sequences(X):

        # Achatar (flatten) as sequências em um array simples
        sequences = np.array(X)
        flat_sequences = sequences.reshape(-1)  # flatten() # reshape(-1)

        # One-hot encode the characters
        encoder = OneHotEncoder(categories='auto', sparse_output=False)
        one_hot_encoded = encoder.fit_transform(flat_sequences[:, None])  # .toarray()
        return one_hot_encoded.reshape(sequences.shape[0], -1)

    def preprocess_data(self, X, y):
        """Preprocessar dados com normalização"""
        X_scaled = self.scaler.fit_transform(X)
        return torch.FloatTensor(X_scaled), torch.FloatTensor(y)

    def remove_outliers(self, X, y, threshold=3):
        """Remover outliers usando z-score"""
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = (z_scores < threshold).all(axis=1)
        if np.sum(mask) == 0:
            raise ValueError("Todos os dados foram considerados outliers e removidos.")
        return X[mask], y[mask]

    def grid_search(self, X, y, hidden_sizes_list, learning_rates, dropout_rates):
        """Grid search para otimização de hiperparâmetros"""
        best_score = float('-inf')
        best_params = None

        for hidden_sizes in hidden_sizes_list:
            for lr in learning_rates:
                for dropout in dropout_rates:
                    score, _ = self._evaluate_model(X, y, hidden_sizes, lr, dropout)

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'hidden_sizes': hidden_sizes,
                            'learning_rate': lr,
                            'dropout_rate': dropout
                        }

        return best_params, best_score

    def _evaluate_model(self, X, y, hidden_sizes, learning_rate, dropout_rate,
                        n_folds=5, epochs=100, batch_size=32):
        """Avaliar modelo usando k-fold cross validation"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        total_conf_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))))  # Para a matriz de confusão total

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = KABNN(self.input_size, hidden_sizes, self.output_size, dropout_rate)
            optimizer = Adam(model.parameters(), lr=learning_rate)

            # Treinamento
            for epoch in range(epochs):
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i + batch_size]
                    batch_y = y_train[i:i + batch_size]
                    label_encoder = LabelEncoder()
                    batch_y_enc = label_encoder.fit_transform(batch_y)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = F.cross_entropy(outputs, torch.from_numpy(batch_y_enc))  # Função de perda para classificação
                    loss.backward()
                    optimizer.step()

            # Validação
            with torch.no_grad():
                label_encoder = LabelEncoder()
                y_val_enc = label_encoder.fit_transform(y_val)
                val_outputs = model(X_val)
                val_loss = F.cross_entropy(val_outputs, torch.from_numpy(y_val_enc))
                scores.append(-val_loss.item())  # Negative loss as score for minimization

                # Acumulando a matriz de confusão
                y_pred = torch.argmax(val_outputs, dim=1)
                conf_matrix = confusion_matrix(y_val, y_pred.numpy(), labels=np.unique(y))
                total_conf_matrix += conf_matrix

                # Calcule e imprima as métricas para este fold
            average_score = np.mean(scores)
            print(f"Acurácia média neste fold: {-average_score:.4f}")

            return average_score, total_conf_matrix

    def train_best_model(self, X, y, best_params, epochs=200):
        """Treinar modelo final com os melhores parâmetros"""
        X_processed, y_processed = self.preprocess_data(X, y)

        model = KABNN(self.input_size, best_params['hidden_sizes'],
                      self.output_size, best_params['dropout_rate'])
        optimizer = Adam(model.parameters(), lr=best_params['learning_rate'])

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_processed)
            loss = F.cross_entropy(outputs, y_processed.long())  # Usando função de perda para classificação

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        return model

    # Exemplo de uso
    def optimize_kabnn(X, y):
        optimizer = KABNNOptimizer(input_size=X.shape[1], output_size=len(np.unique(y)))

        # Remover outliers
        #X_clean, y_clean = optimizer.remove_outliers(X, y)

        # Definir grades de hiperparâmetros
        hidden_sizes_list = [
            [60, 60],
            [60, 60, 60],
            [60, 60, 60]
        ]
        learning_rates = [0.001, 0.01, 0.1]
        dropout_rates = [0.3, 0.5, 0.7]

        # Encontrar melhores hiperparâmetros
        best_params, best_score = optimizer.grid_search(
            X, y,  # X_clean, y_clean,
            hidden_sizes_list,
            learning_rates,
            dropout_rates
        )

        print("Melhores parâmetros:", best_params)
        print("Melhor score:", best_score)

        # Treinar modelo final
        final_model = optimizer.train_best_model(X, y, best_params)

        return final_model, best_params

# Execução do processo de otimização
molecular_biology_splice_junction_gene_sequences = fetch_ucirepo(id=69)
X = molecular_biology_splice_junction_gene_sequences.data.features
y = molecular_biology_splice_junction_gene_sequences.data.targets

pd.DataFrame(X).to_csv('x_original.csv', sep=";", index=False, header=False)  # Salva X

# Verifique e converta para arrays Numpy conforme necessário
if isinstance(X, pd.DataFrame):
    X = X.values
if isinstance(y, (pd.DataFrame, pd.Series)):
    y = y.values.ravel()  # Converte para um vetor unidimensional

# Inicia o processo de otimização
final_model, best_params = KABNNOptimizer.optimize_kabnn(X, y)