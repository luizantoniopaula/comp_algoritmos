import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
import torch.nn.functional as F
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#
# Carregar o conjunto de dados
molecular_biology_splice_junction_gene_sequences = fetch_ucirepo(id=69)
X = molecular_biology_splice_junction_gene_sequences.data.features
y = molecular_biology_splice_junction_gene_sequences.data.targets

# Garanta que X e y são arrays NumPy
# if isinstance(X, pd.DataFrame):
#     X = X.values  # Converte para array NumPy se necessário
#
# if isinstance(y, (pd.DataFrame, pd.Series)):
#     y = y.values  # Converte para array NumPy se necessário

# Se X for um DataFrame, converta para apenas colunas numéricas
if isinstance(X, pd.DataFrame):
    X = X.values # select_dtypes(include=[np.number]).values  # Pega apenas colunas numéricas

# Certifique-se de que y é um array unidimensional
if isinstance(y, (pd.DataFrame, pd.Series)):
    y = y.values.ravel()  # Converte para array 1D


# Codificação One-hot para sequências de DNA
def encode_sequences(X):

    # Achatar (flatten) as sequências em um array simples
    sequences = np.array(X)
    flat_sequences = sequences.reshape(-1)  # flatten() # reshape(-1)

    # One-hot encode the characters
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    one_hot_encoded = encoder.fit_transform(flat_sequences[:, None]) # .toarray()
    return one_hot_encoded.reshape(sequences.shape[0], -1)


X_encoded = encode_sequences(X)

x_tensor = torch.tensor(X_encoded, dtype=torch.float32)

# Codificação das classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # .to_numpy().ravel()) # Problemas na função ravel() do numpy vindo de dataframe Pandas



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
        self.prior_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, prior_knowledge=None):
        # Converter para tensor do PyTorch
        x_tensor = torch.tensor(X_encoded, dtype=torch.float32)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x_tensor))
            x = self.dropout(x_tensor)

        x = self.layers[-1](x)

        if prior_knowledge is not None:
            x = x + self.prior_scale * prior_knowledge

        return x


class KABNNOptimizer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.scaler = StandardScaler()

    def preprocess_data(self, X, y):
        """Preprocessar dados com normalização"""
        X_scaled = self.scaler.fit_transform(X)
        return torch.FloatTensor(X_scaled), torch.FloatTensor(y)

    def remove_outliers(self, X, y, threshold=3):
        """Remover outliers usando z-score"""
        # Verifique a forma de X_encoded
        print("X_encoded:", X_encoded)
        z_scores = np.abs((X_encoded - np.mean(X_encoded, axis=0)) / np.std(X_encoded, axis=0))
        # z_scores = np.abs((X - X.mean()) / X.std())
        mask = (z_scores < threshold).all(axis=1)
        if np.sum(mask) == 0:
            raise ValueError("Todos os dados foram considerados outliers e removidos.")
        return X_encoded[mask], y[mask]

    def grid_search(self, X, y, hidden_sizes_list, learning_rates, dropout_rates):
        """Grid search para otimização de hiperparâmetros"""
        best_score = float('-inf')
        best_params = None

        for hidden_sizes in hidden_sizes_list:
            for lr in learning_rates:
                for dropout in dropout_rates:
                    score = self._evaluate_model(X, y, hidden_sizes, lr, dropout)

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

        print("Forma de X antes da divisão:", X.shape)

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

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = F.mse_loss(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Validação
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = F.mse_loss(val_outputs, y_val)
                scores.append(-val_loss.item())  # Negative MSE as score

        return np.mean(scores)

    def train_best_model(self, X, y, best_params, epochs=200):
        """Treinar modelo final com os melhores parâmetros"""
        X_processed, y_processed = self.preprocess_data(X, y)

        model = KABNN(self.input_size, best_params['hidden_sizes'],
                      self.output_size, best_params['dropout_rate'])
        optimizer = Adam(model.parameters(), lr=best_params['learning_rate'])

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_processed)
            loss = F.mse_loss(outputs, y_processed)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        return model


# Exemplo de uso
def optimize_kabnn(X, y):
    optimizer = KABNNOptimizer(input_size=X.shape[1], output_size=len(np.unique(y)))

    # Remover outliers
    #  X_clean, y_clean = optimizer.remove_outliers(X, y)

    # Definir grades de hiperparâmetros
    hidden_sizes_list = [
        [64, 32],
        [128, 64, 32],
        [256, 128, 64]
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

optimize_kabnn(X,y)