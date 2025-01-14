import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, labels, algo):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels.values(), yticklabels=labels.values())
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - ' + algo)
    plt.show()


def plot_accuracies_and_errors(accuracy_mean, accuracy_std, avg_error_rates, labels, algo):
    categories = list(labels.values())
    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))

    # Gráfico de Acurácias
    plt.bar(x - 0.2, [accuracy_mean * 100] * len(categories), width=0.4, label='Acurácia Média (%)', color='blue',
            alpha=0.6)

    # Gráfico de Erros
    plt.bar(x + 0.2, [rate * 100 if rate is not None else 0 for rate in avg_error_rates], width=0.4,
            label='Erro Médio (%)', color='red', alpha=0.6)

    plt.xticks(x, categories)
    plt.ylabel('Percentagem (%)')
    plt.title('Acurácia e Erro por Categoria - Algoritmo ' + algo)
    plt.legend()
    plt.ylim(0, 100)

    plt.show()

# Exemplo de chamada da função, passando seus dados X_encoded, y_encoded e labels
# labels = {0: 'EI', 1: 'IA', 2: 'N'}  # Exemplo de mapeamento, ajuste conforme seu caso
# random_forest(X_encoded, y_encoded, labels)
