import matplotlib.pyplot as plt
from src.split_dataset import split_train_test
from sklearn.metrics import f1_score, recall_score


def model_metrics(model):
    _, ds_test, labels_test = split_train_test()
    
    y_true = labels_test
    y_pred = []
    
    for x in ds_test:
        pred = 1 if model.predict(x.tolist()) else 0
        y_pred.append(pred)

    # Calcula métricas
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Plota gráfico
    metrics = [recall, f1]
    labels = ['Recall', 'F1-score']
    
    plt.bar(labels, metrics, color=['orange', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Valor')
    plt.title('Métricas do Modelo')
    
    # Mostra valores acima das barras
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.show()
    
    return recall, f1
