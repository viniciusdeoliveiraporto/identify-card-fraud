from utils.test_utils import get_data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def model_metrics(model, count):
    data_test = get_data(count)
    rows = data_test["fraud"] + data_test["legit"]

    y_true = []
    y_pred = []

    for row in rows:
        expected = 1 if int(row[-1]) == 1 else 0
        predicted = 1 if model.predict(row) else 0
        y_true.append(expected)
        y_pred.append(predicted)


    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    metrics = [precision, recall, f1]
    labels = ['Precisão', 'Recall', 'F1-score']

    plt.bar(labels, metrics, color=['blue', 'orange', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Valor')
    plt.title('Métricas do Modelo')
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.show()