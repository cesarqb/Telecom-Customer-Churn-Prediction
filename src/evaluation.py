import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ==========================================
# MATRIZ DE CONFUSIÓN
# ==========================================

def obtener_matriz_confusion(
    y_true,
    y_pred
):
    """
    Calcula la matriz de confusión.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


# ==========================================
# GRAFICAR MATRIZ DE CONFUSIÓN
# ==========================================

def confusion_matrix_graph(cm):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No', 'Yes'],
        yticklabels=['No', 'Yes']
    )

    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')

    plt.show()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

def model_classification_report(
    y_true,
    y_pred
):
    """
    Muestra Accuracy, Precision,
    Recall y F1 Score.
    """

    print(
        classification_report(
            y_true,
            y_pred
        )
    )


# ==========================================
# CURVA ROC
# ==========================================

def roc_curve_graph(
    y,
    prob
):

    y_prob = prob[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f'AUC = {roc_auc:.2f}'
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    plt.title('Curva ROC')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid()

    plt.show()


# ==========================================
# ROC AUC SCORE
# ==========================================

def calcular_auc(
    y_true,
    prob
):
    """
    Calcula el ROC AUC Score.
    """

    y_prob = prob[:, 1]

    return metrics.roc_auc_score(
        y_true,
        y_prob
    )