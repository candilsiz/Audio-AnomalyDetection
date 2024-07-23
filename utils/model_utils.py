import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score

def model_evaluate(y_pred, y_test, optimal_threshold=None):
    optimal_accuracy = accuracy_score(y_test, y_pred)
    optimal_precision = precision_score(y_test, y_pred)
    optimal_recall = recall_score(y_test, y_pred)
    optimal_f1 = f1_score(y_test, y_pred)
    if optimal_threshold:
        print(f"Optimal Threshold: {optimal_threshold}\n")
    print(f"Accuracy: {optimal_accuracy}")
    print(f"Precision: {optimal_precision}")
    print(f"Recall: {optimal_recall}")
    print(f"F1 Score: {optimal_f1}")

def get_anomaly_treshold(score, y_test):
    precisions, recalls, thresholds = precision_recall_curve(y_test, score) 
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def reduce_dimensionality(method, n_comp):
    pass

def model_hyperparameter_tuner(model, **params):
    pass