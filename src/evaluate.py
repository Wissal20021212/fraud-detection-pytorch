
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

from model import FraudDetectionModel
from prepare_data import load_data


# 1. Charger les données
X_train, X_val, X_test, y_train, y_val, y_test = load_data()


# 2. Charger le modèle entraîné
model = FraudDetectionModel()

model.load_state_dict(
    torch.load("fraud_model.pth", map_location="cpu", weights_only=True)
)

model.eval()


# 3. Faire les prédictions
with torch.no_grad():
    logits = model(X_test)

    probabilities = torch.sigmoid(logits)


# 4. Transformer les probabilités en classes
threshold = 0.5

y_pred = (probabilities >= threshold).int().squeeze(1)

y_true = y_test.int()


# 5. Calculer les métriques
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

pr_auc = average_precision_score(
    y_true,
    probabilities.squeeze(1)
)


# 6. Afficher les résultats
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("PR-AUC:", pr_auc)
