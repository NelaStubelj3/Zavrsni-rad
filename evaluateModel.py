import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Definirajte funkcije za evaluaciju
def evaluate_classification_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            track_targets = targets[:, :1]
            resman_targets = targets[:, 1:3]
            sysmon_targets = targets[:, 3:]

            track_output, resman_output, sysmon_output = model(inputs)

            # Task 1: Track
            track_predictions = (track_output > 0.5).float()  # Binary classification
            all_predictions.extend(track_predictions.numpy())
            all_targets.extend(track_targets.numpy())

            # Task 2: Resman
            resman_predictions = torch.argmax(resman_output, dim=1)
            all_predictions.extend(resman_predictions.numpy())
            all_targets.extend(resman_targets.numpy())

            # Task 3: Sysmon
            sysmon_predictions = torch.argmax(sysmon_output, dim=1)
            all_predictions.extend(sysmon_predictions.numpy())
            all_targets.extend(sysmon_targets.numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Evaluacija za Task 1: Track
    track_accuracy = accuracy_score(all_targets[:, 0], all_predictions[:, 0])
    track_conf_matrix = confusion_matrix(all_targets[:, 0], all_predictions[:, 0])
    track_f1 = f1_score(all_targets[:, 0], all_predictions[:, 0])

    # Evaluacija za Task 2: Resman
    resman_accuracy = accuracy_score(all_targets[:, 1:3].flatten(), all_predictions[:, 1:3].flatten())
    resman_conf_matrix = confusion_matrix(all_targets[:, 1:3].flatten(), all_predictions[:, 1:3].flatten())
    resman_f1 = f1_score(all_targets[:, 1:3].flatten(), all_predictions[:, 1:3].flatten(), average='weighted')

    # Evaluacija za Task 3: Sysmon
    sysmon_auc = roc_auc_score(pd.get_dummies(all_targets[:, 3:]), pd.get_dummies(all_predictions[:, 3:]), average='macro')
    sysmon_precision, sysmon_recall, _ = precision_recall_curve(pd.get_dummies(all_targets[:, 3:]), pd.get_dummies(all_predictions[:, 3:]))
    sysmon_auc_pr = auc(sysmon_recall, sysmon_precision)

    return track_accuracy, track_conf_matrix, track_f1, resman_accuracy, resman_conf_matrix, resman_f1, sysmon_auc, sysmon_auc_pr

# Evaluacija na testnom skupu
track_accuracy, track_conf_matrix, track_f1, resman_accuracy, resman_conf_matrix, resman_f1, sysmon_auc, sysmon_auc_pr = evaluate_classification_model(model, combined_loader)

# Ispis rezultata
print("Track Accuracy:", track_accuracy)
print("Track Confusion Matrix:")
print(track_conf_matrix)
print("Track F1 Score:", track_f1)

print("Resman Accuracy:", resman_accuracy)
print("Resman Confusion Matrix:")
print(resman_conf_matrix)
print("Resman F1 Score:", resman_f1)

print("Sysmon AUC-ROC Score:", sysmon_auc)
print("Sysmon AUC-PR Score:", sysmon_auc_pr)
