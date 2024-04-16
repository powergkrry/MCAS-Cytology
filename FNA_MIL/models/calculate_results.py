import os
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


fig, axs = plt.subplots(nrows=1, ncols=1)
to_loads = ['9images_mil9_mobilenet_testfold0', '9images_mil9_mobilenet_testfold1', '9images_mil9_mobilenet_testfold2', '9images_mil9_mobilenet_testfold3']
for i, to_load in enumerate(to_loads):
    folder_path = os.path.join('/media/data1/kanghyun/FNA_MIL/models', to_load, 'test_inference.json')
    with open(folder_path) as json_file:
        data = json.load(json_file)
    
    y_pred = []
    y_true = []
    for value in data.values():
        y_pred += [value['predictions']]
        y_true += [value['label']]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    axs.plot(fpr, tpr, label=f'Fold {i+1} AUC = {auc:.3f}')
axs.set_title(f"ROC Curves - 4 Folds", fontsize=20)
axs.set_xlabel("False Positive Rate", fontsize=15)
axs.set_ylabel("True Positive Rate", fontsize=15)
axs.tick_params(axis='both', which='major', labelsize=12)
axs.legend(loc="lower right", fontsize=15)
fig.savefig('ROC_curves')

# y_pred[y_pred>=0.5] = 1
# y_pred[y_pred<0.5] = 0
# print('precision: ', precision_score(y_true, y_pred))
# print('recall: ', recall_score(y_true, y_pred))