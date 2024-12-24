import pandas as pd
import numpy as np
import joblib
from numpy import *
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import statistics
import os, sys


# Training Process
# replace BrendaSubstrateOrProductfiles.csv with the input file for brenda models. Please refer to ./Data/Example files
df = pd.read_csv('Data/BrendaSubstrateOrProductfiles.csv')
num_fold = 5
skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)
fold_cnt = 0
accuracy_all = []
auc_all = []
confusion_matrix_all = []
precision_all = []
recall_all = []
f1_all = []
balanced_accuracy_all = []
FPR_all = []
mcc_all = []
for train_index, test_index in skf.split(df, df['label']):
    print(f'fold {fold_cnt + 1} start:')
    train_set = df.iloc[train_index, :]
    test_set = df.iloc[test_index, :]
    rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=30,
                                           min_samples_leaf=4, min_samples_split=5)
    rf_classifier.fit(train_set.drop(columns=['substrate', 'EC', 'label']), train_set['label'])

    joblib.dump(rf_classifier, f'Trained_model/substrate_ec_classifier_trained_model_core2core_fold_{fold_cnt+1}.pkl')

    # # load the trained model
    # rf_classifier = joblib.load(f'Trained_model/substrate_ec_classifier_trained_model_fold_{fold_cnt}.pkl')

    # Evaluate the Model
    X_test = test_set.drop(columns=['substrate', 'EC', 'label'])
    y_test = test_set['label']

    # Evaluate the Model
    y_pred = rf_classifier.predict(X_test)

    test_acc = metrics.accuracy_score(y_test, y_pred)

    y_score = rf_classifier.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    confusionmatrix = metrics.confusion_matrix(y_test, y_pred)
    p_score = metrics.precision_score(y_test, y_pred)
    r_score = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)

    TP = confusionmatrix[1][1]
    TN = confusionmatrix[0][0]
    FP = confusionmatrix[0][1]
    FN = confusionmatrix[1][0]
    fpr_fromCM = FP / (FP + TN)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    accuracy_all.append(test_acc)
    auc_all.append(roc_auc)
    confusion_matrix_all.append(confusionmatrix)
    precision_all.append(p_score)
    recall_all.append(r_score)
    f1_all.append(f1)
    balanced_accuracy_all.append(bal_acc)
    FPR_all.append(fpr_fromCM)
    mcc_all.append(mcc)

    print('accuracy', test_acc)
    print('auc', roc_auc)
    print('confusion matrix', confusionmatrix)
    print('precision', p_score)
    print('recall', r_score)
    print('f1', f1)
    print('balanced accuracy', bal_acc)
    print('FPR', fpr_fromCM)
    print('mcc', mcc)

    fold_cnt += 1

test_acc_mean = statistics.mean(accuracy_all)
auc_mean = statistics.mean(auc_all)
CM_all = np.sum(confusion_matrix_all, axis=0)
precision_mean = statistics.mean(precision_all)
recall_mean = statistics.mean(recall_all)
f1_mean = statistics.mean(f1_all)
bac_mean = statistics.mean(balanced_accuracy_all)
fpr_mean = statistics.mean(FPR_all)
mcc_mean = statistics.mean(mcc_all)
print('########## overall accuracy #############')
print('accuracy', test_acc_mean)
print('auc', auc_mean)
print('confusion matrix', CM_all)
print('precision', precision_mean)
print('recall', recall_mean)
print('f1', f1_mean)
print('balanced accuracy', bac_mean)
print('FPR', fpr_mean)
print('mcc', mcc_mean)

