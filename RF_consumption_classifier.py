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
files = sys.argv[1]
df = pd.read_pickle(files)
df = df.dropna()

# separate features and labels
X_train = df.drop(columns=['chemical', 'taxon', 'label'])  # Features
y_train = df['label']  # Target label

# Initialize rf classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None,min_samples_leaf=1, min_samples_split=5)


rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'Trained_model/consumption_classifier.pkl')

