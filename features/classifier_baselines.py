from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.svm import SVC

import os
import pandas as pd

from config.config import CHECKPOINT_DIR, CASES
from utils.feature_utils import create_data, read_data, create_dir

probs_path = str(CHECKPOINT_DIR) + f"\\valid_probs"
save_path = create_dir(str(CHECKPOINT_DIR) + "\\classifier_baselines")

def run_classifier_baselines():
    for case in CASES:
        filename = case + "_valid_probs.json"
        data = read_data(probs_path, filename)
        # Sorted Probabilities and labels
        valid_probs, y = create_data(data, filter=False)
        # Feature vector of statistical properties and labels
        valid_probs_filter = create_data(data, filter=True)[0]
        # Add class_weight=compute_class_weight() instead of  class_weight='balanced'
        # e.g. 
        # class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        # class_weight_dict = dict(zip(classes, class_weights))
        # SVC(kernel="linear", class_weight=class_weight_dict))
        classifiers = {
            "SVM": SVC(kernel="linear", class_weight='balanced'),
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }
        data_probs = train_test_split(valid_probs, y, test_size=0.2, random_state=0, stratify=y)
        data_features = train_test_split(valid_probs_filter, y, test_size=0.2, random_state=0, stratify=y)
        results_probs = {}
        results_features = {}
        for model_name, classifier in classifiers.items():
            clf1 = clone(classifier)
            clf1.fit(data_probs[0], data_probs[2])
            clf2 = clone(classifier)
            clf2.fit(data_features[0], data_features[2])
            preds1 = clf1.predict(data_probs[1])
            preds2 = clf2.predict(data_features[1])
            f1_score_probs = f1_score(data_probs[3], preds1, average='macro')
            accuracy_score_probs = accuracy_score(data_probs[3], preds1)
            f1_score_features = f1_score(data_features[3], preds2, average='macro')
            accuracy_score_features = accuracy_score(data_features[3], preds2)
            
            results_probs[model_name] = [accuracy_score_probs, f1_score_probs]
            results_features[model_name] = [accuracy_score_features, f1_score_features]
        df_probs = pd.DataFrame.from_dict(results_probs, orient="index", columns=["accuracy", "f1-macro"])
        df_features = pd.DataFrame.from_dict(results_features, orient="index", columns=["accuracy", "f1-macro"])
        df_probs.to_csv(os.path.join(save_path, f"{case}_probs.csv"), index=True)
        df_features.to_csv(os.path.join(save_path, f"{case}_features.csv"), index=True)
