from config.config import CASES, CHECKPOINT_DIR
from utils.feature_utils import read_data, create_data, select_features, create_dir

import os

import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

save_path = create_dir(str(CHECKPOINT_DIR) + "\\filter")
probs_path = str(CHECKPOINT_DIR) + f"\\valid_probs"
classifiers = {
            "SVM": SVC(kernel="linear", class_weight='balanced'),
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced')
}

def run_filter_methods():    
    for case in CASES:
        data_file = case + "_valid_probs.json"
        data_tuple = read_data(probs_path, data_file)
        data, labels = create_data(data_tuple, filter=True)
        filter_dict = select_features(data, labels)
        case_results = {}
        features_results = {}
        for method_name, feature_list in filter_dict.items():
            X_selected = data[feature_list]  # Subset the features
            X_train, X_test, y_train, y_test = train_test_split(X_selected, labels, test_size=0.2, random_state=42)
            clf_results = {}
            for clf_name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='macro')
                clf_results[clf_name] = f1
            case_results[method_name] = clf_results
            features_results[method_name] = ', '.join(feature_list)
         
        # Convert case_results to DataFrame and save
        scores_df = pd.DataFrame.from_dict(case_results, orient='index')
        csv_filename = os.path.join(save_path, f"f1_score_{case}.csv")
        scores_df.to_csv(csv_filename)
        
        features_df = pd.DataFrame.from_dict(features_results, orient='index', columns=['Selected_Features'])
        features_csv_filename = os.path.join(save_path, f"selected_features_{case}.csv")
        features_df.to_csv(features_csv_filename)