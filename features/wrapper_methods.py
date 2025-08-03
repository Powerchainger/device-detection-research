from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import os
from pathlib import Path
import pandas as pd
from utils.feature_utils import create_data, read_data, create_dir, train_classifier
from config.config import CHECKPOINT_DIR, CASES


save_path = create_dir(str(CHECKPOINT_DIR) + "\\wrapper")

# These can be used as hyperparameters as well
folds=10
##############################################
# THESE ARE FOR STORING THE CV RESULTS FOR EVERY STEP OF THE FEATURE SELECTION PROCESS
# BWE = from n_features to 1
# FSE = from 1 to n_features
# RFE = from n_features to 1
# CV results are stored and then the feature group with the best results
# is selected.
num_of_features = 5
#############################################

classifiers = {
    "SVM": SVC(kernel="linear", class_weight='balanced'),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced')
}

results = {
    "Forward_Selection": {model: None for model in classifiers.keys()},
    "Backward_Elimination": {model: None for model in classifiers.keys()},
    "RFE": {model: None for model in classifiers.keys()},
    "selected_features": {model: {"Forward_Selection": None,
                                  "Backward_Elimination": None, 
                                  "RFE": None} for model in classifiers.keys()}
}
probs_path = str(CHECKPOINT_DIR) + f"\\valid_probs"
def run_wrapper_methods(plot_cv=False):
    for case in CASES:
        data_file = case + "_valid_probs.json" 
        data_tuple = read_data(probs_path, data_file)
        X, y = create_data(data_tuple, filter=True)
        n_features = X.shape[1]
        data_tuple = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        scorer = make_scorer(f1_score, average="macro")
        for model_name, classifier in classifiers.items():
            wrappers = {
            "Forward_Selection": SequentialFeatureSelector(classifier,
                                                            n_features_to_select=num_of_features,
                                                            direction="forward",
                                                            scoring=scorer,                                                         
                                                            cv=StratifiedKFold(folds, shuffle=True, random_state=0)),
            "Backward_Elimination": SequentialFeatureSelector(classifier,
                                                                n_features_to_select=num_of_features,
                                                                direction="backward",
                                                                scoring=scorer,
                                                                cv=StratifiedKFold(folds, shuffle=True, random_state=0)),
            "RFE": RFECV(classifier, 
                            min_features_to_select=num_of_features,
                            scoring=scorer,
                            cv=StratifiedKFold(folds, shuffle=True, random_state=0))
        }
            for wr_name, wr_object, in wrappers.items():
                print(f"New loop starting")
                score, features = train_classifier(data_tuple, wr_object, classifier=classifier)
                print(f"Case: {case}, Model: {model_name}, Method: {wr_name}")
                # if plot_cv:
                    # if hasattr(wr_object, "cv_results_"):
                    #     cv_results = pd.DataFrame(fit_wrapper.cv_results_)
                    # else:
                    #     cv_results = pd.DataFrame.from_dict(fit_wrapper.scores, orient='index',
                    #                                         columns=['fold_mean', 'fold_std', 'feature_mask'])
                    # cv_path = create_dir(save_path + '\\cv_results')
                    # cv_results.to_csv(os.path.join(cv_path, f"cv_results_{wr_name}_{model_name}_{case}.csv"), index=True)
                    # TODO: check plot code from notebook to create a plot_cv() function to call it here or in the main.py
                results[wr_name][model_name] = score
                results["selected_features"][model_name][wr_name] = features
        f1_scores = dict(list(results.items())[:3])
        f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient='index')
        f1_scores_df.to_csv(os.path.join(save_path, f"f1_scores_{case}.csv"), index=True)   
        wrapper_model_df = pd.DataFrame({model: {method: ", ".join(features) for method, features in methods.items()}
                    for model, methods in results["selected_features"].items()})
        wrapper_model_df.to_csv(os.path.join(save_path, f"selected_features_{case}.csv"), index=True)    