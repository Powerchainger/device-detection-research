from tqdm import tqdm
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys, os
from pathlib import Path
import pandas as pd

from experiments.utils import create_data, train_classifier, create_dir


root = str(Path(os.getcwd()).resolve().parents[1])
path = root + '\\extracted_probabilities_seed'
save_path = create_dir(root + '\\data\\result_files')
case = sys.argv[1]
seeds = range(0,3)
#These can be used as hyperparameters as well
folds=10
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
    "selected_features": {model: {"Forward_Selection": [],
                                  "Backward_Elimination": [], 
                                  "RFE": []} for model in classifiers.keys()}
}

if __name__ == '__main__':
    for seed in tqdm(seeds, desc="Seed loop"):
        file = case + '_validation_seed_' + str(seed) + '.json'
        X, y = create_data(path, file) 
        for model_name, classifier in tqdm(classifiers.items(), desc="Model loop"):
            wrappers = {
            "Forward_Selection": SequentialFeatureSelector(classifier, n_features_to_select=num_of_features,
                                                        direction="forward", cv=folds),
            "Backward_Elimination": SequentialFeatureSelector(classifier, n_features_to_select=num_of_features,
                                                            direction="backward", cv=folds),
            "RFE": RFECV(classifier, min_features_to_select=num_of_features, cv=folds)
        }
            data_tuple = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
            for wr_name, wr_object in tqdm(wrappers.items(), desc="Wrapper loop"):
                score, features = train_classifier(data_tuple, classifier, wr_object)
                results[wr_name][model_name] = score
                results["selected_features"][model_name][wr_name].append(features)     
        f1_scores = dict(list(results.items())[:3])
        f1_scores_df = pd.DataFrame.from_dict(f1_scores, orient='index')
        f1_scores_df.to_csv(os.path.join(save_path, f"f1_scores_{case}_seed_{seed}.csv"), index=True)   
        wrapper_model_df = pd.DataFrame({model: {method: ", ".join(features[0]) for method, features in methods.items()}
                    for model, methods in results["selected_features"].items()})
        wrapper_model_df.to_csv(os.path.join(save_path, f"wrapper_model_{case}_seed_{seed}.csv"), index=True)
        print(f"Files for case: {case} seed: {seed}, saved successfully!")