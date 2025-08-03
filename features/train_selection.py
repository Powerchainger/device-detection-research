import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config.config import CHECKPOINT_DIR
from utils.feature_utils import load_data, train_and_evaluate

classifiers = {
        "SVM": SVC(kernel="linear", class_weight='balanced', probability=True),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }

def voter_function(test_set, quantile=None):
    '''
    Function to apply quantile voting on the predictions from the test set.
    Args:
        test_set: A tuple containing the predictions and true labels.
        quantile: The quantile to use for voting. If None, defaults to 0.5 (median).
    Returns:
        f1: The F1 score of the quantile voting predictions against the true labels.
    '''
    
    window_predictions = test_set[0]  # Assuming test_set[0] contains the predictions
    labels = test_set[1]  # Assuming test_set[1] contains the true labels
    
    q_preds = window_predictions.quantile(q=quantile if quantile is not None else 0.5, axis=1)
    predictions = np.rint(q_preds).astype(int)
    f1 = f1_score(labels, predictions, average='macro')
    return f1


def run_train_selection():
    case_results = pd.read_csv(CHECKPOINT_DIR / "case_best" / "results.csv")
    best_config_case = {}

    for _, row in case_results.iterrows():
        print(f"\n=== Processing {row['Case']} ===")
        case_option = row["Case"]
        classifier_name = row["Classifier"]
        method_tag = row["Method_Tag"]
        feature_selection = row["Feature_Selection"]
        quantile = row["Quantile"]

        apply_filter = method_tag not in ['raw', 'voter']
        validation_set, test_set = load_data(case_option, apply_filter)
        selected_features = None
        
        if method_tag != 'voter':
            clf = classifiers[classifier_name]
            
            if method_tag in ['filter', 'wrapper']:
                feature_file = CHECKPOINT_DIR / method_tag / f'selected_features_{case_option}.csv'
                df = pd.read_csv(feature_file, index_col=0)
                key = 'Selected_Features' if method_tag == 'filter' else classifier_name
                selected_features = df.at[feature_selection, key]
                selected_features = [feat.strip() for feat in selected_features.split(',')]
                f1_macro = train_and_evaluate(clf, validation_set, test_set, selected_features)
            else:
                f1_macro = train_and_evaluate(clf, validation_set, test_set)
        # === Save trained model ===                
            model_path = CHECKPOINT_DIR / "trained_voters" / f"{case_option}_{classifier_name}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, model_path)
            
            if apply_filter:
                filename = f"{case_option}_{classifier_name}_{'features' if selected_features else 'all_features'}.csv"
                features_to_save = selected_features if selected_features else validation_set[0].columns.tolist()
                feat_path = CHECKPOINT_DIR / "trained_voters" / filename
                pd.Series(features_to_save).to_csv(feat_path, index=False)
        else:
            # Voter case
            f1_macro = voter_function(test_set, quantile)
            
        # === THIS CODE IS TO GET A PREVIEW OF THE SCORES FOR EACH CASE ===    
        best_config_case = {
                "Case": case_option,
                "Classifier": classifier_name,
                "Method_Tag": method_tag,
                "Feature_Selection": feature_selection if method_tag in ['filter', 'wrapper'] else None,
                "Selected_Features": selected_features,
                "F1_Score": f1_macro,
                "Quantile": quantile if method_tag == 'voter' else None
            }
        print("\n=== Best Configuration ===")
        for key, value in best_config_case.items():
            print(f"{key}: {value}")
        # =================================================================


