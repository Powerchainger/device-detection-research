import json, os, re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score
from sklearn.feature_selection import  SelectKBest, mutual_info_classif, f_classif

from scipy.stats import skew, kurtosis, entropy

from config.config import CHECKPOINT_DIR

def read_data(path, file):
    '''
    Read json data
    
    Parameters:
    path: directory in string format
    file: file name in string format
    '''
    with open(path + '\\' + file) as f:
         return json.load(f)
     
def create_data(data, filter=False):
    '''
    JSON data is converted to a DataFrame
    If filter is True, the data is processed to create features
    
    Parameters:
    data: dictionary (JSON data)
    filter: boolean
    '''       
    rows, labels = [], []
    return_labels = False

    for v in data.values():
        has_label = (
        isinstance(v, list)
        and len(v) == 2
        and isinstance(v[0], list)
        and isinstance(v[1], list)
    )
        return_labels = return_labels or has_label

        if filter:
            row = create_features(v)  
            # row = create_features(v[0] if has_label else v, has_label)
        else:
            row = sorted(v[0] if has_label else v)
        rows.append(row)

        if has_label:
            labels.append(v[1][0])

    df = pd.DataFrame(rows)
    return (df, pd.Series(labels)) if return_labels else df
    


# Create features
def create_features(data, has_label=True):
    '''
    Create features from the data
    These include statistical features such as mean, median, variance, etc.
    
    Parameters:
    data: list
    
    Returns:
    dictionary of features
    '''
    if has_label:
        mean = np.mean(data[0])
        median = np.median(data[0])
        variance = np.var(data[0])
        std_dev = np.std(data[0])
        range_val = np.max(data[0]) - np.min(data[0])
        min_val = np.min(data[0])
        max_val = np.max(data[0])
        entropy_val = entropy(data[0])
        skewness = skew(data[0], bias=False) if np.std(data[0]) > 1e-8 else 0.0
        kurt = kurtosis(data[0], bias=False) if np.std(data[0]) > 1e-8 else 0.0

    else:
        mean = np.mean(data)
        median = np.median(data)
        variance = np.var(data)
        std_dev = np.std(data)
        range_val = np.max(data) - np.min(data)
        min_val = np.min(data)
        max_val = np.max(data)
        entropy_val = entropy(data)
        skewness = skew(data, bias=False) if np.std(data) > 1e-8 else 0.0
        kurt = kurtosis(data, bias=False) if np.std(data) > 1e-8 else 0.0

    cv = std_dev / mean

    return {"mean": mean,"median": median, "variance": variance,"std_dev": std_dev,
            "range": range_val, "min": min_val, "max": max_val, "skewness": skewness,
            "kurtosis": kurt, "entropy": entropy_val,"cv": cv}


def select_features(X, y, num_of_features=5):
    '''
    Use sklearn's feature selection methods to select the best features for the dataset
    These are filter methods for feature selection
    
    Parameters:
    X: DataFrame
    y: Series
    num_of_features: int
    
    Returns:
    dictionary of selected features for each method
    '''
    
    # Create feature selection objects

    kbest_selector = SelectKBest(score_func=f_classif, k=num_of_features)
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=num_of_features)
    
    # Fit the selectors
    kbest_selector.fit_transform(X, y)
    mi_selector.fit_transform(X, y)
    
    #Select the features
    kbest_features = X.columns[kbest_selector.get_support()].tolist()
    mi_features = X.columns[mi_selector.get_support()].tolist()
    
    return {
        "kbest_features": kbest_features,
        "mi_features": mi_features,
    }

def train_classifier(data, wrapper, **kwargs):  
    '''
    Train a classifier using a wrapper feature selection method.

    Parameters:
    - data: tuple (X_train, X_test, y_train, y_test)
    - wrapper: feature selection algorithm (e.g. RFECV instance)

    Returns:
    - performance: float (macro F1-score)
    - selected_features: list of selected feature names
    - trained_wrapper: the wrapper object after fitting #If cv option is applied
    '''
    # X_train, X_test, y_train, y_test = data
    # wrapper.fit(X_train, y_train)
    # selected_features = X_train.columns[wrapper.get_support()].to_list()
    # if hasattr(wrapper, "predict"):  # For RFECV or similar wrappers with predict()
    #     y_pred = wrapper.predict(X_test)
    # else:  # For SequentialFeatureSelector, etc.
    #     classifier = kwargs.get("classifier")
    #     classifier.fit(X_train[selected_features], y_train)
    #     y_pred = classifier.predict(X_test[selected_features])
    # performance = f1_score(y_test, y_pred, average="macro")
    # return performance, selected_features, wrapper
    wrapper.fit(data[0], data[2])
    selected_features = data[0].columns[wrapper.get_support()].to_list()
    classifier = kwargs.get("classifier")
    classifier.fit(data[0][selected_features], data[2])
    y_pred = classifier.predict(data[1][selected_features])
    performance = f1_score(data[3], y_pred, average="macro")
    return performance, selected_features

    
def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")
    return path

def extract_model_data(log, start_pos, model_pattern, accuracy_pattern, f1_score_pattern):
    models = re.findall(model_pattern, log[start_pos:])
    accuracies = re.findall(accuracy_pattern, log[start_pos:])
    f1_scores = re.findall(f1_score_pattern, log[start_pos:])
    return models, accuracies, f1_scores

def load_data(case, apply_filter):
    """Helper function to read validation & test data"""
    valid_file = f"{case}_valid_probs.json"
    test_file = f"{case}_test_probs.json"

    data_valid = read_data(str(CHECKPOINT_DIR) + "\\valid_probs", valid_file)
    data_test = read_data(str(CHECKPOINT_DIR) + "\\test_probs", test_file)

    return create_data(data_valid, filter=apply_filter), create_data(data_test, filter=apply_filter)

def train_and_evaluate(clf, validation_set, test_set, selected_features=None):
    """Train classifier and compute F1-score."""
    X_train, y_train = validation_set
    X_test, y_test = test_set

    if selected_features:
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return f1_score(y_test, preds, average='macro')

def plot_cv_results(sfs1, sfs2, rfecv):
    '''
    Plot the cv results for every step of the SFS and RFECV feature selection methods.'''
    # Separate means and stds
    features_fwd = np.array(list(sfs1.scores.keys()), dtype=int)
    means_fwd, stds_fwd = np.array([(v[0], v[1]) for v in sfs1.scores.values()]).T
    features_bwd = np.array(list(sfs2.scores.keys()), dtype=int)[::-1] # Reverse order for backward selection
    means_bwd, stds_bwd = np.array([(v[0], v[1]) for v in sfs2.scores.values()]).T
    cv_results = pd.DataFrame(rfecv.cv_results_)
    features_rfecv = cv_results["n_features"]
    means_rfecv = cv_results["mean_test_score"]
    stds_rfecv = cv_results["std_test_score"]

    # Find best features and scores
    best_idx_fwd = np.argmax(means_fwd)
    best_feat_fwd = features_fwd[best_idx_fwd]
    best_score_fwd = means_fwd[best_idx_fwd]
    best_idx_bwd = np.argmax(means_bwd)
    best_feat_bwd = features_bwd[best_idx_bwd]
    best_score_bwd = means_bwd[best_idx_bwd]
    best_idx_rfecv = np.argmax(means_rfecv)
    best_feat_rfecv = features_rfecv[best_idx_rfecv]
    best_score_rfecv = means_rfecv[best_idx_rfecv]

    plt.figure(figsize=(10, 6))
    plt.plot(features_fwd, means_fwd, marker='o', label='SFS Forward', color='dodgerblue')
    plt.fill_between(features_fwd, means_fwd - stds_fwd, means_fwd + stds_fwd, alpha=0.2, color='dodgerblue')

    plt.plot(features_bwd, means_bwd, marker='o', label='SFS Backward', color='orange')
    plt.fill_between(features_bwd, means_bwd - stds_bwd, means_bwd + stds_bwd, alpha=0.2, color='orange')

    plt.plot(features_rfecv, means_rfecv, marker='o', label='RFECV', color='crimson')
    plt.fill_between(features_rfecv, means_rfecv - stds_rfecv, means_rfecv + stds_rfecv, alpha=0.2, color='crimson')

    # Vertical lines for best points
    plt.axvline(best_feat_fwd, color='blue', linestyle='--', label=f'Best SFS Forward ({best_feat_fwd}, {best_score_fwd:.2f})')
    plt.axvline(best_feat_bwd, color='darkorange', linestyle='--', label=f'Best SFS Backward ({best_feat_bwd}, {best_score_bwd:.2f})')
    plt.axvline(best_feat_rfecv, color='darkred', linestyle='--', label=f'Best RFECV ({best_feat_rfecv}, {best_score_rfecv:.2f})')

    plt.xlabel("Number of Features Selected")
    plt.ylabel("F1 Macro Score")
    plt.ylim(0.5, 1)
    plt.title("Feature Selection Comparison: SFS vs RFECV (SVM)")
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()
