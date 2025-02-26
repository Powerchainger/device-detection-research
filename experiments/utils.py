# Imports
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.feature_selection import  VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, SelectPercentile


from scipy.stats import skew, kurtosis, entropy, pearsonr, spearmanr

# Create dataset
'''
Read and export a tabular form of the data

Parameters:
path: str
file: str

Returns:
data as DataFrame
labels as Series
'''
def create_data(path, file):
    rows, labels = [], []
    with open(path + '\\' + file) as f:
        data = json.load(f)
        # Create DataFrame from first list of values
        for _, v in data.items():
            features = create_features(v)
            rows.append(features)
            labels.append(v[1][0])
    
    return pd.DataFrame(rows), pd.Series(labels)

# Create features
def create_features(data):
    '''
    Create features from the data
    These include statistical features such as mean, median, variance, etc.
    
    Parameters:
    data: list
    
    Returns:
    dictionary of features
    '''
    
    mean = np.mean(data[0])
    median = np.median(data[0])
    variance = np.var(data[0])
    std_dev = np.std(data[0])
    range_val = np.max(data[0]) - np.min(data[0])
    min_val = np.min(data[0])
    max_val = np.max(data[0])
    # Skewness and Kurtosis
    skewness = skew(data[0])
    kurt = kurtosis(data[0])

    # Entropy (Shannon Entropy)
    entropy_val = entropy(data[0])

    # Coefficient of Variation
    cv = std_dev / mean
    
    return {"mean": mean,"median": median, "variance": variance,"std_dev": std_dev,
            "range": range_val, "min": min_val, "max": max_val, "skewness": skewness,
            "kurtosis": kurt, "entropy": entropy_val,"cv": cv}
    

def feature_scores(X, y, seed):
    '''
    Calculate statistical scores for each feature in the dataset
    These can be used to search for feature dependencies and correlations 
    as well as outliers and other statistical properties of the dataset
    
    Parameters:
    X: DataFrame
    y: Series
    
    Returns:
    pearson_dict: Dictionary of Pearson Correlation Coefficients
    spearman_dict: Dictionary of Spearman Correlation Coefficients
    mi_dict: Dictionary of Mutual Information Scores
    '''
    pearson_correlations = {}
    spearman_correlations = {}
    skewness_dict = {}
    kurtosis_dict = {}
        
    for feature in X.columns:
        corr, _ = pearsonr(X[feature], y)
        pearson_correlations[feature] = corr
        corr, _ = spearmanr(X[feature], y)
        spearman_correlations[feature] = corr
        skewness_dict[feature] = skew(X[feature])
        kurtosis_dict[feature] = kurtosis(X[feature])
    
    mi_scores = mutual_info_classif(X, y, random_state=seed)   
    mi_dict = dict(zip(X.columns, mi_scores))  
    mi_median = np.median(list(mi_dict.values()))
    mi_threshold = mi_median
    
    linear_features = []
    nonlinear_features = []
    relevant_features = []
    irrelevant_features = []
    
    for feature in X.columns:
        # Identify Linear Features
        if abs(pearson_correlations[feature]) >= 0.5:
            linear_features.append(feature)

        # Identify Non-Linear Features
        elif abs(pearson_correlations[feature]) < 0.3 and mi_dict[feature] > 0.1:
            nonlinear_features.append(feature)
        
        # Identify Irrelevant Features
        if (abs(pearson_correlations[feature]) < 0.15 and abs(spearman_correlations[feature]) < 0.15) or (mi_dict[feature] < mi_median /2):
            irrelevant_features.append(feature)
            
        if (abs(pearson_correlations[feature]) >= 0.3 and abs(spearman_correlations[feature]) >= 0.3) and (mi_dict[feature] >= mi_threshold):
            relevant_features.append(feature)
    
    return {
        "linear_features": linear_features,
        "nonlinear_features": nonlinear_features,
        "relevant_features": relevant_features,
        "irrelevant_features": irrelevant_features,
        "pearson_sorted": sorted(pearson_correlations.items(), key=lambda x: abs(x[1]), reverse=True),
        "spearman_sorted": sorted(spearman_correlations.items(), key=lambda x: abs(x[1]), reverse=True),
        "mi_sorted": sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    }
    
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
    variance_selector = VarianceThreshold(threshold=0.1)
    kbest_selector = SelectKBest(f_classif, k=num_of_features)
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=num_of_features)
    percentile_selector = SelectPercentile(score_func=f_classif, percentile=50) # I don't know about this one
    
    # Fit the selectors
    variance_selector.fit_transform(X) # Univariate approach so no need for relationship with the target
    kbest_selector.fit_transform(X, y)
    mi_selector.fit_transform(X, y)
    percentile_selector.fit_transform(X, y)
    
    #Select the features
    variance_features = X.columns[variance_selector.get_support()].tolist()
    kbest_features = X.columns[kbest_selector.get_support()].tolist()
    mi_features = X.columns[mi_selector.get_support()].tolist()
    percentile_features = X.columns[percentile_selector.get_support()].tolist() 
    
    return {
        "variance_features": variance_features,
        "kbest_features": kbest_features,
        "mi_features": mi_features,
        "percentile_features": percentile_features
    }

def train_classifier(data, classifier, wrapper):
    '''
    Train a classifier based on a wrapper feature selection method
    
    Parameters:
    data: tuple (X_train, X_test, y_train, y_test)
    classifier: object e.g. SVC()
    wrapper: object e.g. RFECV()
    
    Returns:
    f1_score: float (macro)
    selected_features: list (#NUM selected features)
    '''
    
    wrapper.fit(data[0], data[2])
    selected_features = data[0].columns[wrapper.get_support()]
    classifier.fit(data[0][selected_features], data[2])
    y_pred = classifier.predict(data[1][selected_features])
    return f1_score(data[3], y_pred, average="macro"), list(selected_features)

def plot_cm_matrix(true_labels, predictions, clf_name):
    '''
    Plot the confusion matrix
    
    Parameters:
    true_labels: Series
    predictions: Series
    
    '''
    cm = confusion_matrix(true_labels, predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_name.classes_)
    cm_display.plot()
    plt.show()
    
def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")
    return path