### TODO:
- Push code to github
- Train baseline classifiers without the feature selection
- Create a .py and .ps1 file that will perform feature selection using embedded methods. Most probably Lasso and Ridge regularization.
- Create a .py file that read the .csv files from the feature selection script
- See if somewhow the thesis_src can be recognized as a package without appending the path to the sys.path
- Extract the test data from the original data
- Perform the feature selection and test the model's f1_macro not on a stratisfied validation set but on the bigger test set

### IN PROGRESS
### DONE `✓`
-  Performed Unsupservised Learning for the Transformer model
- Added a classification header for every case and saved the model
- Trained each header for three different seeds and saved a checkpoint for each
- Extracted the prediction for the validation set that the voter uses to select a percentile threshold for each header and seed
- Create a vector of statistical features for each predicted probabilities dataset. The feature are the mean, median, var, std, min, max, range, skewness, kurtosis, entropy and coefficient of variation.
- Performed statistical analysis of the features to check for linearity, non-linearity and relevance
- Performed feature selection using filter methods. These include Variance threshold, KBest with f-regression, KBest with mutual information and 50-percentile approach. The number of features selected is 5 from total 11 features.
- Performed feature selection using wrapper methods. These include Recursive Feature Elimination with Cross-Validation and Sequential Feature Selection (Backward and Forward with Cross-Validation). The number of features selected is 5 from total 11 features. The split is stratified to take into account the imbalance of the dataset.
- Trained each model on the selected features and saved the F1-macro score for each model, wrapper, case and seed.