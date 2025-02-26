# Imports
import os, sys
from pathlib import Path
from experiments.utils import create_data, feature_scores


root = str(Path(os.getcwd()).resolve().parents[0])
path = root + '\\extracted_probabilities_seed'
case = sys.argv[1]
seeds = range(0,3)
    
if __name__ == '__main__':
    ovrl_case = {}
    for seed in seeds:
        file = case + '_validation_seed_' + str(seed) + '.json'
        data, labels = create_data(path, file)
        results = feature_scores(data, labels, seed)
        ovrl_case[case + '_' + str(seed)] = results

    for k, v in ovrl_case.items():
        print.info(f"{k} - Linear Features: {v['linear_features']}")
        print.info(f"{k} - Non-Linear Features: {v['nonlinear_features']}")
        print.info(f"{k} - Relevant Features: {v['relevant_features']}")
        print.info("==========================================================================================")