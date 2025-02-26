#Imports
from pathlib import Path
import os, sys
from experiments.utils import create_data, select_features


root = str(Path(os.getcwd()).resolve().parents[0])
path = root + '\\extracted_probabilities_seed'
case = sys.argv[1]
seeds = range(0,3)

if __name__ == '__main__':
    ovrl_case = {}
    for seed in seeds:
        file = case + '_validation_seed_' + str(seed) + '.json'
        data, labels = create_data(path, file)
        results = select_features(data, labels)
        ovrl_case[case + '_' + str(seed)] = results
    
    for k, v in ovrl_case.items():
        print(f"{k} - Variance Experiment Features: {v['variance_features']}")
        print(f"{k} - K-best Features: {v['kbest_features']}")
        print(f"{k} - Mutual Information Features: {v['mi_features']}")
        print(f"{k} - 50-percentile Features: {v['percentile_features']}")
        print("==========================================================================================")
