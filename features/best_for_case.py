import pandas as pd
import os, json

from utils.feature_utils import create_dir
from config.config import CHECKPOINT_DIR, CASES

return_dict = {}
case_dfs = []

def run_find_best_for_case():
    for case in CASES:
        df_raw = pd.read_csv(str(CHECKPOINT_DIR) + "\\classifier_baselines\\" + f'{case}_probs.csv', index_col=0)
        df_features = pd.read_csv(str(CHECKPOINT_DIR) + "\\classifier_baselines\\" + f'{case}_features.csv', index_col=0)
        df_filter = pd.read_csv(str(CHECKPOINT_DIR) + "\\filter\\" + f'f1_score_{case}.csv', index_col=0)
        df_wrapper = pd.read_csv(str(CHECKPOINT_DIR) + "\\wrapper\\" + f'f1_scores_{case}.csv', index_col=0)
        
        best_model_raw = df_raw.idxmax()[0]
        best_score_raw = df_raw.max()[0]
        
        best_model_features = df_features.idxmax()[0]
        best_score_features = df_features.max()[0]
        
        best_model_filter = df_filter.max().idxmax()
        best_score_filter = df_filter.max().max()
        best_approach_filter = df_filter.idxmax().loc[best_model_features]
        
        best_model_wrapper = df_wrapper.max().idxmax()
        best_score_wrapper = df_wrapper.max().max()
        best_approach_wrapper = df_wrapper.idxmax().loc[best_model_features]
        
        with open(str(CHECKPOINT_DIR) + "\\voter_metrics\\" + f'{case}.json', 'r') as f:
            voter_metrics = json.load(f)
        
        results = {'raw': {'model': best_model_raw, 
                        'score': best_score_raw},
                    'features': {'model': best_model_features,
                                'score': best_score_features},
                    'filter': {'model': best_model_filter, 
                            'approach': best_approach_filter, 
                            'score': best_score_filter},
                    'wrapper': {'model': best_model_wrapper, 
                                'approach': best_approach_wrapper, 
                                'score': best_score_wrapper},
                    'voter': {'quantile': voter_metrics['quantile'],
                            'score': voter_metrics['F1_SCORE_MACRO']}
                    }
        
        best_key = max(results, key=lambda k: results[k]["score"])  # Get the key with the max score
        best_entry = results[best_key]  # Extract the corresponding dictionary
        
        if best_key == 'voter':
            return_dict[case] = {best_key: {
                    'Classifier': f'Voter (q={best_entry["quantile"]})',
                    'F1_Score': best_entry['score'],
                    'Feature_Selection': None,
                    'Quantile': best_entry['quantile']
                    }
            }
        elif 'approach' in best_entry:
            return_dict[case] = {best_key: {
                    'Classifier': best_entry['model'],
                    'F1_Score': best_entry['score'],
                    'Feature_Selection': best_entry['approach'],
                    'Quantile': None
                    }
            }
        else:
            return_dict[case] = {best_key: {
                    'Classifier': best_entry['model'],
                    'F1_Score': best_entry['score'],
                    'Feature_Selection': None,
                    'Quantile': None
                    }
            }

    rows = []
    for case, methods in return_dict.items():
        for method_tag, info in methods.items():
            rows.append({
                'Case': case,
                'Method_Tag': method_tag,
                'Classifier': info['Classifier'],
                'F1_Score': info['F1_Score'],
                'Feature_Selection': info['Feature_Selection'],
                'Quantile': info['Quantile']
            })

    df_results = pd.DataFrame(rows)
    _ = create_dir(str(CHECKPOINT_DIR) + "\\case_best")
    filedir = os.path.join(_, "results.csv")
    df_results.to_csv(filedir, index=False)