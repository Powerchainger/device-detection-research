import argparse
import logging
import sys

from config.config import TRAINING_PARAMS
from models.unsupervised import run_pretraining
from models.supervised import run_training
from features.extract import extract
from features.filter_methods import run_filter_methods
from features.wrapper_methods import run_wrapper_methods
from features.classifier_baselines import run_classifier_baselines
from features.best_for_case import run_find_best_for_case
from features.train_selection import run_train_selection
from models.inference import run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all(input_path, data_name):
    logger.info(">>> Running full pipeline...")
    if input_path and data_name:
        run_pretraining(input_path, data_name)
        run_training(input_path, data_name)
        extract(input_path, data_name)
        run_filter_methods(input_path, data_name)
        run_wrapper_methods(input_path, data_name)
        run_classifier_baselines(input_path, data_name)
        run_find_best_for_case()
        extract(input_path, data_name, mode='test')
        run_train_selection()
        logger.info(">>> Full pipeline completed.")
    else:
        run_pretraining()
        run_training()
        extract()
        run_filter_methods()
        run_wrapper_methods()
        run_classifier_baselines()
        run_find_best_for_case()
        extract(mode='test')
        run_train_selection()
        logger.info(">>> Full pipeline completed without input parameters.")
        
def run_new_device(input_path, data_name):
    logger.info(">>> Fine tuning and selecting best classification approach for new device...")
    if input_path and data_name:
        run_training(input_path, data_name)
        extract(input_path, data_name)
        run_filter_methods(input_path, data_name)
        run_wrapper_methods(input_path, data_name)
        run_classifier_baselines(input_path, data_name)
        run_find_best_for_case()
        extract(input_path, data_name, mode='test')
        run_train_selection()
        logger.info(">>> New device pipeline completed.")
    else:
        logger.error("Input path and data name are required for new device inference.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run specific steps of the ML pipeline.")
    
    parser.add_argument(
        '--step',
        type=str,
        required=True,
        choices=[
            "pretraining", "training", "extract_validation", "filter",
            "wrapper", "baselines", "best_case", "extract_test",
            "train_selection", "inference", "all", "new_device"
        ],
        help="Choose which pipeline step to run. Possible choices: pretraining, training, extract_validation, filter, wrapper, baselines, best_case, extract_test, train_selection, inference, all, new_device"
    )
    parser.add_argument(
    '--input_path',
    type=str,
    required=False,
    help="Root folder where input data is located (expects f'{input_path}/Inputs/{data_name}.csv' inside)"
    )

    parser.add_argument(
        '--data_name',
        type=str,
        required=False,
        help="Name of the house CSV file (e.g., 'x_residential_25728')"
    )
    
    parser.add_argument(
        '--window_size',
        type=int,
        required=False,
        help="Length of processing windows (length of 1 translates to 30 minutes)"
    )
    args = parser.parse_args()
    step = args.step
    input_path = args.input_path
    data_name = args.data_name
    window_size = args.window_size
    
    if window_size:
        TRAINING_PARAMS['window_size'] = window_size
        logger.info(f"Window size overridden to {window_size}")
    
    try:
        if step == "pretraining":
            if not data_name:
                run_pretraining()
            else:
                run_pretraining(input_path, data_name)
        elif step == "training":
            if not data_name:
                run_training()
            else:
                run_training(input_path, data_name)
        elif step == "extract_validation":
            if not data_name:
                extract()
            else:
                extract(input_path, data_name)
        elif step == "filter":
            run_filter_methods()
        elif step == "wrapper":
            run_wrapper_methods()
        elif step == "baselines":
            run_classifier_baselines()
        elif step == "best_case":
            run_find_best_for_case()
        elif step == "extract_test":
            if not data_name:
                extract(mode='test')
            else:
                extract(input_path, data_name, mode='test')
        elif step == "train_selection":
            run_train_selection()
        elif step == "all":
            run_all(input_path, data_name)
        elif step == "new_device":
            run_new_device(input_path, data_name)
        elif step == "inference":
            if not data_name or not input_path:
                parser.error("--data_name and --input_path are required for inference.")
            else:
                run_inference(input_path, data_name)

        else:
            logger.error(f"Unknown step: {step}")
            parser.print_help()
        
    except Exception as e:
        logger.error(f"Error while running step '{step}': {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()