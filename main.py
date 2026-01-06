import json
import pandas as pd 
import numpy as np 
import sys 
import os 

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))

from Logging.logger import get_logger 
from Source.Data.load_data import DataLoader
from Source.Data.clean_data import DataCleaner
from Source.Models.train_prophet import ProphetTrainer
from Source.Evalution.evalution import interactive_evalution

class SCMPipeline:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, 'Source', 'config.json')
        self.logger = get_logger("Main-Logger")
        self.config = self._load_config()
        self.state_model_configs = self._get_state_model_configs()

    def _load_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _get_state_model_configs(self):
        """Returns the configuration for Prophet Model for each state."""
        return {
            'Gujarat': {
                'changepoint_prior_scale': 0.1,
                'seasonality_mode': 'multiplicative',
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'interval_width': 0.80,
            },
            'Maharashtra': {
                'changepoint_prior_scale': 0.1,
                'seasonality_mode': 'multiplicative',
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'interval_width': 0.80,
            },
            'Chattisgarh': {
                'changepoint_prior_scale': 0.1,
                'seasonality_mode': 'multiplicative',
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'interval_width': 0.80,
            },
            'TamilNadu': {
                'changepoint_prior_scale': 0.1,
                'seasonality_mode': 'multiplicative',
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'interval_width': 0.80,
            }
        }

    def process_state(self, state_name, data):
        """Train model and evaluate for a specific state."""
        self.logger.info(f"{state_name} Started......")
        try:
            model_params = self.state_model_configs.get(state_name)
            if not model_params:
                self.logger.warning(f"No specific config found for {state_name}, using defaults in Trainer.")
            
            trainer = ProphetTrainer(state_name, self.config_path, model_params)
            prophet_data, forecast_future, prophet_data_pred = trainer.run(data)
            
            interactive_evalution(prophet_data, forecast_future, prophet_data_pred, state_name)
            self.logger.info(f"{state_name} Completed ")
        except Exception as e:
            self.logger.error(f"Error processing {state_name}: {e}", exc_info=True)

    def run(self):
        """Orchestrates the entire SCM pipeline."""
        self.logger.info("Main pipeline started.")
        
        # 1. Load Data
        self.logger.info("Loading Data...")
        data_loader = DataLoader(self.config_path)
        raw_data = data_loader.load_raw_data()
        self.logger.info("Data Loading Completed.")

        # 2. Clean Data
        self.logger.info("Cleaning Data...")
        cleaner = DataCleaner(self.config_path)
        # clean_data returns: data_GJ, data_CG, data_MH, data_TN, data
        data_GJ, data_CG, data_MH, data_TN, data_all = cleaner.process_all(raw_data)
        self.logger.info("Cleaning Completed.")

        # 3. Process States
        self.logger.info("Processing States...")
        self.process_state('Gujarat', data_GJ)
        self.process_state('Maharashtra', data_MH)
        self.process_state('Chattisgarh', data_CG)
        self.process_state('TamilNadu', data_TN)
        
        self.logger.info("Main pipeline finished successfully.")

if __name__ == "__main__":
    pipeline = SCMPipeline()
    pipeline.run()
